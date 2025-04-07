#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring> // For strerror
#include <sys/time.h>
#include <thread>
#include <omp.h>
#include <unistd.h>
#include "kseq++/kseq++.hpp"

#include "FastxStream.h"
#include "FastxChunk.h"
#include "DataQueue.h"
#include "Formater.h"

#include "index.hpp"
#include "indexparameters.hpp"
#include "cmdline.hpp"
#include "exceptions.hpp"
#include "io.hpp"
#include "randstrobes.hpp"
#include "refs.hpp"
#include "logger.hpp"
#include "pc.hpp"
#include "readlen.hpp"
#include "my_struct.hpp"

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define my_bucket_index_t StrobemerIndex::bucket_index_t

#define rescue_threshold 100

inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}

InputBuffer get_input_buffer(const CommandLineOptions &opt) {
    if (opt.is_SE) {
        return InputBuffer(opt.reads_filename1, "", opt.chunk_size, false);

    } else if (opt.is_interleaved) {
        if (opt.reads_filename2 != "") {
            throw BadParameter("Cannot specify both --interleaved and specify two read files");

        }
        return InputBuffer(opt.reads_filename1, "", opt.chunk_size, true);

    } else {
        return InputBuffer(opt.reads_filename1, opt.reads_filename2, opt.chunk_size, false);

    }

}

int producer_pe_fastq_task(std::string file, std::string file2, rabbit::fq::FastqDataPool &fastqPool,
                           rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> &dq) {
    rabbit::fq::FastqFileReader *fqFileReader;
    fqFileReader = new rabbit::fq::FastqFileReader(file, fastqPool, false, file2);
    int n_chunks = 0;
    int line_sum = 0;
    while (true) {
        rabbit::fq::FastqDataPairChunk *fqdatachunk = new rabbit::fq::FastqDataPairChunk;
        fqdatachunk = fqFileReader->readNextPairChunk();
        if (fqdatachunk == NULL) break;
        //std::cout << "readed chunk: " << n_chunks << std::endl;
        dq.Push(n_chunks, fqdatachunk);
        n_chunks++;
    }

    dq.SetCompleted();
    delete fqFileReader;
    std::cerr << "file " << file << " has " << n_chunks << " chunks" << std::endl;
    return 0;
}


__device__ inline randstrobe_hash_t gpu_get_hash(const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, my_bucket_index_t position) {
    if (position < d_randstrobes_size) {
        return d_randstrobes[position].hash;
    } else {
        return static_cast<randstrobe_hash_t>(-1);
    }
}

__device__ inline bool gpu_is_filtered(const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, my_bucket_index_t position,
                unsigned int filter_cutoff) {
    return gpu_get_hash(d_randstrobes, d_randstrobes_size, position) ==
           gpu_get_hash(d_randstrobes, d_randstrobes_size, position + filter_cutoff);
}

__device__ int gpu_get_count(
        const RefRandstrobe *d_randstrobes,
        const my_bucket_index_t *d_randstrobe_start_indices,
        my_bucket_index_t position,
        int bits
) {
    const auto key = d_randstrobes[position].hash;
    const unsigned int top_N = key >> (64 - bits);
    int64_t position_end = d_randstrobe_start_indices[top_N + 1];
    int64_t position_start = position;

    if(position_end == 0) return 0;
    int64_t low = position_start, high = position_end - 1, ans = 0;
    while (low <= high) {
        int64_t mid = (low + high) / 2;
        if (d_randstrobes[mid].hash == key) {
            low = mid + 1;
            ans = mid;
        } else {
            high = mid - 1;
        }
    }
    // int count = 0;
    // for (auto i = position_start; i < position_end; ++i) {
    //     if (d_randstrobes[i].hash == key){
    //         count += 1;
    //     } else break;
    // }
    // if(count != ans - position_start + 1) {
    //     // print d_randstrobes[position_start - position_end]
    //     // for(int i = position_start; i < position_end; i++) {
    //     //     printf("hash %llu ", d_randstrobes[i].hash);
    //     // }
    //     // printf("\n");
    //     printf("count %d %llu\n", count, ans - position_start + 1);
    // }

    return ans - position_start + 1;
}

// __device__ int gpu_get_count(
//         const RefRandstrobe *d_randstrobes,
//         const my_bucket_index_t *d_randstrobe_start_indices,
//         my_bucket_index_t position,
//         int bits

// )  {
//     const auto key = d_randstrobes[position].hash;
//     const unsigned int top_N = key >> (64 - bits);
//     my_bucket_index_t position_end = d_randstrobe_start_indices[top_N + 1];
//     // change the following for-loop to binary search

//     int count = 1;
//     for (my_bucket_index_t position_start = position + 1; position_start < position_end; ++position_start) {
//         if (d_randstrobes[position_start].hash == key){
//             count += 1;
//         } else break;
//     }
//     return count;
// }

__device__ inline size_t gpu_find(
        const RefRandstrobe *d_randstrobes,
        const my_bucket_index_t *d_randstrobe_start_indices,
        const randstrobe_hash_t key,
        int bits
) {
    const unsigned int top_N = key >> (64 - bits);
    my_bucket_index_t position_start = d_randstrobe_start_indices[top_N];
    my_bucket_index_t position_end = d_randstrobe_start_indices[top_N + 1];
    if(position_end - position_start < 64) {
        for (my_bucket_index_t i = position_start; i < position_end; ++i) {
            if (d_randstrobes[i].hash == key) {
                return i;
            }
        }
        return static_cast<size_t>(-1); // No match
    } else {
        my_bucket_index_t low = position_start, high = position_end;
        while (low < high) {
            my_bucket_index_t mid = low + (high - low) / 2;
            if (d_randstrobes[mid].hash < key) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if (d_randstrobes[low].hash == key) {
            return low;
        } else return static_cast<size_t>(-1); // No match
    }
}

template <typename T>
__device__ int partition(T* data, int low, int high) {
    T pivot = data[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (data[j] < pivot) {
            ++i;
            T temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    T temp = data[i + 1];
    data[i + 1] = data[high];
    data[high] = temp;

    return i + 1;
}

template <typename T>
__device__ void dfs_quick_sort(T* data, int low, int high) {
    if (low < high) {
        int pivot_index = partition(data, low, high);

        // Recursively sort elements before and after the pivot
        dfs_quick_sort(data, low, pivot_index - 1);
        dfs_quick_sort(data, pivot_index + 1, high);
    }
}

template <typename T>
struct DefaultCompare {
    __device__ __forceinline__
    bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

//template <typename T, typename Compare = DefaultCompare<T>>
//__device__ void quick_sort_iterative(
//        T* data,
//        int low,
//        int high,
//        Compare comp = DefaultCompare<T>()
//) {
////    static_assert(
////            __is_invocable_bool<Compare, T, T>::value,
////            "Invalid comparison function signature"
////    );
//
//    int stack[64];
//    int top = -1;
//
//    stack[++top] = low;
//    stack[++top] = high;
//
//    while (top >= 0) {
//        high = stack[top--];
//        low = stack[top--];
//
//        if (high - low < 32) {
//            insertion_sort(data, low, high, comp);
//            continue;
//        }
//
//        int mid = low + (high - low)/2;
//        if (comp(data[high], data[mid])) my_swap(data[mid], data[high]);
//        if (comp(data[high], data[low])) my_swap(data[low], data[high]);
//        if (comp(data[mid], data[low])) my_swap(data[low], data[mid]);
//
//        T pivot = data[mid];
//
//        int i = low - 1;
//        int j = high + 1;
//        while (true) {
//            do { ++i; } while (comp(data[i], pivot));
//            do { --j; } while (comp(pivot, data[j]));
//            if (i >= j) break;
//            my_swap(data[i], data[j]);
//        }
//        int p = j;
//
//        if (p - low < high - p) {
//            if (low < p) {
//                stack[++top] = low;
//                stack[++top] = p;
//            }
//            if (p+1 < high) {
//                stack[++top] = p+1;
//                stack[++top] = high;
//            }
//        } else {
//            if (p+1 < high) {
//                stack[++top] = p+1;
//                stack[++top] = high;
//            }
//            if (low < p) {
//                stack[++top] = low;
//                stack[++top] = p;
//            }
//        }
//    }
//}
//
//
//template <typename T, typename Compare>
//__device__ void insertion_sort(T* data, int low, int high, Compare comp) {
//    for (int i = low + 1; i <= high; ++i) {
//        T key = data[i];
//        int j = i - 1;
//
//#pragma unroll
//        while (j >= low && comp(key, data[j])) {
//            data[j + 1] = data[j];
//            --j;
//        }
//        data[j + 1] = key;
//    }
//}

template <typename T, typename Compare = DefaultCompare<T>>
__device__ void quick_sort_iterative(
        T* data,
        int low,
        int high,
        Compare comp = DefaultCompare<T>()
) {
    if (low > high) return;
    int vec_size = high - low + 1;
    my_vector<int>stack_vec(vec_size * 2);
    int* stack = stack_vec.data;
    int top = -1;
    stack[++top] = low;
    stack[++top] = high;
    while (top >= 0) {
        high = stack[top--];
        low = stack[top--];
        // Partition
        T pivot = data[high];
        int i = low - 1;
        for (int j = low; j < high; ++j) {
            //if (data[j] < pivot) {
            if (comp(data[j], pivot)) {
                ++i;
                T temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }
        T temp = data[i + 1];
        data[i + 1] = data[high];
        data[high] = temp;
        int pivot_index = i + 1;
        if (pivot_index - 1 > low) {
            stack[++top] = low;
            stack[++top] = pivot_index - 1;
        }
        if (pivot_index + 1 < high) {
            stack[++top] = pivot_index + 1;
            stack[++top] = high;
        }
    }
}


template <typename T>
__device__ void bubble_sort(T* data, int size) {
    for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - i - 1; ++j) {
            if (data[j + 1] < data[j]) {
                T temp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = temp;
            }
        }
    }
}

template <typename T>
__device__ void quick_sort(T* data, int size) {
    quick_sort_iterative(data, 0, size - 1);
//    bubble_sort(data, size);

}

struct Rescue_Seeds {
    int read_id;
    int read_fr;
    int seeds_num;
    QueryRandstrobe* seeds;
};

__device__ void print_nam(Nam nam) {
    printf("nam_id: %d, ref_id: %d, ref_start: %d, ref_end: %d, query_start: %d, query_end: %d, n_hits: %d, is_rc: %d\n",
           nam.nam_id, nam.ref_id, nam.ref_start, nam.ref_end, nam.query_start, nam.query_end, nam.n_hits, nam.is_rc);
}

__device__ void print_str(my_string str) {
    for(int i = 0; i < str.size(); i++) {
        printf("%c", str[i]);
    }
    printf("\n");
}

struct GPURead {
    char* seq;
    char* rc;
    int length;
    __device__ int size() const { return length; }
};

struct GPUAlignment {
    int ref_id;
    int ref_start;
    //my_vector<uint32_t> cigar;
    int edit_distance;
    int global_ed;
    int score;
    int length;
    bool is_rc;
    bool is_unaligned{false};
    // Whether a gapped alignment function was used to obtain this alignment
    // (even if true, the alignment can still be without gaps)
    bool gapped{false};
    //__device__ void move_to(GPUAlignment& dest) {
    //    dest.ref_id = ref_id;
    //    dest.ref_start = ref_start;
    //    //dest.cigar.move_from(cigar);
    //    dest.edit_distance = edit_distance;
    //    dest.global_ed = global_ed;
    //    dest.score = score;
    //    dest.length = length;
    //    dest.is_rc = is_rc;
    //    dest.is_unaligned = is_unaligned;
    //    dest.gapped = gapped;
    //}

};

struct GPUAlignTmpRes {
    int type;
    // type 0 : size1 == 0 size2 == 0, unmapped_pair
    // type 1 : size1 == 0, rescue read1
    // type 2 : size2 == 0, rescue read2
    // type 3 : good pair
    // type 4 : for loop
    int mapq1;
    int mapq2;
    int type4_loop_size;
    my_vector<bool> is_extend_seed;
    my_vector<bool> consistent_nam;
    my_vector<bool> is_read1;
    my_vector<Nam> type4_nams;
    my_vector<Nam> todo_nams;
    my_vector<bool> done_align;
    // if done_align, align_res is the alignment results
    my_vector<GPUAlignment> align_res;
    __device__ void init() {
        type = 0;
        mapq1 = 0;
        mapq2 = 0;
        type4_loop_size = 0;
    }
};

struct GPUReferences {
    my_vector<my_string> sequences;
    my_vector<size_t> lengths;
    int num_refs;
};

struct GPUAlignmentInfo {
    my_vector<uint32_t> cigar;
    unsigned int edit_distance{0};
    unsigned int ref_start{0};
    unsigned int ref_end{0};
    unsigned int query_start{0};
    unsigned int query_end{0};
    int sw_score{0};

    __device__ int ref_span() const { return ref_end - ref_start; }
};

__device__ void cigar_push(my_vector<uint32_t>& m_ops, uint8_t op, int len) {
    assert(op < 16);
    if (m_ops.empty() || (m_ops.back() & 0xf) != op) {
        m_ops.push_back(len << 4 | op);
    } else {
        m_ops.back() += len << 4;
    }
}

__device__ void my_hamming_align(const my_string &query, const my_string &ref, int match, int mismatch, int end_bonus, GPUAlignmentInfo& aln) {
    if (query.length() != ref.length()) {
        return;
    }
    size_t n = query.length();

    size_t start = 0; // start of the current segment
    int score = end_bonus; // accumulated score so far in the current segment

    size_t best_start = 0;
    size_t best_end = 0;
    int best_score = 0;
    for (size_t i = 0; i < n; ++i) {
        if (query[i] == ref[i]) {
            score += match;
        } else {
            score -= mismatch;
        }
        if (score < 0) {
            start = i + 1;
            score = 0;
        }
        if (score > best_score) {
            best_start = start;
            best_score = score;
            best_end = i + 1;
        }
    }
    if (score + end_bonus > best_score) {
        best_score = score + end_bonus;
        best_end = query.length();
        best_start = start;
    }

    size_t segment_start = best_start;
    size_t segment_end = best_end;
    score = best_score;

    if (segment_start > 0) {
        cigar_push(aln.cigar, CIGAR_SOFTCLIP, segment_start);
    }

    // Create CIGAR string and count mismatches
    int counter = 0;
    bool prev_is_match = false;
    int mismatches = 0;
    bool first = true;
    for (size_t i = segment_start; i < segment_end; i++) {
        bool is_match = query[i] == ref[i];
        mismatches += is_match ? 0 : 1;
        if (!first && is_match != prev_is_match) {
            cigar_push(aln.cigar, prev_is_match ? CIGAR_EQ : CIGAR_X, counter);
            counter = 0;
        }
        counter++;
        prev_is_match = is_match;
        first = false;
    }
    if (!first) {
        cigar_push(aln.cigar, prev_is_match ? CIGAR_EQ : CIGAR_X, counter);
    }

    int soft_right = query.length() - segment_end;
    if (soft_right > 0) {
        cigar_push(aln.cigar, CIGAR_SOFTCLIP, soft_right);
    }

    aln.sw_score = score;
    aln.edit_distance = mismatches;
    aln.ref_start = segment_start;
    aln.ref_end = segment_end;
    aln.query_start = segment_start;
    aln.query_end = segment_end;
    return;
}

__device__ bool extend_seed_part(
        GPUAlignTmpRes& align_tmp_res,
        const AlignmentParameters& aligner_parameters,
        const Nam& nam,
        const GPUReferences& references,
        const GPURead& read,
        bool consistent_nam
) {
    const my_string query(nam.is_rc ? read.rc : read.seq, read.length);
    const my_string ref = references.sequences[nam.ref_id];

    const auto projected_ref_start = my_max(0, nam.ref_start - nam.query_start);
    const auto projected_ref_end = my_min(nam.ref_end + query.size() - nam.query_end, ref.size());

    GPUAlignmentInfo info;
    int result_ref_start;
    bool gapped = true;
    if (projected_ref_end - projected_ref_start == query.size() && consistent_nam) {
        my_string ref_segm_ham = ref.substr(projected_ref_start, query.size());
        int hamming_dist = 0;
        for (size_t i = 0; i < query.size(); ++i) {
            if (query[i] != ref_segm_ham[i]) {
                ++hamming_dist;
            }
        }
        if (hamming_dist >= 0 && (((float) hamming_dist / query.size()) < 0.05)) {  //Hamming distance worked fine, no need to ksw align
            my_hamming_align(
                    query, ref_segm_ham, aligner_parameters.match, aligner_parameters.mismatch,
                    aligner_parameters.end_bonus, info
            );
            result_ref_start = projected_ref_start + info.ref_start;
            gapped = false;
        }
    }

    align_tmp_res.todo_nams.push_back(nam);
    align_tmp_res.is_extend_seed.push_back(true);
    if (gapped) {
        // not pass hamming, pending to do align on GPU, tag is false
        GPUAlignment alignment;
        align_tmp_res.done_align.push_back(false);
        align_tmp_res.align_res.push_back(alignment);
    } else {
        // pass hamming, store result, tag is true
        align_tmp_res.done_align.push_back(true);
        int softclipped = info.query_start + (query.size() - info.query_end);
        GPUAlignment alignment;
        //alignment.cigar.move_from(info.cigar);
        alignment.edit_distance = info.edit_distance;
        alignment.global_ed = info.edit_distance + softclipped;
        alignment.score = info.sw_score;
        alignment.ref_start = result_ref_start;
        alignment.length = info.ref_span();
        alignment.is_rc = nam.is_rc;
        alignment.is_unaligned = false;
        alignment.ref_id = nam.ref_id;
        alignment.gapped = gapped;
        align_tmp_res.align_res.push_back(alignment);
    }
    return gapped;
}

__device__ bool has_shared_substring(const my_string& read_seq, const my_string& ref_seq, int k) {
    int sub_size = 2 * k / 3;
    int step_size = k / 3;
    my_string submer;
    for (size_t i = 0; i + sub_size < read_seq.size(); i += step_size) {
        submer = read_seq.substr(i, sub_size);
        if (ref_seq.find(submer) != -1) {
            return true;
        }
    }
    return false;
}

__device__ bool rescue_mate_part(
        GPUAlignTmpRes& align_tmp_res,
        const AlignmentParameters& aligner_parameters,
        const Nam& nam,
        const GPUReferences& references,
        const GPURead& read,
        float mu,
        float sigma,
        int k
) {
    GPUAlignment alignment;
    int a, b;
    my_string r_tmp;
    auto read_len = read.size();

    if (nam.is_rc) {
        r_tmp = my_string(read.seq, read_len);
        a = nam.ref_start - nam.query_start - (mu + 5 * sigma);
        b = nam.ref_start - nam.query_start + read_len / 2;  // at most half read overlap
    } else {
        r_tmp = my_string(read.rc, read_len);                                             // mate is rc since fr orientation
        a = nam.ref_end + (read_len - nam.query_end) - read_len / 2;  // at most half read overlap
        b = nam.ref_end + (read_len - nam.query_end) + (mu + 5 * sigma);
    }

    auto ref_len = static_cast<int>(references.lengths[nam.ref_id]);
    auto ref_start = my_max(0, my_min(a, ref_len));
    auto ref_end = my_min(ref_len, my_max(0, b));

    align_tmp_res.todo_nams.push_back(nam);
    align_tmp_res.is_extend_seed.push_back(false);
    if (ref_end < ref_start + k) {
//        alignment.cigar = Cigar();
        alignment.edit_distance = read_len;
        alignment.score = 0;
        alignment.ref_start = 0;
        alignment.is_rc = nam.is_rc;
        alignment.ref_id = nam.ref_id;
        alignment.is_unaligned = true;
        align_tmp_res.done_align.push_back(true);
        align_tmp_res.align_res.push_back(alignment);
        return true;
    }
    my_string ref_segm = references.sequences[nam.ref_id].substr(ref_start, ref_end - ref_start);

    if (!has_shared_substring(r_tmp, ref_segm, k)) {
//        alignment.cigar = Cigar();
        alignment.edit_distance = read_len;
        alignment.score = 0;
        alignment.ref_start = 0;
        alignment.is_rc = nam.is_rc;
        alignment.ref_id = nam.ref_id;
        alignment.is_unaligned = true;
        align_tmp_res.done_align.push_back(true);
        align_tmp_res.align_res.push_back(alignment);
        return true;
    }

    align_tmp_res.done_align.push_back(false);
    align_tmp_res.align_res.push_back(alignment);
    return false;
}


__device__ bool reverse_nam_if_needed(Nam& nam, const GPURead& read, const GPUReferences& references, int k) {
    auto read_len = read.size();
    my_string ref_start_kmer = references.sequences[nam.ref_id].substr(nam.ref_start, k);
    my_string ref_end_kmer = references.sequences[nam.ref_id].substr(nam.ref_end - k, k);


    my_string seq, seq_rc;
    if (nam.is_rc) {
        seq = my_string(read.rc, read_len);
        seq_rc = my_string(read.seq, read_len);
    } else {
        seq = my_string(read.seq, read_len);
        seq_rc = my_string(read.rc, read_len);
    }
    my_string read_start_kmer = seq.substr(nam.query_start, k);
    my_string read_end_kmer = seq.substr(nam.query_end - k, k);
    if (ref_start_kmer == read_start_kmer && ref_end_kmer == read_end_kmer) {
        return true;
    }

    // False forward or false reverse (possible due to symmetrical hash values)
    //    we need two extra checks for this - hopefully this will remove all the false hits we see (true hash collisions should be very few)
    int q_start_tmp = read_len - nam.query_end;
    int q_end_tmp = read_len - nam.query_start;
    // false reverse hit, change coordinates in nam to forward
    read_start_kmer = seq_rc.substr(q_start_tmp, k);
    read_end_kmer = seq_rc.substr(q_end_tmp - k, k);
    if (ref_start_kmer == read_start_kmer && ref_end_kmer == read_end_kmer) {
        nam.is_rc = !nam.is_rc;
        nam.query_start = q_start_tmp;
        nam.query_end = q_end_tmp;
        return true;
    }
    return false;
}

__device__ void rescue_read_part(
        int flag,
        GPUAlignTmpRes& align_tmp_res,
        const GPURead& read2,  // read to be rescued
        const GPURead& read1,  // read that has NAMs
        const AlignmentParameters& aligner_parameters,
        const GPUReferences& references,
        my_vector<Nam>& nams1,
        int max_tries,
        float dropoff,
        int k,
        float mu,
        float sigma,
        size_t max_secondary,
        double secondary_dropoff,
        bool swap_r1r2
) {
    //align_tmp_res.type = flag;
    Nam n_max1 = nams1[0];
    int tries = 0;
    // this loop is safe, loop size is stable
    for (int i = 0; i < nams1.size(); i++) {
        Nam &nam = nams1[i];
        float score_dropoff1 = (float) nam.n_hits / n_max1.n_hits;
        // only consider top hits (as minimap2 does) and break if below dropoff cutoff.
        if (tries >= max_tries || score_dropoff1 < dropoff) {
            break;
        }

        const bool consistent_nam = reverse_nam_if_needed(nam, read1, references, k);
        // reserve extend and store info
        if(flag == 1) align_tmp_res.is_read1.push_back(true);
        else align_tmp_res.is_read1.push_back(false);
        bool gapped = extend_seed_part(align_tmp_res, aligner_parameters, nam, references, read1, consistent_nam);

        // Force SW alignment to rescue mate
        if(flag == 1) align_tmp_res.is_read1.push_back(false);
        else align_tmp_res.is_read1.push_back(true);
        bool is_unaligned = rescue_mate_part(align_tmp_res, aligner_parameters, nam, references, read2, mu, sigma, k);
        tries++;
    }
}

__device__ bool is_proper_nam_pair(const Nam nam1, const Nam nam2, float mu, float sigma) {
    if (nam1.ref_id != nam2.ref_id || nam1.is_rc == nam2.is_rc) {
        return false;
    }
    int a = my_max(0, nam1.ref_start - nam1.query_start);
    int b = my_max(0, nam2.ref_start - nam2.query_start);

    // r1 ---> <---- r2
    bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);
    if(r1_r2) return 1;

    // r2 ---> <---- r1
    bool r2_r1 = nam1.is_rc && (b <= a) && (a - b < mu + 10 * sigma);
    if(r2_r1) return 1;
    return 0;

//    return r1_r2 || r2_r1;
}

__device__ float top_dropoff(my_vector<Nam>& nams) {
    auto& n_max = nams[0];
    if (n_max.n_hits <= 2) {
        return 1.0;
    }
    if (nams.size() > 1) {
        return (float) nams[1].n_hits / n_max.n_hits;
    }
    return 0.0;
}

__device__ uint8_t get_mapq(const my_vector<Nam>& nams, const Nam& n_max) {
    if (nams.size() <= 1) {
        return 60;
    }
    const float s1 = n_max.score;
    const float s2 = nams[1].score;
    // from minimap2: MAPQ = 40(1−s2/s1) ·min{1,|M|/10} · log s1
    const float min_matches = my_min(n_max.n_hits / 10.0, 1.0);
    const int uncapped_mapq = 40 * (1 - s2 / s1) * min_matches * log(s1);
    return my_min(uncapped_mapq, 60);
}

__device__ bool is_proper_pair(const GPUAlignment& alignment1, const GPUAlignment& alignment2, float mu, float sigma) {
    const int dist = alignment2.ref_start - alignment1.ref_start;
    const bool same_reference = alignment1.ref_id == alignment2.ref_id;
    const bool both_aligned = same_reference && !alignment1.is_unaligned && !alignment2.is_unaligned;
    const bool r1_r2 = !alignment1.is_rc && alignment2.is_rc && dist >= 0; // r1 ---> <---- r2
    const bool r2_r1 = !alignment2.is_rc && alignment1.is_rc && dist <= 0; // r2 ---> <---- r1
    const bool rel_orientation_good = r1_r2 || r2_r1;
    const bool insert_good = std::abs(dist) <= mu + 6 * sigma;

    return both_aligned && insert_good && rel_orientation_good;
}

struct GPUInsertSizeDistribution {
    float sample_size = 1;
    float mu = 300;
    float sigma = 100;
    float V = 10000;
    float SSE = 10000;

    // Add a new observation
    __device__ void update(int dist) {
        if (dist >= 2000) {
            return;
        }
        const float e = dist - mu;
        mu += e / sample_size;  // (1.0/(sample_size +1.0)) * (sample_size*mu + d);
        SSE += e * (dist - mu);
        if (sample_size > 1) {
            //d < 1000 ? ((sample_size +1.0)/sample_size) * ( (V*sample_size/(sample_size +1)) + ((mu-d)*(mu-d))/sample_size ) : V;
            V = SSE / (sample_size - 1.0);
        } else {
            V = SSE;
        }
        sigma = sqrtf(V);
        sample_size = sample_size + 1.0;
        if (mu < 0) {
            printf("mu negative, mu: %f sigma: %f SSE: %f sample size: %f\n", mu, sigma, SSE, sample_size);
            assert(false);
        }
        if (SSE < 0) {
            printf("SSE negative, mu: %f sigma: %f SSE: %f sample size: %f\n", mu, sigma, SSE, sample_size);
            assert(false);
        }
    }
};


struct NamPair {
    int score;
    Nam nam1;
    Nam nam2;
};

__device__ my_vector<NamPair> get_best_scoring_nam_pairs(
        const my_vector<Nam>& nams1,
        const my_vector<Nam>& nams2,
        float mu,
        float sigma
) {
    my_vector<NamPair> joint_nam_scores;
    if (nams1.empty() && nams2.empty()) {
        return joint_nam_scores;
    }

    my_vector<bool> added_n1(nams1.size());
    my_vector<bool> added_n2(nams2.size());
    for(int i = 0; i < nams1.size(); i++) added_n1[i] = false;
    for(int i = 0; i < nams2.size(); i++) added_n2[i] = false;

    int best_joint_hits = 0;

    my_vector<Nam> nams2_sorted[2];
    for(int i = 0; i < nams2.size(); i++) {
        Nam nam2 = nams2[i];
        nams2_sorted[nam2.is_rc].push_back(nam2);
    }

    for(int id = 0; id < 2; id++) {
        quick_sort_iterative(&(nams2_sorted[id][0]), 0, nams2_sorted[id].size() - 1, [](const Nam &n1, const Nam &n2) {
            int val1 = my_max(0, n1.ref_start - n1.query_start);
            int val2 = my_max(0, n2.ref_start - n2.query_start);
//            if (val1 == val2) {
//                return n1.nam_id < n2.nam_id;
//            } else {
                return val1 < val2;
//            }
        });

//        Nam* data = &(nams2_sorted[id][0]);
//        int size = nams2_sorted[id].size();
//        for (int i = 0; i < size - 1; ++i) {
//            for (int j = 0; j < size - i - 1; ++j) {
//                Nam n1 = data[j];
//                Nam n2 = data[j + 1];
//                int val1 = my_max(0, n1.ref_start - n1.query_start);
//                int val2 = my_max(0, n2.ref_start - n2.query_start);
//                if (val1 > val2) {
//                    Nam temp = data[j];
//                    data[j] = data[j + 1];
//                    data[j + 1] = temp;
//                }
//            }
//        }

        // check if sort res is ok
        for (int i = 0; i < nams2_sorted[id].size(); i++) {
            Nam nam2 = nams2_sorted[id][i];
            int val1 = my_max(0, nam2.ref_start - nam2.query_start);
            if (i > 0) {
                Nam nam2_prev = nams2_sorted[id][i - 1];
                int val2 = my_max(0, nam2_prev.ref_start - nam2_prev.query_start);
                if (val1 < val2) {
                    printf("sort error: %d %d %d %d\n", i, val1, val2, id);
                    assert(false);
                }
            }
        }

    }

    for(int i = 0; i < nams1.size(); i++) {
        Nam nam1 = nams1[i];
        int nam1_val = my_max(0, nam1.ref_start - nam1.query_start);
        if(nam1.is_rc == 1) {
            float L_val = nam1_val - (mu + 10 * sigma);
            float R_val = nam1_val;
            int ll = 0, rr = nams2_sorted[0].size() - 1, ans_pos = nams2_sorted[0].size();
            while(ll <= rr) {
                int mid  = (ll + rr) / 2;
                int now_val = my_max(0, nams2_sorted[0][mid].ref_start - nams2_sorted[0][mid].query_start);
                if(now_val > L_val) {
                    rr = mid - 1;
                    ans_pos = mid;
                } else {
                    ll = mid + 1;
                }
            }

            for (int id = ans_pos; id < nams2_sorted[0].size(); id++) {
                Nam nam2 = nams2_sorted[0][id];
                int joint_hits = nam1.n_hits + nam2.n_hits;
                if(nam1.ref_id != nam2.ref_id) continue;
                int a = my_max(0, nam1.ref_start - nam1.query_start);
                int b = my_max(0, nam2.ref_start - nam2.query_start);
                if(b > R_val - 1e-6) break;

                // nam1 is rec, nam2 is fwd

                //// r1 ---> <---- r2
                //bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);

                // r2 ---> <---- r1
                bool r2_r1 = (a - b >= 0) && (a - b < mu + 10 * sigma);

                if (r2_r1) {
                    joint_nam_scores.push_back(NamPair{joint_hits, nam1, nam2});
                    added_n1[nam1.nam_id] = 1;
                    added_n2[nam2.nam_id] = 1;
                    //best_joint_hits = my_max(joint_hits, best_joint_hits);
                }
            }
        } else{
            float L_val = nam1_val;
            float R_val = nam1_val + mu + 10 * sigma;
            int ll = 0, rr = nams2_sorted[1].size() - 1, ans_pos = nams2_sorted[1].size();
            while(ll <= rr) {
                int mid  = (ll + rr) / 2;
                int now_val = my_max(0, nams2_sorted[1][mid].ref_start - nams2_sorted[1][mid].query_start);
                if(now_val >= L_val) {
                    rr = mid - 1;
                    ans_pos = mid;
                } else {
                    ll = mid + 1;
                }
            }

            for (int id = ans_pos; id < nams2_sorted[1].size(); id++) {
                Nam nam2 = nams2_sorted[1][id];
                int joint_hits = nam1.n_hits + nam2.n_hits;
                if(nam1.ref_id != nam2.ref_id) continue;

                int a = my_max(0, nam1.ref_start - nam1.query_start);
                int b = my_max(0, nam2.ref_start - nam2.query_start);
                if(b >= R_val - 1e-6) break;

                // nam1 is fwd, nam2 is rec

                // r1 ---> <---- r2
                bool r1_r2 = (b - a >= 0) && (b - a < mu + 10 * sigma);

                if (r1_r2) {
                    joint_nam_scores.push_back(NamPair{joint_hits, nam1, nam2});
                    added_n1[nam1.nam_id] = 1;
                    added_n2[nam2.nam_id] = 1;
                    //best_joint_hits = my_max(joint_hits, best_joint_hits);
                }
            }
        }

    }

//    printf("joint_nam_scores : %d\n", joint_nam_scores.size());



    // Find high-scoring R1 NAMs that are not part of a proper pair
    Nam dummy_nam;
    dummy_nam.ref_start = -1;
    if (!nams1.empty()) {
        int best_joint_hits1 = best_joint_hits > 0 ? best_joint_hits : nams1[0].n_hits;
        for(int i = 0; i < nams1.size(); i++) {
            Nam nam1 = nams1[i];
            if (nam1.n_hits < best_joint_hits1 / 2) {
                break;
            }
            if (added_n1[nam1.nam_id]) {
                continue;
            }
            joint_nam_scores.push_back(NamPair{nam1.n_hits, nam1, dummy_nam});
        }
    }


    // Find high-scoring R2 NAMs that are not part of a proper pair
    if (!nams2.empty()) {
        int best_joint_hits2 = best_joint_hits > 0 ? best_joint_hits : nams2[0].n_hits;
        for(int i = 0; i < nams2.size(); i++) {
            Nam nam2 = nams2[i];
            if (nam2.n_hits < best_joint_hits2 / 2) {
                break;
            }
            if (added_n2[nam2.nam_id]) {
                continue;
            }
            joint_nam_scores.push_back(NamPair{nam2.n_hits, dummy_nam, nam2});
        }
    }



    quick_sort_iterative(&(joint_nam_scores[0]), 0, joint_nam_scores.size() - 1, [](const NamPair &n1, const NamPair &n2) {
//        if (n1.score == n2.score) {
//            if (n1.nam1.nam_id == n2.nam1.nam_id) {
//                return n1.nam2.nam_id < n2.nam2.nam_id;
//            } else {
//                return n1.nam1.nam_id < n2.nam1.nam_id;
//            }
//        } else {
            return n1.score > n2.score;
//        }
    });

//    NamPair* data = &(joint_nam_scores[0]);
//    int size = joint_nam_scores.size();
//    for (int i = 0; i < size - 1; ++i) {
//        for (int j = 0; j < size - i - 1; ++j) {
//            NamPair n1 = data[j];
//            NamPair n2 = data[j + 1];
//            if (n1.score < n2.score) {
//                NamPair temp = data[j];
//                data[j] = data[j + 1];
//                data[j + 1] = temp;
//            }
//        }
//    }
    // check if sort res is ok
    for (int i = 0; i < joint_nam_scores.size(); i++) {
        NamPair nam2 = joint_nam_scores[i];
        if (i > 0) {
            NamPair nam2_prev = joint_nam_scores[i - 1];
            if (nam2.score > nam2_prev.score) {
                printf("sort error: %d %d %d\n", i, nam2.score, nam2_prev.score);
                assert(false);
            }
        }
    }

    return joint_nam_scores;
}

__device__ static unsigned char revcomp_table[256] = {
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'T', 'N', 'G',  'N', 'N', 'N', 'C',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'A', 'A', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'T', 'N', 'G',  'N', 'N', 'N', 'C',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'A', 'A', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N'
};

struct GPUScoredAlignmentPair {
    double score;
    GPUAlignment alignment1;
    GPUAlignment alignment2;
};



__device__ void align_PE_part(
        GPUAlignTmpRes& align_tmp_res,
        const AlignmentParameters& aligner_parameters,
        my_vector<Nam>& nams1,
        my_vector<Nam>& nams2,
        char* seq1, int seq_len1,
        char* seq2, int seq_len2,
        int k,
        const GPUReferences& references,
        float dropoff,
        GPUInsertSizeDistribution& isize_est,
        unsigned max_tries,
        size_t max_secondary
) {
    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    char* rc1 = (char*)my_malloc(seq_len1 + 1);
    char* rc2 = (char*)my_malloc(seq_len2 + 1);
    for(int i = 0; i < seq_len1; i++) {
        rc1[i] = revcomp_table[static_cast<int>(seq1[seq_len1 - i - 1])];
    }
    for(int i = 0; i < seq_len2; i++) {
        rc2[i] = revcomp_table[static_cast<int>(seq2[seq_len2 - i - 1])];
    }
    GPURead read1{seq1, rc1, seq_len1};
    GPURead read2{seq2, rc2, seq_len2};
    double secondary_dropoff = 2 * aligner_parameters.mismatch + aligner_parameters.gap_open;


    if (nams1.empty() && nams2.empty()) {
        // None of the reads have any NAMs
        align_tmp_res.type = 0;
        my_free(rc1);
        my_free(rc2);
        return;
    } 

    if (!nams1.empty() && nams2.empty()) {
        // Only read 1 has NAMS: attempt to rescue read 2
        align_tmp_res.type = 1;
        rescue_read_part(
                1, align_tmp_res, read2, read1, aligner_parameters, references, nams1, max_tries, dropoff, k, mu,
                sigma, max_secondary, secondary_dropoff, false
        );
        my_free(rc1);
        my_free(rc2);
        return;
    }

    if (nams1.empty() && !nams2.empty()) {
        // Only read 2 has NAMS: attempt to rescue read 1
        align_tmp_res.type = 2;
        rescue_read_part(
                2, align_tmp_res, read1, read2, aligner_parameters, references, nams2, max_tries, dropoff, k, mu,
                sigma, max_secondary, secondary_dropoff, true
        );
        my_free(rc1);
        my_free(rc2);
        return;
    }

    //    printf("type3\n");

    // If we get here, both reads have NAMs
    assert(!nams1.empty() && !nams2.empty());

    // Deal with the typical case that both reads map uniquely and form a proper pair
    if (top_dropoff(nams1) < dropoff && top_dropoff(nams2) < dropoff &&
        is_proper_nam_pair(nams1[0], nams2[0], mu, sigma)) {
        align_tmp_res.type = 3;
        Nam n_max1 = nams1[0];
        Nam n_max2 = nams2[0];

        bool consistent_nam1 = reverse_nam_if_needed(n_max1, read1, references, k);
        bool consistent_nam2 = reverse_nam_if_needed(n_max2, read2, references, k);

        align_tmp_res.is_read1.push_back(true);
        bool gapped1 = extend_seed_part(align_tmp_res, aligner_parameters, n_max1, references, read1, consistent_nam1);


        align_tmp_res.is_read1.push_back(false);
        bool gapped2 = extend_seed_part(align_tmp_res, aligner_parameters, n_max2, references, read2, consistent_nam2);

        int mapq1 = get_mapq(nams1, n_max1);
        int mapq2 = get_mapq(nams2, n_max2);
        align_tmp_res.mapq1 = mapq1;
        align_tmp_res.mapq2 = mapq2;


        if(!gapped1 && !gapped2) {
            int res_size = align_tmp_res.align_res.size();
            if(res_size < 2) {
                printf("align_tmp_res.align_res.size error %d\n", res_size);
                assert(false);
            }
            GPUAlignment alignment1 = align_tmp_res.align_res[res_size - 2];
            GPUAlignment alignment2 = align_tmp_res.align_res[res_size - 1];
            if(alignment1.gapped || alignment2.gapped) {
                printf("alignment gapped error\n");
                assert(false);
            }
            bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
            if ((isize_est.sample_size < 400) && (alignment1.edit_distance + alignment2.edit_distance < 3) && is_proper) {
                isize_est.update(my_abs(alignment1.ref_start - alignment2.ref_start));
            }
        }
        my_free(rc1);
        my_free(rc2);
        return;
    }

//    printf("type4\n");

    my_vector<NamPair> joint_nam_scores = get_best_scoring_nam_pairs(nams1, nams2, mu, sigma);
    align_tmp_res.type = 4 + joint_nam_scores.size();
    my_free(rc1);
    my_free(rc2);
    return;

    //if (threadIdx.x == 0 && blockIdx.x % 1000 == 0) printf("joint_nam_scores size %d\n", joint_nam_scores.size());
    my_vector<bool> is_aligned1(nams1.size());
    my_vector<bool> is_aligned2(nams2.size());
    for (int i = 0; i < nams1.size(); i++) is_aligned1[i] = 0;
    for (int i = 0; i < nams2.size(); i++) is_aligned2[i] = 0;

    {
        Nam n1_max = nams1[0];
        bool consistent_nam1 = reverse_nam_if_needed(n1_max, read1, references, k);
        align_tmp_res.is_read1.push_back(true);
        bool gapped1 = extend_seed_part(align_tmp_res, aligner_parameters, n1_max, references, read1, consistent_nam1);
        is_aligned1[n1_max.nam_id] = 1;

        Nam n2_max = nams2[0];
        bool consistent_nam2 = reverse_nam_if_needed(n2_max, read2, references, k);
        align_tmp_res.is_read1.push_back(false);
        bool gapped2 = extend_seed_part(align_tmp_res, aligner_parameters, n2_max, references, read2, consistent_nam2);
        is_aligned2[n2_max.nam_id] = 1;
    }

    // Turn pairs of high-scoring NAMs into pairs of alignments
    my_vector<GPUScoredAlignmentPair> high_scores;
    double max_score = joint_nam_scores[0].score;
    align_tmp_res.type4_loop_size = 0;
    for(int i = 0; i < joint_nam_scores.size(); i++) {
        double score_ = joint_nam_scores[i].score;
        Nam n1 = joint_nam_scores[i].nam1;
        Nam n2 = joint_nam_scores[i].nam2;
        float score_dropoff = (float) score_ / max_score;
        if (high_scores.size() >= max_tries || score_dropoff < dropoff) {
            break;
        }

        align_tmp_res.type4_nams.push_back(n1);
        align_tmp_res.type4_nams.push_back(n2);
        align_tmp_res.type4_loop_size++;

        if (n1.ref_start >= 0) {
            if (is_aligned1[n1.nam_id] == 1) {

            } else {
                bool consistent_nam = reverse_nam_if_needed(n1, read1, references, k);
                align_tmp_res.is_read1.push_back(true);
                bool gapped = extend_seed_part(align_tmp_res, aligner_parameters, n1, references, read1, consistent_nam);
                is_aligned1[n1.nam_id] = 1;
            }
        } else {
            reverse_nam_if_needed(n2, read2, references, k);
            align_tmp_res.is_read1.push_back(true);
            bool is_unaligned = rescue_mate_part(align_tmp_res, aligner_parameters, n2, references, read1, mu, sigma, k);

        }

        // ref_start == -1 is a marker for a dummy NAM
        if (n2.ref_start >= 0) {
            if (is_aligned2[n2.nam_id] == 1) {

            } else {
                bool consistent_nam = reverse_nam_if_needed(n2, read2, references, k);
                align_tmp_res.is_read1.push_back(false);
                bool gapped = extend_seed_part(align_tmp_res, aligner_parameters, n2, references, read2, consistent_nam);
                is_aligned2[n2.nam_id] = 1;
            }
        } else {
            reverse_nam_if_needed(n1, read1, references, k);
            align_tmp_res.is_read1.push_back(false);
            bool is_unaligned = rescue_mate_part(align_tmp_res, aligner_parameters, n1, references, read2, mu, sigma, k);
        }
        GPUScoredAlignmentPair aln_pair;
        high_scores.push_back(aln_pair);
    }

    my_free(rc1);
    my_free(rc2);
    return;
}

#define BLOCK_SIZE 32


__device__ void sort_hits_single(
        my_vector<my_pair<int, Hit>>& hits_per_ref
) {
    //bubble_sort(&(hits_per_ref[0]), hits_per_ref.size());
    quick_sort(&(hits_per_ref[0]), hits_per_ref.size());
}

struct ref_ids_edge {
    int pre;
    int ref_id;
};

#define key_mod_val 29

__device__ int find_ref_ids(int ref_id, int* head, ref_ids_edge* edges) {
    int key = ref_id % key_mod_val;
    for (int i = head[key]; i != -1; i = edges[i].pre) {
        if (edges[i].ref_id == ref_id) return i;
    }
    return -1;
}

__device__ void sort_hits_by_refid_get_ref_size(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        my_vector<my_pair<int, int>>& ref_size
) {
    my_vector<my_pair<int, my_vector<Hit>*>> all_hits;
    int *head = (int*)my_malloc(key_mod_val * sizeof(int));
    my_vector<ref_ids_edge> edges;
    for(int i = 0; i < key_mod_val; i++) head[i] = -1;
    int ref_ids_num = 0;
    for(int i = 0; i < hits_per_ref.size(); i++) {
        int ref_id = hits_per_ref[i].first;
        int find_ref_id_rank = find_ref_ids(ref_id, head, edges.data);
        if (find_ref_id_rank == -1) {
            find_ref_id_rank = ref_ids_num;
            int key = ref_id % key_mod_val;
            edges.push_back({head[key], ref_id});
            head[key] = ref_ids_num++;
            my_vector<Hit>* hits = (my_vector<Hit>*)my_malloc(sizeof(my_vector<Hit>));
            hits->init();
            all_hits.push_back({ref_id, hits});
        }
        all_hits[find_ref_id_rank].second->push_back(hits_per_ref[i].second);
    }
    hits_per_ref.clear();
    my_vector<my_pair<int, int>>* ref_sizes_ptr = (my_vector<my_pair<int, int>>*)my_malloc(sizeof(my_vector<my_pair<int, int>>));
    ref_sizes_ptr->init();
    for(int i = 0; i < all_hits.size(); i++) {
        ref_sizes_ptr->push_back({hits_per_ref.size(), all_hits[i].second->size()});
        for(int j = 0; j < all_hits[i].second->size(); j++) {
            hits_per_ref.push_back({all_hits[i].first, (*all_hits[i].second)[j]});
        }
        all_hits[i].second->release();
        my_free(all_hits[i].second);
    }
    ref_size = *ref_sizes_ptr;
    my_free(ref_sizes_ptr);
    my_free(head);
}

__device__ void sort_hits_by_refid(
        my_vector<my_pair<int, Hit>>& hits_per_ref
) {
    my_vector<my_pair<int, my_vector<Hit>*>> all_hits(32);
    int *head = (int*)my_malloc(key_mod_val * sizeof(int));
    my_vector<ref_ids_edge> edges(32);
    for(int i = 0; i < key_mod_val; i++) head[i] = -1;
    int ref_ids_num = 0;
    for(int i = 0; i < hits_per_ref.size(); i++) {
        int ref_id = hits_per_ref[i].first;
        int find_ref_id_rank = find_ref_ids(ref_id, head, edges.data);
        if (find_ref_id_rank == -1) {
            find_ref_id_rank = ref_ids_num;
            int key = ref_id % key_mod_val;
            edges.push_back({head[key], ref_id});
            head[key] = ref_ids_num++;
            my_vector<Hit>* hits = (my_vector<Hit>*)my_malloc(sizeof(my_vector<Hit>));
            hits->init(32);
            all_hits.push_back({ref_id, hits});
        }
        all_hits[find_ref_id_rank].second->push_back(hits_per_ref[i].second);
    }
    hits_per_ref.clear();
    for(int i = 0; i < all_hits.size(); i++) {
        for(int j = 0; j < all_hits[i].second->size(); j++) {
            hits_per_ref.push_back({all_hits[i].first, (*all_hits[i].second)[j]});
        }
        all_hits[i].second->release();
        my_free(all_hits[i].second);
    }
    my_free(head);
}

__device__ void sort_hits_parallel(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int k,
        bool is_revcomp,
        int tid
) {
    if(hits_per_ref.size() == 0) return;
    int num_hits = hits_per_ref.size();

    const int items_per_thread = 160;
    int real_num_hits = items_per_thread * BLOCK_SIZE;
    if(real_num_hits < num_hits) {
        printf("real_num_hits %d num_hits %d\n", real_num_hits, num_hits);
    }
    assert(real_num_hits >= num_hits);

    typedef cub::BlockRadixSort<unsigned long long, BLOCK_SIZE, items_per_thread, int> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    unsigned long long thread_keys[items_per_thread];
    int thread_indices[items_per_thread];

    __shared__ int* old_ref_end;
    __shared__ int* old_query_end;
    if(tid == 0) {
        old_ref_end = (int*)my_malloc(real_num_hits * sizeof(int));
        old_query_end = (int*)my_malloc(real_num_hits * sizeof(int));
    }
    __syncthreads();

    for (int i = 0; i < items_per_thread; ++i) {
        int idx = tid * items_per_thread + i;
        if (idx < num_hits) {
            thread_keys[i] = (static_cast<unsigned long long>(hits_per_ref[idx].first) << 48) |
                             (static_cast<unsigned long long>(hits_per_ref[idx].second.query_start & 0xFFFF) << 32) |
                             (static_cast<unsigned long long>(hits_per_ref[idx].second.ref_start) & 0xFFFFFFFF);
            thread_indices[i] = idx;
            old_ref_end[idx] = hits_per_ref[idx].second.ref_end;
            old_query_end[idx] = hits_per_ref[idx].second.query_end;
        } else {
            thread_keys[i] = ULLONG_MAX;
            thread_indices[i] = -1;
            old_ref_end[idx] = 0;
            old_query_end[idx] = 0;
        }
    }
    __syncthreads();

    BlockRadixSort(temp_storage).Sort(thread_keys, thread_indices);
    __syncthreads();

    for (int i = 0; i < items_per_thread; ++i) {
        int idx = tid * items_per_thread + i;
        if (idx < num_hits) {
            hits_per_ref[idx].first = thread_keys[i] >> 48;
            hits_per_ref[idx].second.query_start = (thread_keys[i] >> 32) & 0xFFFF;
            hits_per_ref[idx].second.ref_start = thread_keys[i] & 0xFFFFFFFF;
            hits_per_ref[idx].second.ref_end = old_ref_end[thread_indices[i]];
            hits_per_ref[idx].second.query_end = old_query_end[thread_indices[i]];
        }
    }
    __syncthreads();
    if(tid == 0) {
        my_free(old_ref_end);
        my_free(old_query_end);
    }


}

__device__ size_t my_lower_bound(my_pair<int, Hit>* hits, size_t i_start, size_t i_end, int target) {
    size_t left = i_start, right = i_end;
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (hits[mid].second.ref_start < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__device__ void salign_merge_hits(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int k,
        bool is_revcomp,
        my_vector<Nam>& nams
) {
    if(hits_per_ref.size() == 0) return;
    int ref_num = 0;
    my_vector<int> each_ref_size;
    int pre_ref_id = hits_per_ref[0].first;
    int now_ref_num = 1;
    for(int i = 1; i < hits_per_ref.size(); i++) {
        int ref_id = hits_per_ref[i].first;
        Hit hit = hits_per_ref[i].second;
        if(ref_id != pre_ref_id) {
            ref_num++;
            pre_ref_id = ref_id;
            each_ref_size.push_back(now_ref_num);
            now_ref_num = 1;
        } else {
            now_ref_num++;
        }
    }
    ref_num++;
    each_ref_size.push_back(now_ref_num);
    //int mx_hits_per_ref = 0;
    //for (int i = 0; i < each_ref_size.size(); i++) {
    //    mx_hits_per_ref = my_max(mx_hits_per_ref, each_ref_size[i]);
    //}

    my_vector<Nam> open_nams;
    //(mx_hits_per_ref);

    int now_vec_pos = 0;
    for (int rid = 0; rid < ref_num; rid++) {
        if(rid != 0) now_vec_pos += each_ref_size[rid - 1];
        int ref_id = hits_per_ref[now_vec_pos].first;
        open_nams.clear();
        unsigned int prev_q_start = 0;
        size_t hits_size = each_ref_size[rid];
        my_pair<int, Hit>* hits = &(hits_per_ref[now_vec_pos]);
        for (size_t i = 0; i < hits_size; ) {
            size_t i_start = i;
            size_t i_end = i + 1;
            size_t i_size;
            while(i_end < hits_size && hits[i_end].second.query_start == hits[i].second.query_start) i_end++;
            i = i_end;
            i_size = i_end - i_start;
            //for(int j = 0; j < i_size - 1; j++) {
            //    assert(hits[i_start + j].second.ref_start <= hits[i_start + j + 1].second.ref_start);
            //}
            //quick_sort(&(hits[i_start]), i_size);
            my_vector<bool> is_added(i_size);
            for(size_t j = 0; j < i_size; j++) is_added.push_back(false);
            int query_start = hits[i_start].second.query_start;
            int cnt_done = 0;
            for (int k = 0; k < open_nams.size(); k++) {
                Nam& o = open_nams[k];
                if ( query_start > o.query_end ) continue;
                size_t lower = my_lower_bound(hits, i_start, i_end, o.ref_prev_hit_startpos + 1);
                size_t upper = my_lower_bound(hits, i_start, i_end, o.ref_end + 1);
                for (size_t j = lower; j < upper; j++) {
                    if(is_added[j - i_start]) continue;
                    Hit& h = hits[j].second;
                    {
                        if (o.ref_prev_hit_startpos < h.ref_start && h.ref_start <= o.ref_end) {
                            if ((h.query_end > o.query_end) && (h.ref_end > o.ref_end)) {
                                o.query_end = h.query_end;
                                o.ref_end = h.ref_end;
                                //                        o.previous_query_start = h.query_s;
                                //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                                o.query_prev_hit_startpos = h.query_start;
                                o.ref_prev_hit_startpos = h.ref_start;
                                o.n_hits++;
                                //                        o.score += (float)1/ (float)h.count;
                                is_added[j - i_start] = true;
                                cnt_done++;
                                break;
                            } else if ((h.query_end <= o.query_end) && (h.ref_end <= o.ref_end)) {
                                //                        o.previous_query_start = h.query_s;
                                //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                                o.query_prev_hit_startpos = h.query_start;
                                o.ref_prev_hit_startpos = h.ref_start;
                                o.n_hits++;
                                //                        o.score += (float)1/ (float)h.count;
                                is_added[j - i_start] = true;
                                cnt_done++;
                                break;
                            }
                        }
                    }
                }
                if(cnt_done == i_size) break;
            }

            // Add the hit to open matches
            for(size_t j = 0; j < i_size; j++) {
                if (!is_added[j]){
                    Nam n;
                    n.query_start = hits[i_start + j].second.query_start;
                    n.query_end = hits[i_start + j].second.query_end;
                    n.ref_start = hits[i_start + j].second.ref_start;
                    n.ref_end = hits[i_start + j].second.ref_end;
                    n.ref_id = ref_id;
                    //                n.previous_query_start = h.query_s;
                    //                n.previous_ref_start = h.ref_s;
                    n.query_prev_hit_startpos = hits[i_start + j].second.query_start;
                    n.ref_prev_hit_startpos = hits[i_start + j].second.ref_start;
                    n.n_hits = 1;
                    n.is_rc = is_revcomp;
                    //                n.score += (float)1 / (float)h.count;
                    open_nams.push_back(n);
                }
            }

            // Only filter if we have advanced at least k nucleotides
            if (query_start > prev_q_start + k) {

                // Output all NAMs from open_matches to final_nams that the current hit have passed
                for (int k = 0; k < open_nams.size(); k++) {
                    Nam& n = open_nams[k];
                    if (n.query_end < query_start) {
                        int n_max_span = my_max(n.query_span(), n.ref_span());
                        int n_min_span = my_min(n.query_span(), n.ref_span());
                        float n_score;
                        n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
                        //                        n_score = n.n_hits * n.query_span();
                        n.score = n_score;
                        n.nam_id = nams.size();
                        nams.push_back(n);
                    }
                }

                // Remove all NAMs from open_matches that the current hit have passed
                auto c = query_start;
                int old_open_size = open_nams.size();
                open_nams.clear();
                for (int in = 0; in < old_open_size; ++in) {
                    if (!(open_nams[in].query_end < c)) {
                        open_nams.push_back(open_nams[in]);
                    }
                }
                prev_q_start = query_start;
            }
        }
        // Add all current open_matches to final NAMs
        for (int k = 0; k < open_nams.size(); k++) {
            Nam& n = open_nams[k];
            int n_max_span = my_max(n.query_span(), n.ref_span());
            int n_min_span = my_min(n.query_span(), n.ref_span());
            float n_score;
            n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
            //            n_score = n.n_hits * n.query_span();
            n.score = n_score;
            n.nam_id = nams.size();
            nams.push_back(n);
        }
    }
}

__device__ int lock = 0;

__device__ void acquire_lock() {
    while (atomicCAS(&lock, 0, 1) != 0) {
    }
}

__device__ void release_lock() {
    atomicExch(&lock, 0);
}

__device__ void merge_hits(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int k,
        bool is_revcomp,
        my_vector<Nam>& nams
) {
    if(hits_per_ref.size() == 0) return;
    int num_hits = hits_per_ref.size();

    int ref_num = 0;
    my_vector<int> each_ref_size;
    int pre_ref_id = hits_per_ref[0].first;
    int now_ref_num = 1;
    for(int i = 1; i < hits_per_ref.size(); i++) {
        int ref_id = hits_per_ref[i].first;
        Hit hit = hits_per_ref[i].second;
        if(ref_id != pre_ref_id) {
            ref_num++;
            pre_ref_id = ref_id;
            each_ref_size.push_back(now_ref_num);
            now_ref_num = 1;
        } else {
            now_ref_num++;
        }
    }
    ref_num++;
    each_ref_size.push_back(now_ref_num);

    my_vector<Nam> open_nams;

    int now_vec_pos = 0;
    for (int i = 0; i < ref_num; i++) {

        if(i != 0) now_vec_pos += each_ref_size[i - 1];
        int ref_id = hits_per_ref[now_vec_pos].first;
        open_nams.clear();
        unsigned int prev_q_start = 0;

        for (int j = 0; j < each_ref_size[i]; j++) {
            Hit& h = hits_per_ref[now_vec_pos + j].second;
            bool is_added = false;
            for (int k = 0; k < open_nams.size(); k++) {
                Nam& o = open_nams[k];

                // Extend NAM
                if ((o.query_prev_hit_startpos < h.query_start) && (h.query_start <= o.query_end ) && (o.ref_prev_hit_startpos < h.ref_start) && (h.ref_start <= o.ref_end) ){
                    if ( (h.query_end > o.query_end) && (h.ref_end > o.ref_end) ) {
                        o.query_end = h.query_end;
                        o.ref_end = h.ref_end;
                        //                        o.previous_query_start = h.query_s;
                        //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                        o.query_prev_hit_startpos = h.query_start;
                        o.ref_prev_hit_startpos = h.ref_start;
                        o.n_hits ++;
                        //                        o.score += (float)1/ (float)h.count;
                        is_added = true;
                        break;
                    }
                    else if ((h.query_end <= o.query_end) && (h.ref_end <= o.ref_end)) {
                        //                        o.previous_query_start = h.query_s;
                        //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                        o.query_prev_hit_startpos = h.query_start;
                        o.ref_prev_hit_startpos = h.ref_start;
                        o.n_hits ++;
                        //                        o.score += (float)1/ (float)h.count;
                        is_added = true;
                        break;
                    }
                }

            }

            // Add the hit to open matches
            if (!is_added){
                Nam n;
                n.query_start = h.query_start;
                n.query_end = h.query_end;
                n.ref_start = h.ref_start;
                n.ref_end = h.ref_end;
                n.ref_id = ref_id;
                //                n.previous_query_start = h.query_s;
                //                n.previous_ref_start = h.ref_s;
                n.query_prev_hit_startpos = h.query_start;
                n.ref_prev_hit_startpos = h.ref_start;
                n.n_hits = 1;
                n.is_rc = is_revcomp;
                //                n.score += (float)1 / (float)h.count;
                open_nams.push_back(n);
            }

            // Only filter if we have advanced at least k nucleotides
            if (h.query_start > prev_q_start + k) {
                // Output all NAMs from open_matches to final_nams that the current hit have passed
                for (int k = 0; k < open_nams.size(); k++) {
                    Nam& n = open_nams[k];
                    if (n.query_end < h.query_start) {
                        int n_max_span = my_max(n.query_span(), n.ref_span());
                        int n_min_span = my_min(n.query_span(), n.ref_span());
                        float n_score;
                        n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
                        //                        n_score = n.n_hits * n.query_span();
                        n.score = n_score;
                        n.nam_id = nams.size();
                        nams.push_back(n);
                    }
                }

                // Remove all NAMs from open_matches that the current hit have passed
                auto c = h.query_start;
                int old_open_size = open_nams.size();
                open_nams.clear();
                for (int in = 0; in < old_open_size; ++in) {
                    if (!(open_nams[in].query_end < c)) {
                        open_nams.push_back(open_nams[in]);
                    }
                }
                prev_q_start = h.query_start;
            }
        }

        // Add all current open_matches to final NAMs
        for (int k = 0; k < open_nams.size(); k++) {
            Nam& n = open_nams[k];
            int n_max_span = my_max(n.query_span(), n.ref_span());
            int n_min_span = my_min(n.query_span(), n.ref_span());
            float n_score;
            n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
            //            n_score = n.n_hits * n.query_span();
            n.score = n_score;
            n.nam_id = nams.size();
            nams.push_back(n);
        }
    }
}

__device__ void add_to_hits_per_ref(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int query_start,
        int query_end,
        size_t position,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        int k
) {
    int min_diff = 1 << 30;
    for (const auto hash = gpu_get_hash(d_randstrobes, d_randstrobes_size, position); gpu_get_hash(d_randstrobes, d_randstrobes_size, position) == hash; ++position) {
        int ref_start = d_randstrobes[position].position;
        int ref_end = ref_start + d_randstrobes[position].strobe2_offset() + k;
        int diff = std::abs((query_end - query_start) - (ref_end - ref_start));
        if (diff <= min_diff) {
            hits_per_ref.push_back({d_randstrobes[position].reference_index(), Hit{query_start, query_end, ref_start, ref_end}});
            min_diff = diff;
        }
    }
}


#define GPU_thread_task_size 1

__global__ void gpu_rescue_get_hits(
        int bits,
        unsigned int filter_cutoff,
        int rescue_cutoff,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        const my_bucket_index_t *d_randstrobe_start_indices,
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_hits_num,
        my_vector<QueryRandstrobe>* global_randstrobes,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        my_vector<my_pair<int, Hit>>* hits_per_ref0;
        my_vector<my_pair<int, Hit>>* hits_per_ref1;
        hits_per_ref0 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref1 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref0->init();
        hits_per_ref1->init();


        my_vector<RescueHit> hits_t0;
        my_vector<RescueHit> hits_t1;
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[id][i];
            //size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
            size_t position = q.hash;
            if (position != static_cast<size_t>(-1)) {
                if(position >= d_randstrobes_size) {
                    printf("position > d_randstrobes_size : %llu %llu\n", position, d_randstrobes_size);
                    assert(false);
                }
                unsigned int count = gpu_get_count(d_randstrobes, d_randstrobe_start_indices, position, bits);
                RescueHit rh{position, count, q.start, q.end};
                if(q.is_reverse) hits_t1.push_back(rh);
                else hits_t0.push_back(rh);
            }
        }
        global_randstrobes[id].release();
        quick_sort(&(hits_t0[0]), hits_t0.size());

        int cnt = 0;
        for (int i = 0; i < hits_t0.size(); i++) {
            RescueHit &rh = hits_t0[i];
            if ((rh.count > rescue_cutoff && cnt >= 5) || rh.count > rescue_threshold) {
                break;
            }
            add_to_hits_per_ref(*hits_per_ref0, rh.query_start, rh.query_end, rh.position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
            cnt++;
        }
        quick_sort(&(hits_t1[0]), hits_t1.size());
        cnt = 0;
        for (int i = 0; i < hits_t1.size(); i++) {
            RescueHit &rh = hits_t1[i];
            if ((rh.count > rescue_cutoff && cnt >= 5) || rh.count > rescue_threshold) {
                break;
            }
            add_to_hits_per_ref(*hits_per_ref1, rh.query_start, rh.query_end, rh.position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
            cnt++;
        }
        global_hits_num[id] = hits_per_ref0->size() + hits_per_ref1->size();
        hits_per_ref0s[id] = *hits_per_ref0;
        hits_per_ref1s[id] = *hits_per_ref1;
        my_free(hits_per_ref0);
        my_free(hits_per_ref1);
    }
}

__global__ void gpu_rescue_sort_hits(
        int num_tasks,
        IndexParameters *index_para,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int l_range = bid * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;

    for (int id = l_range; id < r_range; id++) {
        sort_hits_parallel(hits_per_ref0s[id], index_para->syncmer.k, 0, tid);
        sort_hits_parallel(hits_per_ref1s[id], index_para->syncmer.k, 1, tid);
    }
}

__global__ void gpu_rescue_merge_hits(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_nams_info,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        my_vector<Nam> nams(128);
        salign_merge_hits(hits_per_ref0s[id], index_para->syncmer.k, 0, nams);
        salign_merge_hits(hits_per_ref1s[id], index_para->syncmer.k, 1, nams);
        uint64_t local_nams_info = 0;
        for (int i = 0; i < nams.size(); i++) {
            local_nams_info += nams[i].ref_id + int(nams[i].score) + nams[i].query_start + nams[i].query_end;
        }
        global_nams_info[id] += local_nams_info;
        hits_per_ref0s[id].release();
        hits_per_ref1s[id].release();
    }
}


__global__ void gpu_get_randstrobes(
        int num_tasks,
        int *pre_sum,
        int *lens,
        char *all_seqs,
        int *pre_sum2,
        int *lens2,
        char *all_seqs2,
        IndexParameters *index_para,
        int *randstrobe_sizes,
        uint64_t *hashes,
        my_vector<QueryRandstrobe>* global_randstrobes
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    int read_num = num_tasks / 2;
    for (int id = l_range; id < r_range; id++) {
        int read_id = id % read_num;
        int rev = id / read_num;
        size_t len;
        char *seq;
        if (rev == 0) {
            len = lens[read_id];
            seq = all_seqs + pre_sum[read_id];
        } else {
            len = lens2[read_id];
            seq = all_seqs2 + pre_sum2[read_id];
        }

        my_vector<Syncmer> syncmers(len);

        const int k = index_para->syncmer.k;
        const int s = index_para->syncmer.s;
        const int t = index_para->syncmer.t_syncmer;

        const uint64_t kmask = (1ULL << 2 * k) - 1;
        const uint64_t smask = (1ULL << 2 * s) - 1;
        const uint64_t kshift = (k - 1) * 2;
        const uint64_t sshift = (s - 1) * 2;
        uint64_t gpu_qs[200];
        int l_pos = 0;
        int r_pos = 0;
        uint64_t qs_min_val = UINT64_MAX;
        int qs_min_pos = -1;
        int l = 0;
        uint64_t xk[2] = {0, 0};
        uint64_t xs[2] = {0, 0};
        for (size_t i = 0; i < len; i++) {
            int c = seq_nt4_table[(uint8_t) seq[i]];
            if (c < 4) { // not an "N" base
                xk[0] = (xk[0] << 2 | c) & kmask;                  // forward strand
                xk[1] = xk[1] >> 2 | (uint64_t)(3 - c) << kshift;  // reverse strand
                xs[0] = (xs[0] << 2 | c) & smask;                  // forward strand
                xs[1] = xs[1] >> 2 | (uint64_t)(3 - c) << sshift;  // reverse strand
                if (++l < s) {
                    continue;
                }
                // we find an s-mer
                uint64_t ys = xs[0] < xs[1] ? xs[0] : xs[1];
                uint64_t hash_s = syncmer_smer_hash(ys);
                gpu_qs[r_pos++] = hash_s;
                // not enough hashes in the queue, yet
                if (r_pos - l_pos < k - s + 1) {
                    continue;
                }
                if (r_pos - l_pos == k - s + 1) { // We are at the last s-mer within the first k-mer, need to decide if we add it
                    for (int j = l_pos; j < r_pos; j++) {
                        if (gpu_qs[j] < qs_min_val) {
                            qs_min_val = gpu_qs[j];
                            qs_min_pos = i - k + j - l_pos + 1;
                        }
                    }
                } else {
                    // update queue and current minimum and position
                    l_pos++;
                    if (qs_min_pos == i - k) { // we popped the previous minimizer, find new brute force
                        qs_min_val = UINT64_MAX;
                        qs_min_pos = i - s + 1;
                        for (int j = r_pos - 1; j >= l_pos; j--) { //Iterate in reverse to choose the rightmost minimizer in a window
                            if (gpu_qs[j] < qs_min_val) {
                                qs_min_val = gpu_qs[j];
                                qs_min_pos = i - k + j - l_pos + 1;
                            }
                        }
                    } else if (hash_s < qs_min_val) { // the new value added to queue is the new minimum
                        qs_min_val = hash_s;
                        qs_min_pos = i - s + 1;
                    }
                }
                if (qs_min_pos == i - k + t) { // occurs at t:th position in k-mer
                    uint64_t yk = xk[0] < xk[1] ? xk[0] : xk[1];
                    syncmers.push_back(Syncmer{syncmer_kmer_hash(yk), i - k + 1});
                }
            } else {
                // if there is an "N", restart
                qs_min_val = UINT64_MAX;
                qs_min_pos = -1;
                l = xs[0] = xs[1] = xk[0] = xk[1] = 0;
                r_pos = 0;
                l_pos = 0;
            }
        }


        const int w_min = index_para->randstrobe.w_min;
        const int w_max = index_para->randstrobe.w_max;
        const uint64_t q = index_para->randstrobe.q;
        const int max_dist = index_para->randstrobe.max_dist;

        my_vector<QueryRandstrobe> *randstrobes;
        randstrobes = (my_vector<QueryRandstrobe>*)my_malloc(sizeof(my_vector<QueryRandstrobe>));
        randstrobes->init((my_max(syncmers.size() - w_min, 0)) * 2);


        for (int strobe1_index = 0; strobe1_index + w_min < syncmers.size(); strobe1_index++) {
            unsigned int w_end = (strobe1_index + w_max < syncmers.size() - 1) ? (strobe1_index + w_max) : syncmers.size() - 1;
            auto strobe1 = syncmers[strobe1_index];
            auto max_position = strobe1.position + max_dist;
            unsigned int w_start = strobe1_index + w_min;
            uint64_t min_val = 0xFFFFFFFFFFFFFFFF;
            Syncmer strobe2 = strobe1;
            for (auto i = w_start; i <= w_end && syncmers[i].position <= max_position; i++) {
                uint64_t hash_diff = (strobe1.hash ^ syncmers[i].hash) & q;
                uint64_t res = __popcll(hash_diff);
                if (res < min_val) {
                    min_val = res;
                    strobe2 = syncmers[i];
                }
            }
            randstrobes->push_back(
                    QueryRandstrobe{
                            randstrobe_hash(strobe1.hash, strobe2.hash), static_cast<uint32_t>(strobe1.position),
                            static_cast<uint32_t>(strobe2.position) + index_para->syncmer.k, false
                    }
            );
        }


        for (int i = 0; i < syncmers.size() / 2; i++) {
            my_swap(syncmers[i], syncmers[syncmers.size() - i - 1]);
        }
        for (size_t i = 0; i < syncmers.size(); i++) {
            syncmers[i].position = len - syncmers[i].position - (*index_para).syncmer.k;
        }

        for (int strobe1_index = 0; strobe1_index + w_min < syncmers.size(); strobe1_index++) {
            unsigned int w_end = (strobe1_index + w_max < syncmers.size() - 1) ? (strobe1_index + w_max) : syncmers.size() - 1;
            auto strobe1 = syncmers[strobe1_index];
            auto max_position = strobe1.position + max_dist;
            unsigned int w_start = strobe1_index + w_min;
            uint64_t min_val = 0xFFFFFFFFFFFFFFFF;
            Syncmer strobe2 = strobe1;
            for (auto i = w_start; i <= w_end && syncmers[i].position <= max_position; i++) {
                uint64_t hash_diff = (strobe1.hash ^ syncmers[i].hash) & q;
                uint64_t res = __popcll(hash_diff);
                if (res < min_val) {
                    min_val = res;
                    strobe2 = syncmers[i];
                }
            }
            randstrobes->push_back(
                    QueryRandstrobe{
                            randstrobe_hash(strobe1.hash, strobe2.hash), static_cast<uint32_t>(strobe1.position),
                            static_cast<uint32_t>(strobe2.position) + index_para->syncmer.k, true
                    }
            );
        }


        randstrobe_sizes[id] += randstrobes->size();
        for (int i = 0; i < randstrobes->size(); i++) hashes[id] += (*randstrobes)[i].hash;
        global_randstrobes[id] = *randstrobes;
        my_free(randstrobes);
//        randstrobe_sizes[id] += syncmers.size();
    }
}

__global__ void gpu_get_hits_after(
        int bits,
        unsigned int filter_cutoff,
        int rescue_cutoff,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        const my_bucket_index_t *d_randstrobe_start_indices,
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_hits_num,
        my_vector<QueryRandstrobe>* global_randstrobes,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {

        int sum_seeds0 = 0;
        int sum_seeds1 = 0;
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
//            size_t position = global_randstrobes[id][i].hash;
//            if (position == static_cast<size_t>(-1)) continue;
//            bool res = gpu_is_filtered(d_randstrobes, d_randstrobes_size, position, filter_cutoff);
//            if (res) continue;
            if (global_randstrobes[id][i].is_reverse) {
                sum_seeds1++;
            } else {
                sum_seeds0++;
            }
        }
        my_vector<my_pair<int, Hit>>* hits_per_ref0;
        my_vector<my_pair<int, Hit>>* hits_per_ref1;
        hits_per_ref0 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref1 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref0->init(sum_seeds0 * 2);
        hits_per_ref1->init(sum_seeds1 * 2);

        uint64_t local_total_hits = 0;
        uint64_t local_nr_good_hits = 0;
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[id][i];
            size_t position = q.hash;
            if (position != static_cast<size_t>(-1)) {
                local_total_hits++;
                bool res = gpu_is_filtered(d_randstrobes, d_randstrobes_size, position, filter_cutoff);
                if (res) continue;
                local_nr_good_hits++;
                if(q.is_reverse) {
                    add_to_hits_per_ref(*hits_per_ref1, q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                } else {
                    add_to_hits_per_ref(*hits_per_ref0, q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                }
            }
        }
        float nonrepetitive_fraction = local_total_hits > 0 ? ((float) local_nr_good_hits) / ((float) local_total_hits) : 1.0;

        if (nonrepetitive_fraction < 0.7 || hits_per_ref0->size() + hits_per_ref1->size() == 0) {
        } else {
            global_randstrobes[id].release();
        }
        global_hits_num[id] = hits_per_ref0->size() + hits_per_ref1->size();
        hits_per_ref0s[id] = *hits_per_ref0;
        hits_per_ref1s[id] = *hits_per_ref1;
        my_free(hits_per_ref0);
        my_free(hits_per_ref1);
    }
}

__global__ void gpu_get_hits_pre(
        int bits,
        unsigned int filter_cutoff,
        int rescue_cutoff,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        const my_bucket_index_t *d_randstrobe_start_indices,
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_hits_num,
        my_vector<QueryRandstrobe>* global_randstrobes,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[id][i];
            size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
            //if(position != static_cast<size_t>(-1) && position >= d_randstrobes_size) {
            //    printf("position GG %zu %zu\n", position, d_randstrobes_size);
            //    assert(false);
            //}
            global_randstrobes[id][i].hash = position;
        }
    }
}

__global__ void gpu_get_hits(
        int bits,
        unsigned int filter_cutoff,
        int rescue_cutoff,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        const my_bucket_index_t *d_randstrobe_start_indices,
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_hits_num,
        my_vector<QueryRandstrobe>* global_randstrobes,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int read_id = id / 2;
        int rev = id % 2;

        my_vector<my_pair<int, Hit>>* hits_per_ref0;
        my_vector<my_pair<int, Hit>>* hits_per_ref1;
        hits_per_ref0 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref1 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref0->init();
        hits_per_ref1->init();

        uint64_t local_total_hits = 0;
        uint64_t local_nr_good_hits = 0;
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[id][i];
            size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
            global_randstrobes[id][i].hash = position;
            if (position != static_cast<size_t>(-1)) {
                local_total_hits++;
                bool res = gpu_is_filtered(d_randstrobes, d_randstrobes_size, position, filter_cutoff);
                if (res) continue;
                local_nr_good_hits++;
                if(q.is_reverse) {
                    add_to_hits_per_ref(*hits_per_ref1, q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                } else {
                    add_to_hits_per_ref(*hits_per_ref0, q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                }
            }
        }
        float nonrepetitive_fraction = local_total_hits > 0 ? ((float) local_nr_good_hits) / ((float) local_total_hits) : 1.0;

        if (nonrepetitive_fraction < 0.7 || hits_per_ref0->size() + hits_per_ref1->size() == 0) {

        } else {
            global_randstrobes[id].release();
        }
        global_hits_num[id] = hits_per_ref0->size() + hits_per_ref1->size();
        hits_per_ref0s[id] = *hits_per_ref0;
        hits_per_ref1s[id] = *hits_per_ref1;
        my_free(hits_per_ref0);
        my_free(hits_per_ref1);
    }
}

__global__ void gpu_get_randstrobes_and_hits(
        int bits,
        unsigned int filter_cutoff,
        int rescue_cutoff,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        const my_bucket_index_t *d_randstrobe_start_indices,
        IndexParameters *index_para,
        int num_tasks,
        int *pre_sum,
        int *lens,
        char *all_seqs,
        int *pre_sum2,
        int *lens2,
        char *all_seqs2,
        uint64_t *global_hits_num,
        my_vector<QueryRandstrobe>* global_randstrobes,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        int *randstrobe_sizes,
        uint64_t *hashes
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int read_id = id / 2;
        int rev = id % 2;
        size_t len;
        char *seq;
        if (rev == 0) {
            len = lens[read_id];
            seq = all_seqs + pre_sum[read_id];
        } else {
            len = lens2[read_id];
            seq = all_seqs2 + pre_sum2[read_id];
        }

        my_vector<QueryRandstrobe> *randstrobes;
        randstrobes = (my_vector<QueryRandstrobe>*)my_malloc(sizeof(my_vector<QueryRandstrobe>));
        randstrobes->init();

        my_vector<Syncmer> syncmers;
        my_vector<uint64_t> vec4syncmers;

        SyncmerIterator syncmer_iterator{&vec4syncmers, seq, len, (*index_para).syncmer};
        Syncmer syncmer;
        while (1) {
            syncmer = syncmer_iterator.gpu_next();
            if (syncmer.is_end()) break;
            syncmers.push_back(syncmer);
        }
        if (syncmers.size() != 0)  {
            RandstrobeIterator randstrobe_fwd_iter{&syncmers, (*index_para).randstrobe};
            while (randstrobe_fwd_iter.gpu_has_next()) {
                Randstrobe randstrobe = randstrobe_fwd_iter.gpu_next();
                randstrobes->push_back(
                        QueryRandstrobe{
                                randstrobe.hash, randstrobe.strobe1_pos,
                                randstrobe.strobe2_pos + (*index_para).syncmer.k, false
                        }
                );
            }
            for (int i = 0; i < syncmers.size() / 2; i++) {
                my_swap(syncmers[i], syncmers[syncmers.size() - i - 1]);
            }
            for (size_t i = 0; i < syncmers.size(); i++) {
                syncmers[i].position = len - syncmers[i].position - (*index_para).syncmer.k;
            }
            RandstrobeIterator randstrobe_rc_iter{&syncmers, (*index_para).randstrobe};
            while (randstrobe_rc_iter.gpu_has_next()) {
                Randstrobe randstrobe = randstrobe_rc_iter.gpu_next();
                randstrobes->push_back(
                        QueryRandstrobe{
                                randstrobe.hash, randstrobe.strobe1_pos,
                                randstrobe.strobe2_pos + (*index_para).syncmer.k, true
                        }
                );
            }
        }

        randstrobe_sizes[id] += randstrobes->size();
        for (int i = 0; i < randstrobes->size(); i++) hashes[id] += (*randstrobes)[i].hash;

        my_vector<my_pair<int, Hit>>* hits_per_ref0;
        my_vector<my_pair<int, Hit>>* hits_per_ref1;
        hits_per_ref0 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref1 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref0->init();
        hits_per_ref1->init();

        uint64_t local_total_hits = 0;
        uint64_t local_nr_good_hits = 0;
        for (int i = 0; i < randstrobes->size(); i++) {
            QueryRandstrobe q = (*randstrobes)[i];
            size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
            if (position != static_cast<size_t>(-1)) {
                local_total_hits++;
                bool res = gpu_is_filtered(d_randstrobes, d_randstrobes_size, position, filter_cutoff);
                if (res) continue;
                local_nr_good_hits++;
                if(q.is_reverse) {
                    add_to_hits_per_ref(*hits_per_ref1, q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                } else {
                    add_to_hits_per_ref(*hits_per_ref0, q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                }
            }
        }
        float nonrepetitive_fraction = local_total_hits > 0 ? ((float) local_nr_good_hits) / ((float) local_total_hits) : 1.0;
        if (nonrepetitive_fraction < 0.7 || hits_per_ref0->size() + hits_per_ref1->size() == 0) {
            global_randstrobes[id] = *randstrobes;
        } else {
            randstrobes->release();
        }
        global_hits_num[id] = hits_per_ref0->size() + hits_per_ref1->size();
        hits_per_ref0s[id] = *hits_per_ref0;
        hits_per_ref1s[id] = *hits_per_ref1;
        my_free(hits_per_ref0);
        my_free(hits_per_ref1);
        my_free(randstrobes);
    }
}

__global__ void gpu_sort_hits(
        int num_tasks,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        sort_hits_single(hits_per_ref0s[id]);
        sort_hits_single(hits_per_ref1s[id]);
        //sort_hits_by_refid(hits_per_ref0s[id]);
        //sort_hits_by_refid(hits_per_ref1s[id]);
    }
}

__global__ void gpu_merge_hits(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_nams_info,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        my_vector<Nam> nams;
        salign_merge_hits(hits_per_ref0s[id], index_para->syncmer.k, 0, nams);
        salign_merge_hits(hits_per_ref1s[id], index_para->syncmer.k, 1, nams);
        uint64_t local_nams_info = 0;
        for (int i = 0; i < nams.size(); i++) {
            local_nams_info += nams[i].ref_id + int(nams[i].score) + nams[i].query_start + nams[i].query_end;
        }
        global_nams_info[id] += local_nams_info;
        hits_per_ref0s[id].release();
        hits_per_ref1s[id].release();
    }
}

__global__ void gpu_align_PE(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_align_info,
        AlignmentParameters* aligner_parameters,
        int *pre_sum,
        int *lens,
        char *all_seqs,
        int *pre_sum2,
        int *lens2,
        char *all_seqs2,
        GPUReferences *global_references,
        MappingParameters *mapping_parameters,
        my_vector<Nam> *global_nams
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        size_t seq_len1, seq_len2;
        char *seq1, *seq2;
        seq_len1 = lens[id];
        seq1 = all_seqs + pre_sum[id];
        seq_len2 = lens2[id];
        seq2 = all_seqs2 + pre_sum2[id];

        GPUAlignTmpRes align_tmp_res;
        align_tmp_res.init();
        GPUInsertSizeDistribution isize_est;
        align_PE_part(align_tmp_res, *aligner_parameters, global_nams[id], global_nams[id + num_tasks], seq1, seq_len1, seq2, seq_len2, index_para->syncmer.k, *global_references,
                      mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary);
        global_align_info[id] += align_tmp_res.type + align_tmp_res.mapq1 + align_tmp_res.mapq2 + align_tmp_res.type4_loop_size;
        global_align_info[id] += align_tmp_res.is_extend_seed.size() + align_tmp_res.consistent_nam.size() + align_tmp_res.is_read1.size() +
                align_tmp_res.type4_nams.size() + align_tmp_res.todo_nams.size() + align_tmp_res.done_align.size() + align_tmp_res.align_res.size();
        uint64_t local_sum = 0;
        for (int i = 0; i < align_tmp_res.todo_nams.size(); i++) {
            local_sum += align_tmp_res.todo_nams[i].ref_id;
        }
        global_align_info[id] += local_sum;
        global_nams[id].release();
        global_nams[id + num_tasks].release();
    }
}

__global__ void gpu_merge_hits_get_nams(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_nams_info,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        my_vector<Nam> *global_nams
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int real_id = id;
        my_vector<Nam> *nams1 = (my_vector<Nam>*)my_malloc(sizeof(my_vector<Nam>));
        nams1->init();
        salign_merge_hits(hits_per_ref0s[real_id], index_para->syncmer.k, 0, *nams1);
        salign_merge_hits(hits_per_ref1s[real_id], index_para->syncmer.k, 1, *nams1);
        uint64_t local_nams_info = 0;
        for (int i = 0; i < nams1->size(); i++) {
            local_nams_info += (*nams1)[i].ref_id + int((*nams1)[i].score) + (*nams1)[i].query_start + (*nams1)[i].query_end;
        }
        global_nams_info[real_id] += local_nams_info;
        global_nams[real_id] = *nams1;
        my_free(nams1);
        hits_per_ref0s[real_id].release();
        hits_per_ref1s[real_id].release();

        real_id = id + num_tasks;
        my_vector<Nam> *nams2 = (my_vector<Nam>*)my_malloc(sizeof(my_vector<Nam>));
        nams2->init();
        salign_merge_hits(hits_per_ref0s[real_id], index_para->syncmer.k, 0, *nams2);
        salign_merge_hits(hits_per_ref1s[real_id], index_para->syncmer.k, 1, *nams2);
        local_nams_info = 0;
        for (int i = 0; i < nams2->size(); i++) {
            local_nams_info += (*nams2)[i].ref_id + int((*nams2)[i].score) + (*nams2)[i].query_start + (*nams2)[i].query_end;
        }
        global_nams_info[real_id] += local_nams_info;
        global_nams[real_id] = *nams2;
        my_free(nams2);
        hits_per_ref0s[real_id].release();
        hits_per_ref1s[real_id].release();
    }
}


klibpp::KSeq ConvertNeo2KSeq(neoReference ref) {
    klibpp::KSeq res;
    res.name = std::string((char *) ref.base + ref.pname, ref.lname);
    if (!res.name.empty()) {
        size_t space_pos = res.name.find(' ');
        int l_pos = 0;
        if (res.name[0] == '@') l_pos = 1;
        if (space_pos != std::string::npos) {
            res.name = res.name.substr(l_pos, space_pos - l_pos);
        } else {
            res.name = res.name.substr(l_pos);
        }
    }
    res.seq = std::string((char *) ref.base + ref.pseq, ref.lseq);
    res.comment = std::string((char *) ref.base + ref.pstrand, ref.lstrand);
    res.qual = std::string((char *) ref.base + ref.pqual, ref.lqual);
    return res;
}

static Logger &logger = Logger::get();


int main(int argc, char **argv) {
    auto opt = parse_command_line_arguments(argc, argv);

    InputBuffer input_buffer = get_input_buffer(opt);
    if (!opt.r_set && !opt.reads_filename1.empty()) {
        opt.r = estimate_read_length(input_buffer);
        logger.info() << "Estimated read length: " << opt.r << " bp\n";

    }

    IndexParameters index_parameters = IndexParameters::from_read_length(
            opt.r,
            opt.k_set ? opt.k : IndexParameters::DEFAULT,
            opt.s_set ? opt.s : IndexParameters::DEFAULT,
            opt.l_set ? opt.l : IndexParameters::DEFAULT,
            opt.u_set ? opt.u : IndexParameters::DEFAULT,
            opt.c_set ? opt.c : IndexParameters::DEFAULT,
            opt.max_seed_len_set ? opt.max_seed_len : IndexParameters::DEFAULT
    );
    AlignmentParameters aln_params;
    aln_params.match = opt.A;
    aln_params.mismatch = opt.B;
    aln_params.gap_open = opt.O;
    aln_params.gap_extend = opt.E;
    aln_params.end_bonus = opt.end_bonus;

    AlignmentParameters *d_aligner;
    cudaMalloc(&d_aligner, sizeof(AlignmentParameters));
    cudaMemcpy(d_aligner, &aln_params, sizeof(AlignmentParameters), cudaMemcpyHostToDevice);


    MappingParameters map_param;
    map_param.r = opt.r;
    map_param.max_secondary = opt.max_secondary;
    map_param.dropoff_threshold = opt.dropoff_threshold;
    map_param.rescue_level = opt.rescue_level;
    map_param.max_tries = opt.max_tries;
    map_param.is_sam_out = opt.is_sam_out;
    map_param.cigar_ops = opt.cigar_eqx ? CigarOps::EQX : CigarOps::M;
    map_param.output_unmapped = opt.output_unmapped;
    map_param.details = opt.details;
    map_param.verify();

    MappingParameters* d_map_param;
    cudaMalloc(&d_map_param, sizeof(MappingParameters));
    cudaMemcpy(d_map_param, &map_param, sizeof(MappingParameters), cudaMemcpyHostToDevice);

    std::string index_file_path = argv[1];
    References references;
    references = References::from_fasta(opt.ref_filename);
    logger.info() << "Reference size: " << references.total_length() / 1E6 << " Mbp ("
                  << references.size() << " contig" << (references.size() == 1 ? "" : "s")
                  << "; largest: "
                  << (*std::max_element(references.lengths.begin(), references.lengths.end()) / 1E6) << " Mbp)\n";
    if (references.total_length() == 0) {
        throw InvalidFasta("No reference sequences found");
    }
    StrobemerIndex index(references, index_parameters, opt.bits);
    std::string sti_path = opt.ref_filename + index_parameters.filename_extension();
    index.read(sti_path);

    int para_rescue_cutoff = opt.rescue_level < 100 ? opt.rescue_level * index.filter_cutoff : rescue_threshold;

    std::cout << "rescue_cutoff: " << para_rescue_cutoff << std::endl;
    std::cout << "filter_cutoff: " << index.filter_cutoff << std::endl;
    std::cout << "rescue_level: " << opt.rescue_level << std::endl;

    std::cout << "read file : " << opt.reads_filename1 << " " << opt.reads_filename2 << std::endl;

    rabbit::fq::FastqDataPool fastqPool(1024, 1 << 22);
    rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> queue_pe(1024, 1);
    std::thread *producer;
    producer = new std::thread(producer_pe_fastq_task, opt.reads_filename1, opt.reads_filename2, std::ref(fastqPool),
                               std::ref(queue_pe));

    std::vector<neoReference> data1;
    std::vector<neoReference> data2;
    rabbit::fq::FastqDataPairChunk *fqdatachunk = new rabbit::fq::FastqDataPairChunk;
    std::vector<klibpp::KSeq> records1;
    std::vector<klibpp::KSeq> records2;
    long long id;
    while (queue_pe.Pop(id, fqdatachunk)) {
        data1.clear();
        data2.clear();
        rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk *) (fqdatachunk->left_part), data1);
        rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk *) (fqdatachunk->right_part), data2);
        assert(data1.size() == data2.size());
        for (int i = 0; i < data1.size(); i++) {
            auto item1 = data1[i];
            auto item2 = data2[i];
            records1.push_back(ConvertNeo2KSeq(item1));
            records2.push_back(ConvertNeo2KSeq(item2));
        }
        fastqPool.Release(fqdatachunk->left_part);
        fastqPool.Release(fqdatachunk->right_part);
    }
    producer->join();
    printf("read file done, %zu %zu\n", records1.size(), records2.size());

    double t0;

    t0 = GetTime();
    GPUReferences *global_references;
    cudaMallocManaged(&global_references, sizeof(GPUReferences));
    global_references->num_refs = references.size();
    cudaMalloc(&global_references->sequences.data, references.size() * sizeof(my_string));
    global_references->sequences.length = references.size();
    global_references->sequences.capacity = references.size();
    for (int i = 0; i < references.size(); i++) {
        my_string ref;
        ref.slen = references.lengths[i];
        cudaMalloc(&ref.data, references.lengths[i]);
        cudaMemcpy(ref.data, references.sequences[i].data(), references.lengths[i], cudaMemcpyHostToDevice);
        cudaMemcpy(global_references->sequences.data + i, &ref, sizeof(my_string), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&global_references->lengths.data, references.size() * sizeof(int));
    cudaMemcpy(global_references->lengths.data, references.lengths.data(), references.size() * sizeof(int), cudaMemcpyHostToDevice);
    global_references->lengths.length = references.size();
    global_references->lengths.capacity = references.size();

    RefRandstrobe *d_randstrobes;
    my_bucket_index_t *d_randstrobe_start_indices;
    std::cout << index.randstrobes.size() * sizeof(RefRandstrobe) << std::endl;
    cudaMalloc(&d_randstrobes, index.randstrobes.size() * sizeof(RefRandstrobe));
    cudaMalloc(&d_randstrobe_start_indices, index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t));
    cudaMemset(d_randstrobes, 0, index.randstrobes.size() * sizeof(RefRandstrobe));
    cudaMemset(d_randstrobe_start_indices, 0, index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t));
    std::cout << "malloc1 execution time: " << GetTime() - t0 << " seconds, size "
              << index.randstrobes.size() * sizeof(RefRandstrobe) +
                 index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t) << std::endl;

    t0 = GetTime();
    cudaMemcpy(d_randstrobes, index.randstrobes.data(), index.randstrobes.size() * sizeof(RefRandstrobe),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_randstrobe_start_indices, index.randstrobe_start_indices.data(),
               index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t), cudaMemcpyHostToDevice);
    std::cout << "memcpy1 execution time: " << GetTime() - t0 << " seconds, size "
              << index.randstrobes.size() * sizeof(RefRandstrobe) +
                 index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t) << std::endl;


#define batch_size 400000ll
#define batch_seq_szie batch_size * 250ll


    uint64_t num_bytes = 24ll * 1024ll * 1024ll * 1024ll;
    uint64_t seed = 13;
    init_mm(num_bytes, seed);

    printf("Gallatin global allocator initialized with %lu bytes.\n", num_bytes);

    my_vector<QueryRandstrobe> *global_randstrobes;
    cudaMallocManaged(&global_randstrobes, batch_size * 2 * sizeof(my_vector<QueryRandstrobe>));

    int *global_sorted_ids;
    cudaMallocManaged(&global_sorted_ids, batch_size * 4 * sizeof(int));

    int *global_randstrobe_sizes;
    cudaMallocManaged(&global_randstrobe_sizes, batch_size * 2 * sizeof(int));
    uint64_t * global_hashes_value;
    cudaMallocManaged(&global_hashes_value, batch_size * 2 * sizeof(uint64_t));

    my_vector<my_pair<int, Hit>> *global_hits_per_ref0s;
    cudaMallocManaged(&global_hits_per_ref0s, batch_size * 2 * sizeof(my_vector<my_pair<int, Hit>>));

    my_vector<my_pair<int, Hit>> *global_hits_per_ref1s;
    cudaMallocManaged(&global_hits_per_ref1s, batch_size * 2 * sizeof(my_vector<my_pair<int, Hit>>));

    my_vector<Nam> *global_nams;
    cudaMallocManaged(&global_nams, batch_size * 2 * sizeof(my_vector<Nam>));


    t0 = GetTime();
    char *d_seq;
    int *d_len;
    int *d_pre_sum;
    cudaMalloc(&d_seq, batch_seq_szie);
    cudaMemset(d_seq, 0, batch_seq_szie);
    cudaMalloc(&d_len, batch_size * sizeof(int));
    cudaMemset(d_len, 0, batch_size * sizeof(int));
    cudaMalloc(&d_pre_sum, batch_size * sizeof(int));
    cudaMemset(d_pre_sum, 0, batch_size * sizeof(int));

    char *d_seq2;
    int *d_len2;
    int *d_pre_sum2;
    cudaMalloc(&d_seq2, batch_seq_szie);
    cudaMemset(d_seq2, 0, batch_seq_szie);
    cudaMalloc(&d_len2, batch_size * sizeof(int));
    cudaMemset(d_len2, 0, batch_size * sizeof(int));
    cudaMalloc(&d_pre_sum2, batch_size * sizeof(int));
    cudaMemset(d_pre_sum2, 0, batch_size * sizeof(int));
    std::cout << "malloc2 execution time: " << GetTime() - t0 << " seconds, size "
              << batch_seq_szie + batch_size * sizeof(int) << std::endl;

    IndexParameters *d_index_para;
    cudaMalloc(&d_index_para, sizeof(IndexParameters));


    int *h_len = new int[batch_size];
    int *h_pre_sum = new int[batch_size + 1];
    char *h_seq = new char[batch_seq_szie];

    int *h_len2 = new int[batch_size];
    int *h_pre_sum2 = new int[batch_size + 1];
    char *h_seq2 = new char[batch_seq_szie];

    double gpu_cost1 = 0;
    double gpu_cost2 = 0;
    double gpu_cost2_1 = 0;
    double gpu_cost2_2 = 0;
    double gpu_cost3 = 0;
    double gpu_cost4 = 0;
    double gpu_cost5 = 0;
    double gpu_cost6 = 0;
    double gpu_cost7 = 0;
    double gpu_cost8 = 0;
    double cpu_sort = 0;
    double tot_cost = 0;

    uint64_t check_sum = 0;
    uint64_t size_tot = 0;

    uint64_t global_hits_num12 = 0;
    uint64_t global_hits_num3 = 0;

    uint64_t * global_hits_num;
    cudaMallocManaged(&global_hits_num, batch_size * 2 * sizeof(uint64_t));

    uint64_t global_nams_info12 = 0;
    uint64_t global_nams_info3 = 0;
    uint64_t * global_nams_info;
    cudaMallocManaged(&global_nams_info, batch_size * 2 * sizeof(uint64_t));

    uint64_t global_align_info12 = 0;
    uint64_t global_align_info3 = 0;
    uint64_t * global_align_info;
    cudaMallocManaged(&global_align_info, batch_size * sizeof(uint64_t));



    assert(records1.size() == records2.size());

//    print_mm();

    t0 = GetTime();
    for (int l_id = 0; l_id < records1.size(); l_id += batch_size) {
        printf("process %d / %d\n", l_id, records1.size());
        int r_id = l_id + batch_size;
        if (r_id > records1.size()) r_id = records1.size();
        int s_len = r_id - l_id;

        double t1;

        uint64_t tot_len = 0;
        uint64_t tot_len2 = 0;
        h_pre_sum[0] = 0;
        h_pre_sum2[0] = 0;
        for (int i = l_id; i < r_id; i++) {
            tot_len += records1[i].seq.length();
            tot_len2 += records2[i].seq.length();
            h_len[i - l_id] = records1[i].seq.length();
            h_len2[i - l_id] = records2[i].seq.length();
            h_pre_sum[i + 1 - l_id] = h_pre_sum[i - l_id] + h_len[i - l_id];
            h_pre_sum2[i + 1 - l_id] = h_pre_sum2[i - l_id] + h_len2[i - l_id];
        }
#pragma omp parallel for
        for (int i = l_id; i < r_id; i++) {
            memcpy(h_seq + h_pre_sum[i - l_id], records1[i].seq.c_str(), h_len[i - l_id]);
            memcpy(h_seq2 + h_pre_sum2[i - l_id], records2[i].seq.c_str(), h_len2[i - l_id]);
        }

        cudaMemcpy(d_seq, h_seq, tot_len, cudaMemcpyHostToDevice);
        cudaMemcpy(d_len, h_len, s_len * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pre_sum, h_pre_sum, s_len * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(d_seq2, h_seq2, tot_len2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_len2, h_len2, s_len * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pre_sum2, h_pre_sum2, s_len * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(d_index_para, &index_parameters, sizeof(IndexParameters), cudaMemcpyHostToDevice);

        for (int i = 0; i < s_len; i++) {
            global_align_info[i] = 0;
        }

        for (int i = 0; i < s_len * 2; i++) {
            // check infos
            global_randstrobe_sizes[i] = 0;
            global_hashes_value[i] = 0;
            global_hits_num[i] = 0;
            global_nams_info[i] = 0;

            global_hits_per_ref0s[i].data = nullptr;
            global_hits_per_ref0s[i].length = 0;
            global_hits_per_ref1s[i].data = nullptr;
            global_hits_per_ref1s[i].length = 0;

            global_randstrobes[i].data = nullptr;
            global_randstrobes[i].length = 0;

            global_nams[i].data = nullptr;
            global_nams[i].length = 0;
        }


        t1 = GetTime();
        int threads_per_block;
        int reads_per_block;
        int blocks_per_grid;

        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_get_randstrobes<<<blocks_per_grid, threads_per_block>>>(s_len * 2, d_pre_sum, d_len, d_seq, d_pre_sum2, d_len2, d_seq2, d_index_para,
                                                                    global_randstrobe_sizes, global_hashes_value, global_randstrobes);
        cudaDeviceSynchronize();
        gpu_cost1 += GetTime() - t1;

        t1 = GetTime();

        double t11 = GetTime();
        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_get_hits_pre<<<blocks_per_grid, threads_per_block>>>(index.bits, index.filter_cutoff, para_rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                             s_len * 2, d_index_para, global_hits_num, global_randstrobes,
                                                             global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost2_1 += GetTime() - t11;

      
        t11 = GetTime();
        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_get_hits_after<<<blocks_per_grid, threads_per_block>>>(index.bits, index.filter_cutoff, para_rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                             s_len * 2, d_index_para, global_hits_num, global_randstrobes,
                                                             global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost2_2 += GetTime() - t11;

        gpu_cost2 += GetTime() - t1;


        t1 = GetTime();
        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_sort_hits<<<blocks_per_grid, threads_per_block>>>(s_len * 2, global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost3 += GetTime() - t1;

        t1 = GetTime();
        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_merge_hits_get_nams<<<blocks_per_grid, threads_per_block>>>(s_len, d_index_para, global_nams_info, global_hits_per_ref0s, global_hits_per_ref1s, global_nams);
        cudaDeviceSynchronize();
        gpu_cost4 += GetTime() - t1;
        printf("get nams done\n");

        t1 = GetTime();
        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_align_PE<<<blocks_per_grid, threads_per_block>>>(s_len, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq, d_pre_sum2, d_len2, d_seq2,
                                                             global_references, d_map_param, global_nams);
        cudaDeviceSynchronize();
        gpu_cost5 += GetTime() - t1;


        for (size_t i = 0; i < s_len * 2; ++i) {
            size_tot += global_randstrobe_sizes[i];
            check_sum += global_hashes_value[i];
            global_hits_num12 += global_hits_num[i];
            global_nams_info12 += global_nams_info[i];
        }

        for (int i = 0; i < s_len; i++) {
            global_align_info12 += global_align_info[i];
        }

		int rescue_num = 0;
        for(int i = 0; i < s_len * 2; i++) {
            if (global_randstrobes[i].data != nullptr) {
                global_randstrobes[rescue_num] = global_randstrobes[i];
                rescue_num++;
            }
        }

        printf("rescue read num %d\n", rescue_num);

        for (int i = 0; i < rescue_num; i++) {
            global_hits_num[i] = 0;
            global_nams_info[i] = 0;

            global_hits_per_ref0s[i].data = nullptr;
            global_hits_per_ref0s[i].length = 0;
            global_hits_per_ref1s[i].data = nullptr;
            global_hits_per_ref1s[i].length = 0;
        }

        t1 = GetTime();
        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (rescue_num + reads_per_block - 1) / reads_per_block;
        gpu_rescue_get_hits<<<blocks_per_grid, threads_per_block>>>(index.bits, index.filter_cutoff, para_rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                             rescue_num, d_index_para, global_hits_num, global_randstrobes,
                                                             global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost6 += GetTime() - t1;

        t1 = GetTime();
        threads_per_block = 1;
        reads_per_block = GPU_thread_task_size;
        blocks_per_grid = (rescue_num + reads_per_block - 1) / reads_per_block;
        //gpu_rescue_sort_hits<<<rescue_num, threads_per_block>>>(rescue_num, d_index_para, global_hits_per_ref0s, global_hits_per_ref1s);
        gpu_sort_hits<<<blocks_per_grid, threads_per_block>>>(rescue_num, global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost7 += GetTime() - t1;



        t1 = GetTime();
        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (rescue_num + reads_per_block - 1) / reads_per_block;
        gpu_rescue_merge_hits<<<blocks_per_grid, threads_per_block>>>(rescue_num, d_index_para, global_nams_info, global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost8 += GetTime() - t1;

        for (int i = 0; i < rescue_num; ++i) {
            global_hits_num3 += global_hits_num[i];
            global_nams_info3 += global_nams_info[i];
        }

//        print_mm();

    }
    tot_cost += GetTime() - t0;

    std::cout << "gpu cost " << gpu_cost1 << " " << gpu_cost2 << " [" << gpu_cost2_1 << " " << gpu_cost2_2 << "] " << gpu_cost3 << " " << gpu_cost4 << " " << gpu_cost5 << std::endl;
    std::cout << gpu_cost6 << " " << gpu_cost7 << " " << gpu_cost8 << std::endl;
    std::cout << "cpu sort cost " << cpu_sort << std::endl;
    std::cout << "total cost " << tot_cost << std::endl;
    std::cout << "check_sum : " << check_sum << ", size_tot : " << size_tot << std::endl;
    std::cout << "total_hits12 : " << global_hits_num12 << ", nr_good_hits12 : " << global_nams_info12 << std::endl;
    std::cout << "total_hits3 : " << global_hits_num3 << ", nr_good_hits3 : " << global_nams_info3 << std::endl;
    std::cout << "total_align_info12 : " << global_align_info12 << ", nr_good_align_info3 : " << global_align_info3 << std::endl;

    t0 = GetTime();
    cudaFree(d_seq);
    cudaFree(d_len);
    cudaFree(d_pre_sum);
    cudaFree(d_index_para);
    cudaFree(d_randstrobes);
    cudaFree(d_randstrobe_start_indices);
    delete h_seq;
    delete h_len;
    delete h_pre_sum;
    std::cout << "free execution time: " << GetTime() - t0 << " seconds" << std::endl;

    free_mm();
    return 0;
}

