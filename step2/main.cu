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

#define rescue_threshold 1000

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


__device__ randstrobe_hash_t
gpu_get_hash(const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, my_bucket_index_t position) {
    if (position < d_randstrobes_size) {
        return d_randstrobes[position].hash;
    } else {
        return static_cast<randstrobe_hash_t>(-1);
    }
}

__device__ bool
gpu_is_filtered(const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, my_bucket_index_t position,
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

__device__ size_t gpu_find(
        const RefRandstrobe *d_randstrobes,
        const my_bucket_index_t *d_randstrobe_start_indices,
        const randstrobe_hash_t key,
        int bits
) {
    const unsigned int top_N = key >> (64 - bits);
    my_bucket_index_t position_start = d_randstrobe_start_indices[top_N];
    my_bucket_index_t position_end = d_randstrobe_start_indices[top_N + 1];
    if (position_start == position_end) {
        return static_cast<size_t>(-1); // No match
    }
    for (my_bucket_index_t i = position_start; i < position_end; ++i) {
        if (d_randstrobes[i].hash == key) {
            return i;
        }
    }
    return static_cast<size_t>(-1); // No match
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
__device__ void quick_sort_iterative(T* data, int low, int high) {
    int vec_size = high - low + 1;
    if(vec_size == 0) return;
    my_vector<int>stack_vec(vec_size * 2);
    int* stack = stack_vec.data;
    int top = -1;
    stack[++top] = low;
    stack[++top] = high;
    int mx_top = 0;
    while (top >= 0) {
        high = stack[top--];
        low = stack[top--];
        // Partition
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
        int pivot_index = i + 1;
        if (pivot_index - 1 > low) {
            stack[++top] = low;
            stack[++top] = pivot_index - 1;
        }
        if (pivot_index + 1 < high) {
            stack[++top] = pivot_index + 1;
            stack[++top] = high;
        }
        mx_top = my_max(mx_top, top);
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
    //bubble_sort(data, size);

}

struct Rescue_Seeds {
    int read_id;
    int read_fr;
    int seeds_num;
    QueryRandstrobe* seeds;
};



#define BLOCK_SIZE 32


__device__ void sort_hits_single(
        my_vector<my_pair<int, Hit>>& hits_per_ref
) {
    bubble_sort(&(hits_per_ref[0]), hits_per_ref.size());
    //quick_sort(&(hits_per_ref[0]), hits_per_ref.size());
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

    my_vector<Nam> open_nams;

    int now_vec_pos = 0;
    for (int rid = 0; rid < ref_num; rid++) {
        if(rid != 0) now_vec_pos += each_ref_size[rid - 1];
        int ref_id = hits_per_ref[now_vec_pos].first;
        open_nams.clear();
        unsigned int prev_q_start = 0;
        size_t hits_size = each_ref_size[rid];
        my_pair<int, Hit>* hits = &(hits_per_ref[now_vec_pos]);
        for(int i = 0; i < hits_size; i++) {
            assert(hits[i].first == hits_per_ref[now_vec_pos + i].first);
            assert(hits[i].second == hits_per_ref[now_vec_pos + i].second);
        }
        for (size_t i = 0; i < hits_size; ) {
            size_t i_start = i;
            size_t i_end = i + 1;
            size_t i_size;
            while(i_end < hits_size && hits[i_end].second.query_start == hits[i].second.query_start) i_end++;
            i = i_end;
            i_size = i_end - i_start;
            my_vector<bool> is_added(i_size);
//            const int mx_i_size = 1024;
//            bool is_added[mx_i_size];
//            if(i_size >= mx_i_size) {
//                printf("i_size > mx_i_size : %d %d\n", i_size, mx_i_size);
//                assert(false);
//            }
//            for(size_t j = 0; j < i_size; j++) is_added[j] = false;
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
                //            if (1) {

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

__device__ void merge_hits(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int k,
        bool is_revcomp,
        my_vector<Nam>& nams
) {
    if(hits_per_ref.size() == 0) return;
    unsigned long long t_start;
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

        unsigned long long t_start1;
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
    //printf("pos %llu, [%d %d]\n", position, query_start, query_end);
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

__device__ int lock = 0;

__device__ void acquire_lock() {
    while (atomicCAS(&lock, 0, 1) != 0) {
    }
}

__device__ void release_lock() {
    atomicExch(&lock, 0);
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
        Rescue_Seeds *rescue_seeds,
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
        int read_id = rescue_seeds[id].read_id;
        int rev = rescue_seeds[id].read_fr;

        my_vector<my_pair<int, Hit>>* hits_per_ref0;
        my_vector<my_pair<int, Hit>>* hits_per_ref1;
        hits_per_ref0 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref1 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref0->init();
        hits_per_ref1->init();

        my_vector<RescueHit> hits_t0;
        my_vector<RescueHit> hits_t1;
        for (int i = 0; i < rescue_seeds[id].seeds_num; i++) {
            QueryRandstrobe q = rescue_seeds[id].seeds[i];
            size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
            if (position != static_cast<size_t>(-1)) {
                unsigned int count = gpu_get_count(d_randstrobes, d_randstrobe_start_indices, position, bits);
                RescueHit rh{position, count, q.start, q.end};
                if(q.is_reverse) hits_t1.push_back(rh);
                else hits_t0.push_back(rh);
            }
        }
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
        Rescue_Seeds *rescue_seeds,
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
        int read_id = rescue_seeds[id].read_id;
        int rev = rescue_seeds[id].read_fr;
        sort_hits_parallel(hits_per_ref0s[id], index_para->syncmer.k, 0, tid);
        sort_hits_parallel(hits_per_ref1s[id], index_para->syncmer.k, 1, tid);
    }
}

__global__ void gpu_rescue_merge_hits(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_nams_info,
        Rescue_Seeds *rescue_seeds,
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
        int read_id = rescue_seeds[id].read_id;
        int rev = rescue_seeds[id].read_fr;
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
        global_randstrobes[id] = *randstrobes;
        my_free(randstrobes);
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
        Rescue_Seeds *rescue_seeds,
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
            rescue_seeds[id].read_id = read_id;
            rescue_seeds[id].read_fr = rev;
            rescue_seeds[id].seeds_num = global_randstrobes[id].size();
            for (int i = 0; i < global_randstrobes[id].size(); i++) {
                rescue_seeds[id].seeds[i] = global_randstrobes[id][i];
            }
        }
        global_hits_num[id] = hits_per_ref0->size() + hits_per_ref1->size();
        hits_per_ref0s[id] = *hits_per_ref0;
        hits_per_ref1s[id] = *hits_per_ref1;
        my_free(hits_per_ref0);
        my_free(hits_per_ref1);
        global_randstrobes[id].release();
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
        Rescue_Seeds *rescue_seeds,
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

        my_vector<QueryRandstrobe> randstrobes;
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
                randstrobes.push_back(
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
                randstrobes.push_back(
                        QueryRandstrobe{
                                randstrobe.hash, randstrobe.strobe1_pos,
                                randstrobe.strobe2_pos + (*index_para).syncmer.k, true
                        }
                );
            }
        }

        randstrobe_sizes[id] += randstrobes.size();
        for (int i = 0; i < randstrobes.size(); i++) hashes[id] += randstrobes[i].hash;

        my_vector<my_pair<int, Hit>>* hits_per_ref0;
        my_vector<my_pair<int, Hit>>* hits_per_ref1;
        hits_per_ref0 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref1 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref0->init();
        hits_per_ref1->init();

        uint64_t local_total_hits = 0;
        uint64_t local_nr_good_hits = 0;
        for (int i = 0; i < randstrobes.size(); i++) {
            QueryRandstrobe q = randstrobes[i];
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
            rescue_seeds[id].read_id = read_id;
            rescue_seeds[id].read_fr = rev;
            rescue_seeds[id].seeds_num = randstrobes.size();
            for (int i = 0; i < randstrobes.size(); i++) {
                rescue_seeds[id].seeds[i] = randstrobes[i];
            }
        }
        global_hits_num[id] = hits_per_ref0->size() + hits_per_ref1->size();
        hits_per_ref0s[id] = *hits_per_ref0;
        hits_per_ref1s[id] = *hits_per_ref1;
        my_free(hits_per_ref0);
        my_free(hits_per_ref1);
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
        int read_id = id / 2;
        int rev = id % 2;
        sort_hits_single(hits_per_ref0s[id]);
        sort_hits_single(hits_per_ref1s[id]);
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
        int read_id = id / 2;
        int rev = id % 2;
        my_vector<Nam> nams;
        merge_hits(hits_per_ref0s[read_id * 2 + rev], index_para->syncmer.k, 0, nams);
        merge_hits(hits_per_ref1s[read_id * 2 + rev], index_para->syncmer.k, 1, nams);
        uint64_t local_nams_info = 0;
        for (int i = 0; i < nams.size(); i++) {
            local_nams_info += nams[i].ref_id + int(nams[i].score) + nams[i].query_start + nams[i].query_end;
        }
        global_nams_info[id] += local_nams_info;
        hits_per_ref0s[id].release();
        hits_per_ref1s[id].release();
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


    uint64_t num_bytes = 20ll * 1024ll * 1024ll * 1024ll;
    uint64_t seed = 13;
    init_mm(num_bytes, seed);

    printf("Gallatin global allocator initialized with %lu bytes.\n", num_bytes);

    my_vector<QueryRandstrobe> *global_randstrobes;
    cudaMallocManaged(&global_randstrobes, batch_size * 2 * sizeof(my_vector<QueryRandstrobe>));

    int *global_randstrobe_sizes;
    cudaMallocManaged(&global_randstrobe_sizes, batch_size * 2 * sizeof(int));
    uint64_t * global_hashes_value;
    cudaMallocManaged(&global_hashes_value, batch_size * 2 * sizeof(uint64_t));

    my_vector<my_pair<int, Hit>> *global_hits_per_ref0s;
    cudaMallocManaged(&global_hits_per_ref0s, batch_size * 2 * sizeof(my_vector<my_pair<int, Hit>>));

    my_vector<my_pair<int, Hit>> *global_hits_per_ref1s;
    cudaMallocManaged(&global_hits_per_ref1s, batch_size * 2 * sizeof(my_vector<my_pair<int, Hit>>));


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
    double gpu_cost3 = 0;
    double gpu_cost4 = 0;
    double gpu_cost5 = 0;
    double gpu_cost6 = 0;
    double gpu_cost7 = 0;
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


    Rescue_Seeds* global_rescue_seeds;
    cudaMallocManaged(&global_rescue_seeds, batch_size * 2 * sizeof(Rescue_Seeds));
    for(int i = 0; i < batch_size * 2; i++) {
        cudaMallocManaged(&(global_rescue_seeds[i].seeds), 250 * sizeof(QueryRandstrobe));
    }

    assert(records1.size() == records2.size());

//    print_mm();

    t0 = GetTime();
    for (int l_id = 0; l_id < records1.size(); l_id += batch_size) {
        printf("process %d / %d\n", l_id, records1.size());
        int r_id = l_id + batch_size;
        if (r_id > records1.size()) r_id = records1.size();
        int s_len = r_id - l_id;

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

        for (int i = 0; i < s_len * 2; i++) {
            // check infos
            global_randstrobe_sizes[i] = 0;
            global_hashes_value[i] = 0;
            global_hits_num[i] = 0;
            global_nams_info[i] = 0;

            global_rescue_seeds[i].read_id = -1;
            global_rescue_seeds[i].seeds_num = -1;

            global_hits_per_ref0s[i].data = nullptr;
            global_hits_per_ref0s[i].length = 0;
            global_hits_per_ref1s[i].data = nullptr;
            global_hits_per_ref1s[i].length = 0;

            global_randstrobes[i].data = nullptr;
            global_randstrobes[i].length = 0;
        }


        double t1 = GetTime();
        int threads_per_block;
        int reads_per_block;
        int blocks_per_grid;

        threads_per_block = 2;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_get_randstrobes_and_hits<<<blocks_per_grid, threads_per_block>>>(index.bits, index.filter_cutoff, para_rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                    d_index_para, s_len * 2, d_pre_sum, d_len, d_seq, d_pre_sum2, d_len2, d_seq2, global_hits_num, global_rescue_seeds,
                                                                    global_hits_per_ref0s, global_hits_per_ref1s, global_randstrobe_sizes, global_hashes_value);
        cudaDeviceSynchronize();
        gpu_cost1 += GetTime() - t1;

        //threads_per_block = 32;
        //reads_per_block = threads_per_block * GPU_thread_task_size;
        //blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        //gpu_get_randstrobes<<<blocks_per_grid, threads_per_block>>>(s_len * 2, d_pre_sum, d_len, d_seq, d_pre_sum2, d_len2, d_seq2, d_index_para,
        //                                                            global_randstrobe_sizes, global_hashes_value, global_randstrobes);
        //cudaDeviceSynchronize();
        //gpu_cost1 += GetTime() - t1;

        //t1 = GetTime();
        //threads_per_block = 1;
        //reads_per_block = threads_per_block * GPU_thread_task_size;
        //blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        //gpu_get_hits<<<blocks_per_grid, threads_per_block>>>(index.bits, index.filter_cutoff, para_rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
        //                                                     s_len * 2, d_index_para, global_hits_num, global_rescue_seeds, global_randstrobes,
        //                                                     global_hits_per_ref0s, global_hits_per_ref1s);
        //cudaDeviceSynchronize();
        //gpu_cost2 += GetTime() - t1;


        t1 = GetTime();
        threads_per_block = 2;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_sort_hits<<<blocks_per_grid, threads_per_block>>>(s_len * 2, global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost3 += GetTime() - t1;


        t1 = GetTime();
        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_merge_hits<<<blocks_per_grid, threads_per_block>>>(s_len * 2, d_index_para, global_nams_info, global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost4 += GetTime() - t1;


        for (size_t i = 0; i < s_len * 2; ++i) {
            size_tot += global_randstrobe_sizes[i];
            check_sum += global_hashes_value[i];
            global_hits_num12 += global_hits_num[i];
            global_nams_info12 += global_nams_info[i];
        }

		int rescue_num = 0;
        for(int i = 0; i < s_len * 2; i++) {
            if(global_rescue_seeds[i].read_id != -1) {
				global_rescue_seeds[rescue_num].read_id = global_rescue_seeds[i].read_id;
                global_rescue_seeds[rescue_num].read_fr = global_rescue_seeds[i].read_fr;
                global_rescue_seeds[rescue_num].seeds_num = global_rescue_seeds[i].seeds_num;
                for (int j = 0; j < global_rescue_seeds[i].seeds_num; j++) {
                    global_rescue_seeds[rescue_num].seeds[j] = global_rescue_seeds[i].seeds[j];
                }
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
                                                             rescue_num, d_index_para, global_hits_num, global_rescue_seeds,
                                                             global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost5 += GetTime() - t1;

        t1 = GetTime();
        threads_per_block = 32;
        reads_per_block = GPU_thread_task_size;
        blocks_per_grid = (rescue_num + reads_per_block - 1) / reads_per_block;
        gpu_rescue_sort_hits<<<rescue_num, threads_per_block>>>(rescue_num, d_index_para, global_rescue_seeds, global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost6 += GetTime() - t1;



        t1 = GetTime();
        threads_per_block = 1;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (rescue_num + reads_per_block - 1) / reads_per_block;
        gpu_rescue_merge_hits<<<blocks_per_grid, threads_per_block>>>(rescue_num, d_index_para, global_nams_info, global_rescue_seeds, global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost7 += GetTime() - t1;

        for (int i = 0; i < rescue_num; ++i) {
            global_hits_num3 += global_hits_num[i];
            global_nams_info3 += global_nams_info[i];
        }

//        print_mm();

    }
    tot_cost += GetTime() - t0;

    std::cout << "gpu cost " << gpu_cost1 << " " << gpu_cost2 << " " << gpu_cost3 << " " << gpu_cost4 << " " << gpu_cost5 << " " << gpu_cost6 << " " << gpu_cost7 << std::endl;
    std::cout << "total cost " << tot_cost << std::endl;
    std::cout << "check_sum : " << check_sum << ", size_tot : " << size_tot << std::endl;
    std::cout << "total_hits12 : " << global_hits_num12 << ", nr_good_hits12 : " << global_nams_info12 << std::endl;
    std::cout << "total_hits3 : " << global_hits_num3 << ", nr_good_hits3 : " << global_nams_info3 << std::endl;

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

