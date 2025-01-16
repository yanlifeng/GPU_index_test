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

)  {
    const auto key = d_randstrobes[position].hash;
    const unsigned int top_N = key >> (64 - bits);
    my_bucket_index_t position_end = d_randstrobe_start_indices[top_N + 1];
    int count = 1;
    for (my_bucket_index_t position_start = position + 1; position_start < position_end; ++position_start) {
        if (d_randstrobes[position_start].hash == key){
            count += 1;
        } else break;
    }
    return count;
}

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
__device__ void quick_sort_iterative(T* data, int low, int high, my_pool& mpool) {
//    int stack[32];
    my_vector<int>stack_vec(&mpool);
    int* stack = &(stack_vec[0]);
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

    stack_vec.release();
    //if(mx_top >= 32) printf("GG top size %d\n", mx_top);
}

template <typename T>
__device__ void quick_sort(T* data, int size, my_pool& mpool) {
    quick_sort_iterative(data, 0, size - 1, mpool);
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

//#define use_my_time

__device__ void merge_hits_into_nams(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int k,
        bool sort,
        bool is_revcomp,
        my_vector<Nam>& nams,
        my_pool& mpool,
        unsigned long long &t1,
        unsigned long long &t2,
        unsigned long long &t3,
        unsigned long long &t3_1,
        unsigned long long &t3_2,
        unsigned long long &t3_3,
        unsigned long long &t3_4,
        unsigned long long &t3_5
) {

    if(hits_per_ref.size() == 0) return;

//    printf("hits struct num %d\n", hits_per_ref.size());

//    for(int i = 0; i < hits_per_ref.size(); i++) {
//        Hit h = hits_per_ref[i].second;
//        printf("ref %d, hit %d %d %d %d\n", hits_per_ref[i].first, h.query_start, h.query_end, h.ref_start, h.ref_end);
//    }
//    printf("\n");


    unsigned long long t_start;
#ifdef use_my_time
    t_start = clock64();
#endif
    quick_sort(&(hits_per_ref[0]), hits_per_ref.size(), mpool);
    //bubble_sort(&(hits_per_ref[0]), hits_per_ref.size());
#ifdef use_my_time
    t1 += clock64() - t_start;
#endif

//    for(int i = 0; i < hits_per_ref.size(); i++) {
//        printf("ref %d, hit %d\n", hits_per_ref[i].first, hits_per_ref[i].second.query_start);
//    }
//    printf("\n");

#ifdef use_my_time
    t_start = clock64();
#endif
    int ref_num = 0;
    my_vector<int> each_ref_size(&mpool);
    int pre_ref_id = hits_per_ref[0].first;;
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
//    printf("ref_num is %d\n", ref_num);
#ifdef use_my_time
    t2 += clock64() - t_start;
#endif


    my_vector<Nam> open_nams(&mpool);

#ifdef use_my_time
    t_start = clock64();
#endif
    int now_vec_pos = 0;
    for (int i = 0; i < ref_num; i++) {

        unsigned long long t_start1;
#ifdef use_my_time
        t_start1 = clock64();
#endif
        if(i != 0) now_vec_pos += each_ref_size[i - 1];
        int ref_id = hits_per_ref[now_vec_pos].first;
        if (sort) {
//            std::sort(hits.begin(), hits.end());
//            quick_sort(&(hits_per_ref[now_vec_pos]), each_ref_size[i], mpool);
        }
        open_nams.clear();
        unsigned int prev_q_start = 0;
#ifdef use_my_time
        t3_1 += clock64() - t_start1;
#endif
 
        for (int j = 0; j < each_ref_size[i]; j++) {
#ifdef use_my_time
            t_start1 = clock64();
#endif
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
#ifdef use_my_time
            t3_2 += clock64() - t_start1;
#endif
 
            // Add the hit to open matches
            if (!is_added){
#ifdef use_my_time
                t_start1 = clock64();
#endif
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
#ifdef use_my_time
                t3_3 += clock64() - t_start1;
#endif
            }

            // Only filter if we have advanced at least k nucleotides
            if (h.query_start > prev_q_start + k) {

#ifdef use_my_time
                t_start1 = clock64();
#endif
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
#ifdef use_my_time
                t3_4 += clock64() - t_start1;
#endif
            }
        }

#ifdef use_my_time
        t_start1 = clock64();
#endif
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
#ifdef use_my_time
        t3_5 += clock64() - t_start1;
#endif
 
    }
#ifdef use_my_time
    t3 += clock64() - t_start;
#endif

    each_ref_size.release();
    open_nams.release();

}

__device__ void merge_hits_into_nams_forward_and_reverse(
        my_vector<Nam>& nams,
        my_vector<my_pair<int, Hit>> hits_per_ref[2],
        int k,
        bool sort,
        my_pool& mpool,
        unsigned long long &t1,
        unsigned long long &t2,
        unsigned long long &t3,
        unsigned long long &t3_1,
        unsigned long long &t3_2,
        unsigned long long &t3_3,
        unsigned long long &t3_4,
        unsigned long long &t3_5
) {
    for (size_t is_revcomp = 0; is_revcomp < 2; ++is_revcomp) {
        auto& hits_oriented = hits_per_ref[is_revcomp];
        merge_hits_into_nams(hits_oriented, k, sort, is_revcomp, nams, mpool, t1, t2, t3, t3_1, t3_2, t3_3, t3_4, t3_5);
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

#define GPU_read_block_size 1

__global__ void gpu_step2(
        int bits,
        unsigned int filter_cutoff,
        int rescue_cutoff,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        const my_bucket_index_t *d_randstrobe_start_indices,
        int num_reads,
        int *pre_sum,
        int *lens,
        char *all_seqs,
        int *pre_sum2,
        int *lens2,
        char *all_seqs2,
        IndexParameters *index_para,
        int *randstrobe_sizes,
        uint64_t *hashes,
        my_pool *mpools,
        uint64_t *global_total_hits,
        uint64_t *global_nr_good_hits
) {
    int b_tid = blockIdx.x * blockDim.x + threadIdx.x;

    //if(b_tid != 4000) return;

    int l_range = b_tid * GPU_read_block_size;
    int r_range = l_range + GPU_read_block_size;
    if (r_range > num_reads) r_range = num_reads;


    unsigned long long t_time1 = 0;
    unsigned long long t_time2 = 0;
    unsigned long long t_time3 = 0;
    unsigned long long t_time4 = 0;
    unsigned long long t_time4_1 = 0;
    unsigned long long t_time4_2 = 0;
    unsigned long long t_time4_3 = 0;
    unsigned long long t_time4_3_1 = 0;
    unsigned long long t_time4_3_2 = 0;
    unsigned long long t_time4_3_3 = 0;
    unsigned long long t_time4_3_4 = 0;
    unsigned long long t_time4_3_5 = 0;
    unsigned long long t_time5 = 0;
    unsigned long long t_time6 = 0;

    for (int tid = l_range; tid < r_range; tid++) {

        for (int rev = 0; rev < 2; rev++) {


#ifdef use_my_time
            unsigned long long t_start = clock64();
#endif
            my_pool mpool = mpools[tid];

            size_t len;
            char *seq;
            if (rev == 0) {
                len = lens[tid];
                seq = all_seqs + pre_sum[tid];
            } else {
                len = lens2[tid];
                seq = all_seqs2 + pre_sum2[tid];
            }

            // step1: get randstrobes
            unsigned long long t_start1;

            my_vector<QueryRandstrobe> randstrobes(&mpool);

            my_vector<Syncmer> syncmers(&mpool);

            my_vector<uint64_t> vec4syncmers(&mpool);

            if (len < (*index_para).randstrobe.w_max) {
                // randstrobes == null
            } else {

                SyncmerIterator syncmer_iterator{&vec4syncmers, seq, len, (*index_para).syncmer};
                Syncmer syncmer;
                while (1) {
                    syncmer = syncmer_iterator.gpu_next();
                    if (syncmer.is_end()) break;
                    syncmers.push_back(syncmer);
                }


                if (syncmers.size() == 0) {
                    // randstrobes == null
                } else {

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
            }
            syncmers.release();
            vec4syncmers.release();
#ifdef use_my_time
            t_time1 += clock64() - t_start;
#endif

#ifdef use_my_time
            t_start = clock64();
#endif
            randstrobe_sizes[tid] += randstrobes.size();
            for (int i = 0; i < randstrobes.size(); i++) hashes[tid] += randstrobes[i].hash;
//        if(randstrobes.size() > 0) hashes[tid] = randstrobes[randstrobes.size() / 2].hash;
#ifdef use_my_time
            t_time2 += clock64() - t_start;
#endif


            // step2: get nams

#ifdef use_my_time
            t_start = clock64();
#endif
            my_vector<my_pair<int, Hit>> hits_per_ref[2];
            hits_per_ref[0] = my_vector<my_pair<int, Hit>>(&mpool);
            hits_per_ref[1] = my_vector<my_pair<int, Hit>>(&mpool);

            uint64_t local_total_hits = 0;
            uint64_t local_nr_good_hits = 0;
            // total element in hits_per_ref is less than 500 (pair is 6K bytes), get a vector<pair<int, uint64_t>> to store them
            for (int i = 0; i < randstrobes.size(); i++) {
                QueryRandstrobe q = randstrobes[i];
                size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
//                printf("poss %llu\n", position);
                if (position != static_cast<size_t>(-1)) {
                    local_total_hits++;
                    bool res = gpu_is_filtered(d_randstrobes, d_randstrobes_size, position, filter_cutoff);
                    if (res) {
                        continue;
                    }
                    local_nr_good_hits++;
                    add_to_hits_per_ref(hits_per_ref[q.is_reverse], q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                }
            }
#ifdef use_my_time
            t_time3 += clock64() - t_start;
#endif


#ifdef use_my_time
            t_start = clock64();
#endif
            my_vector<Nam> nams(&mpool);

            float nonrepetitive_fraction = local_total_hits > 0 ? ((float) local_nr_good_hits) / ((float) local_total_hits) : 1.0;
            merge_hits_into_nams_forward_and_reverse(nams, hits_per_ref, index_para->syncmer.k, false, mpool, t_time4_1, t_time4_2, t_time4_3,
                         t_time4_3_1, t_time4_3_2, t_time4_3_3, t_time4_3_4, t_time4_3_5);
//            printf("float %f\n", nonrepetitive_fraction);

#ifdef use_my_time
            t_time4 += clock64() - t_start;
#endif


#ifdef use_my_time
            t_start = clock64();
#endif

#ifdef use_my_time
            t_time5 += clock64() - t_start;
#endif

            // step3: rescue nams
#ifdef use_my_time
            t_start = clock64();
#endif
            int call_re = 0;
            if (nams.size() == 0 || nonrepetitive_fraction < 0.7) {
                call_re = 1;
            //if (1) {
                //printf("start rescue, nonrepetitive_fraction %f %d\n", nonrepetitive_fraction, tid);
                hits_per_ref[0].clear();
                hits_per_ref[1].clear();
                my_vector<RescueHit> hits_t[2];
                hits_t[0] = my_vector<RescueHit>(&mpool);
                hits_t[1] = my_vector<RescueHit>(&mpool);

                for (int i = 0; i < randstrobes.size(); i++) {
                    QueryRandstrobe q = randstrobes[i];
                    size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
                    if (position != static_cast<size_t>(-1)) {
                        unsigned int count = gpu_get_count(d_randstrobes, d_randstrobe_start_indices, position, bits);
                        RescueHit rh{position, count, q.start, q.end};
                        hits_t[q.is_reverse].push_back(rh);
                    }
                }

                for (int is_revcomp = 0; is_revcomp < 2; is_revcomp++) {
                    quick_sort(&(hits_t[is_revcomp][0]), hits_t[is_revcomp].size(), mpool);
                    int cnt = 0;
                    for (int i = 0; i < hits_t[is_revcomp].size(); i++) {
                        RescueHit& rh = hits_t[is_revcomp][i];
                        if ((rh.count > rescue_cutoff && cnt >= 5) || rh.count > 1000) {
                            break;
                        }
                        add_to_hits_per_ref(hits_per_ref[is_revcomp], rh.query_start, rh.query_end, rh.position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                        cnt++;
                    }
                }
                nams.clear();
                merge_hits_into_nams_forward_and_reverse(nams, hits_per_ref, index_para->syncmer.k, true, mpool, t_time4_1, t_time4_2, t_time4_3,
                                                         t_time4_3_1, t_time4_3_2, t_time4_3_3, t_time4_3_4, t_time4_3_5);
            }
#ifdef use_my_time
            t_time6 += clock64() - t_start;
#endif
            //bubble_sort((Nam*)&(nams[0]), nams.size());
            //quick_sort((Nam*)&(nams[0]), nams.size(), mpool);


            //for(int i = 0; i < nams.size(); i++) {
            //    local_total_hits += nams[i].ref_id + nams[i].ref_start + nams[i].ref_end;
            //    local_nr_good_hits += nams[i].score + nams[i].query_start + nams[i].query_end;
            //}
            //global_total_hits[tid] = local_total_hits;
            //global_nr_good_hits[tid] = local_nr_good_hits;
            global_total_hits[tid]++;
            global_nr_good_hits[tid] += call_re;


            if(0) {
                printf("total_hits %llu, nr_good_hits %llu, nonrepetitive_fraction %.3f, nam size %d\n", local_total_hits, local_nr_good_hits, nonrepetitive_fraction, nams.size());
                for(int i = 0; i < nams.size(); i++) {
                    printf("query_start: %d, query_end: %d, query_prev_hit_startpos: %d, ref_start: %d, ref_end: %d, ref_prev_hit_startpos: %d, n_hits: %d, ref_id: %d, score: %.2f\n",
//                           nams[i].nam_id,
                           nams[i].query_start,
                           nams[i].query_end,
                           nams[i].query_prev_hit_startpos,
                           nams[i].ref_start,
                           nams[i].ref_end,
                           nams[i].ref_prev_hit_startpos,
                           nams[i].n_hits,
                           nams[i].ref_id,
                           nams[i].score);
                }
                printf("\n\n");
            }
        }
    }

#ifdef use_my_time
    if(b_tid % 100000 == 0) {
//        acquire_lock();
        printf("btid %d, time %.3f %.3f %.3f %.3f ( %.3f %.3f %.3f [ %.3f %.3f %.3f %.3f %.3f ] ) %.3f %.3f\n", b_tid, t_time1 / 1e6, t_time2 / 1e6, t_time3 / 1e6, t_time4 / 1e6, t_time4_1 / 1e6, t_time4_2 / 1e6, t_time4_3 / 1e6, t_time4_3_1 / 1e6, t_time4_3_2 / 1e6, t_time4_3_3 / 1e6, t_time4_3_4 / 1e6, t_time4_3_5 / 1e6, t_time5 / 1e6, t_time6 / 1e6);
//        release_lock();
    }
#endif

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

    int para_rescue_cutoff = opt.rescue_level < 100 ? opt.rescue_level * index.filter_cutoff : 1000;

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

#define batch_size 5000
#define batch_seq_szie batch_size * 250ll

    t0 = GetTime();
    char *d_mpools_data;
    cudaMalloc(&d_mpools_data, batch_size * vec_block_size);
    cudaMemset(d_mpools_data, 0, batch_size * vec_block_size);
    printf("device addr %p\n", d_mpools_data);
    my_pool *h_mpools = new my_pool[batch_size];
    for (int i = 0; i < batch_size; i++) {
        h_mpools[i].size = vec_block_size;
        h_mpools[i].data = d_mpools_data + i * vec_block_size;
    }
    my_pool *d_mpools;
    cudaMalloc(&d_mpools, batch_size * sizeof(my_pool));
    std::cout << "buffer malloc execution time: " << GetTime() - t0 << std::endl;

    int *a_randstrobe_sizes;
    cudaMallocManaged(&a_randstrobe_sizes, batch_size * sizeof(int));
    uint64_t * a_hashes;
    cudaMallocManaged(&a_hashes, batch_size * sizeof(uint64_t));

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

    double gpu_cost = 0;
    double tot_cost = 0;
    size_t check_sum = 0;
    size_t size_tot = 0;

    uint64_t h_global_total_hits = 0;
    uint64_t * a_global_total_hits;
    cudaMallocManaged(&a_global_total_hits, batch_size * sizeof(uint64_t));

    uint64_t h_global_nr_good_hits = 0;
    uint64_t * a_global_nr_good_hits;
    cudaMallocManaged(&a_global_nr_good_hits, batch_size * sizeof(uint64_t));

    assert(records1.size() == records2.size());

    t0 = GetTime();
    for (int l_id = 0; l_id < records1.size(); l_id += batch_size) {
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
//        memset(h_seq, 0, tot_len);
//        memset(h_seq2, 0, tot_len2);
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
            for(int j = 0; j < (1 << vec_block_num_shift); j++)
                h_mpools[i].used[j] = 0;
            h_mpools[i].pos = 0;
            a_randstrobe_sizes[i] = 0;
            a_hashes[i] = 0;
            a_global_total_hits[i] = 0;
            a_global_nr_good_hits[i] = 0;
        }
        cudaMemcpy(d_mpools, h_mpools, s_len * sizeof(my_pool), cudaMemcpyHostToDevice);

        double t1 = GetTime();
        int threads_per_block = 1;
        int reads_per_block = threads_per_block * GPU_read_block_size;
        int blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_step2<<<blocks_per_grid, threads_per_block>>>(index.bits, index.filter_cutoff, para_rescue_cutoff, d_randstrobes,
                                                          index.randstrobes.size(), d_randstrobe_start_indices,
                                                          s_len, d_pre_sum, d_len, d_seq, d_pre_sum2, d_len2, d_seq2,
                                                          d_index_para,
                                                          a_randstrobe_sizes, a_hashes, d_mpools, a_global_total_hits,
                                                          a_global_nr_good_hits);
        cudaDeviceSynchronize();
        gpu_cost += GetTime() - t1;

        for (size_t i = 0; i < s_len; ++i) {
            size_tot += a_randstrobe_sizes[i];
            h_global_total_hits += a_global_total_hits[i];
            h_global_nr_good_hits += a_global_nr_good_hits[i];
        }
        for (size_t i = 0; i < 1; ++i) {
            int id = rand() % s_len;
            //int id = i;
            check_sum += a_randstrobe_sizes[id];
        }
    }
    tot_cost += GetTime() - t0;

    std::cout << "gpu cost " << gpu_cost << std::endl;
    std::cout << "total cost " << tot_cost << std::endl;
    std::cout << "check_sum : " << check_sum << ", size_tot : " << size_tot << std::endl;
    std::cout << "total_hits : " << h_global_total_hits << ", nr_good_hits : " << h_global_nr_good_hits << std::endl;

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

    return 0;
}

