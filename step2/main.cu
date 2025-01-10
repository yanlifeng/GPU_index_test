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

#include <thrust/device_vector.h>

#define my_bucket_index_t StrobemerIndex::bucket_index_t

inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}

InputBuffer get_input_buffer(const CommandLineOptions& opt) {
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

int producer_pe_fastq_task(std::string file, std::string file2, rabbit::fq::FastqDataPool &fastqPool, rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> &dq) {
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




__global__ void gpu_step2(
        const RefRandstrobe* d_randstrobes,
        const my_bucket_index_t* d_randstrobe_start_indices,
        int num_reads,
        int* pre_sum,
        int* lens,
        char* all_seqs,
        IndexParameters* index_para,
        int* randstrobe_sizes, 
        uint64_t *hashes,
        my_pool* mpools
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //if(tid == 0) {
    //    printf("num_reads %d\n", num_reads);
    //}
    if(tid < num_reads) {
        my_pool mpool = mpools[tid];

        unsigned long long t_time;
        unsigned long long t_timed[100];
        int t_pos = 0;

        unsigned long long t_start = clock64();

        size_t len = lens[tid];
        char* seq = all_seqs + pre_sum[tid];

        //if(tid == 0) {
        //    printf("len %d\n", len);
        //    //for(int i = 0; i < len; i++) printf("%c", seq[i]);
        //    //printf("\n");
        //} //else return;

        // step1: get randstrobes
        unsigned long long t_start1;

        t_start1 = clock64();
        my_vector<QueryRandstrobe> randstrobes(&mpool);
        t_timed[t_pos++] = clock64() - t_start1;

        t_start1 = clock64();
        my_vector<Syncmer> syncmers(&mpool);
        t_timed[t_pos++] = clock64() - t_start1;

        t_start1 = clock64();
        my_vector<uint64_t> vec4syncmers(&mpool);
        t_timed[t_pos++] = clock64() - t_start1;
        if (len < (*index_para).randstrobe.w_max) {
            // randstrobes == null
        } else {
            t_start1 = clock64();
            SyncmerIterator syncmer_iterator{&vec4syncmers, seq, len, (*index_para).syncmer};
            Syncmer syncmer;
            while (1) {
                syncmer = syncmer_iterator.gpu_next();
                if(syncmer.is_end()) break;
                syncmers.push_back(syncmer);
                //printf("syncmer %llu %lld\n", syncmer.hash, syncmer.position);
            }
            t_timed[t_pos++] = clock64() - t_start1;

            if (syncmers.size() == 0) {
                // randstrobes == null
            } else {
                t_start1 = clock64();
                RandstrobeIterator randstrobe_fwd_iter{&syncmers, (*index_para).randstrobe};
                while (randstrobe_fwd_iter.gpu_has_next()) {
                    Randstrobe randstrobe = randstrobe_fwd_iter.gpu_next();
                    //printf("value %llu %u %u\n", randstrobe.hash, randstrobe.strobe1_pos, randstrobe.strobe2_pos + (*index_para).syncmer.k);
                    randstrobes.push_back(
                            QueryRandstrobe{
                                    randstrobe.hash, randstrobe.strobe1_pos, randstrobe.strobe2_pos + (*index_para).syncmer.k, false
                            }
                    );
                }
                t_timed[t_pos++] = clock64() - t_start1;

                t_start1 = clock64();
                for(int i = 0; i < syncmers.size() / 2; i++) {
                    my_swap(syncmers[i], syncmers[syncmers.size() - i - 1]);
                }
                for (size_t i = 0; i < syncmers.size(); i++) {
                    syncmers[i].position = len - syncmers[i].position - (*index_para).syncmer.k;
                }
                t_timed[t_pos++] = clock64() - t_start1;

                t_start1 = clock64();
                RandstrobeIterator randstrobe_rc_iter{&syncmers, (*index_para).randstrobe};
                while (randstrobe_rc_iter.gpu_has_next()) {
                    Randstrobe randstrobe = randstrobe_rc_iter.gpu_next();
                    //printf("value %llu %u %u\n", randstrobe.hash, randstrobe.strobe1_pos, randstrobe.strobe2_pos + (*index_para).syncmer.k);
                    randstrobes.push_back(
                            QueryRandstrobe{
                                    randstrobe.hash, randstrobe.strobe1_pos, randstrobe.strobe2_pos + (*index_para).syncmer.k, true
                            }
                    );
                }
                t_timed[t_pos++] = clock64() - t_start1;
            }
        }

//        t_start1 = clock64();
//        randstrobes.~my_vector<QueryRandstrobe>();
//        t_timed[t_pos++] = clock64() - t_start1;
//
//        t_start1 = clock64();
//        syncmers.~my_vector<Syncmer>();
//        t_timed[t_pos++] = clock64() - t_start1;
//
//        t_start1 = clock64();
//        vec4syncmers.~my_vector<uint64_t>();
//        t_timed[t_pos++] = clock64() - t_start1;

        t_time = clock64() - t_start;
        //if(tid % 5000 == 0) {
        //    printf("Task %u took %llu : ( ", tid, blockIdx.x % gridDim.x, t_time);
        //    for(int i = 0; i < t_pos; i++) printf("%llu + ", t_timed[i]);
        //    printf(")\n");
        //}

//        if(tid % 10000 == 0) printf("tid %d get %d randstrobes\n", tid, randstrobes.size());
        randstrobe_sizes[tid] = randstrobes.size();
        if(randstrobes.size() > 0) hashes[tid] = randstrobes[0].hash;
        //printf("randstrobes size %d\n", randstrobes.size());

        // step2: get nams




    }
}


__global__ void gpu_find(
    const RefRandstrobe* d_randstrobes,
    const my_bucket_index_t* d_randstrobe_start_indices,
    const randstrobe_hash_t* d_queries,
    size_t* d_positions,
    int num_queries,
    int bits
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_queries) {
        randstrobe_hash_t key = d_queries[tid];
        const unsigned int top_N = key >> (64 - bits);
        my_bucket_index_t position_start = d_randstrobe_start_indices[top_N];
        my_bucket_index_t position_end = d_randstrobe_start_indices[top_N + 1];

        //d_positions[tid] = position_end - position_start;

        if (position_start == position_end) {
            d_positions[tid] = static_cast<size_t>(-1); // No match
            return;
        }

        for (my_bucket_index_t i = position_start; i < position_end; ++i) {
            if (d_randstrobes[i].hash == key) {
                d_positions[tid] = i;
                return;
            }
        }

        d_positions[tid] = static_cast<size_t>(-1); // No match
    }
}




std::vector<uint64_t> readFileToVector(const std::string& filename) {
    std::vector<uint64_t> data;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        uint64_t value;
        while (iss >> value) {
            data.push_back(value);
        }
    }

    infile.close();
    return data;
}

klibpp::KSeq ConvertNeo2KSeq(neoReference ref) {
    klibpp::KSeq res;
    res.name = std::string((char *) ref.base + ref.pname, ref.lname);
    if(!res.name.empty()) {
        size_t space_pos = res.name.find(' ');
        int l_pos = 0;
        if(res.name[0] == '@') l_pos = 1;
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

static Logger& logger = Logger::get();


int main(int argc, char** argv) {
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

    std::cout << "read file : " << opt.reads_filename1 << " " << opt.reads_filename2 << std::endl;

    rabbit::fq::FastqDataPool fastqPool(1024, 1 << 22);
    rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> queue_pe(1024, 1);
    std::thread *producer;
    producer = new std::thread(producer_pe_fastq_task, opt.reads_filename1, opt.reads_filename2, std::ref(fastqPool), std::ref(queue_pe));

    std::vector<neoReference> data1;
    std::vector<neoReference> data2;
    rabbit::fq::FastqDataPairChunk *fqdatachunk = new rabbit::fq::FastqDataPairChunk;
    std::vector<klibpp::KSeq> records1;
    std::vector<klibpp::KSeq> records2;
    long long id;
    while(queue_pe.Pop(id, fqdatachunk)) {
        data1.clear();
        data2.clear();
        rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk*)(fqdatachunk->left_part), data1);
        rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk*)(fqdatachunk->right_part), data2);
        assert(data1.size() == data2.size());
        for(int i = 0; i < data1.size(); i++) {
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
    RefRandstrobe* d_randstrobes;
    my_bucket_index_t* d_randstrobe_start_indices;
    std::cout << index.randstrobes.size() * sizeof(RefRandstrobe) << std::endl;
    cudaMalloc(&d_randstrobes, index.randstrobes.size() * sizeof(RefRandstrobe));
    cudaMalloc(&d_randstrobe_start_indices, index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t));
    cudaMemset(d_randstrobes, 0, index.randstrobes.size() * sizeof(RefRandstrobe));
    cudaMemset(d_randstrobe_start_indices, 0, index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t));
    std::cout << "malloc1 execution time: " << GetTime() - t0 << " seconds, size " << index.randstrobes.size() * sizeof(RefRandstrobe) + index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t) << std::endl;

    t0 = GetTime();
    cudaMemcpy(d_randstrobes, index.randstrobes.data(), index.randstrobes.size() * sizeof(RefRandstrobe), cudaMemcpyHostToDevice);
    cudaMemcpy(d_randstrobe_start_indices, index.randstrobe_start_indices.data(), index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t), cudaMemcpyHostToDevice);
    std::cout << "memcpy1 execution time: " << GetTime() - t0 << " seconds, size " << index.randstrobes.size() * sizeof(RefRandstrobe) + index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t) << std::endl;

#define batch_size 25000
#define batch_seq_szie batch_size * 250ll

    t0 = GetTime();
    char* d_mpools_data;
    cudaMalloc(&d_mpools_data, batch_size * vec_block_size);
    cudaMemset(d_mpools_data, 0, batch_size * vec_block_size);
    printf("device addr %p\n", d_mpools_data);
    my_pool* h_mpools = new my_pool[batch_size];
    for(int i = 0; i < batch_size; i++) {
        h_mpools[i].size = vec_block_size;
        h_mpools[i].data = d_mpools_data + i * vec_block_size;
        if(i < 10) printf("%p\n", h_mpools[i].data);
    }
    my_pool* d_mpools;
    cudaMalloc(&d_mpools, batch_size * sizeof(my_pool));
    std::cout << "buffer malloc execution time: " << GetTime() - t0 << std::endl;

    int* a_randstrobe_sizes;
    cudaMallocManaged(&a_randstrobe_sizes, batch_size * sizeof(int));
    uint64_t* a_hashes;
    cudaMallocManaged(&a_hashes, batch_size * sizeof(uint64_t));

    t0 = GetTime();
    char* d_seq;
    int* d_len;
    int* d_pre_sum;
    cudaMalloc(&d_seq, batch_seq_szie);
    cudaMemset(d_seq, 0, batch_seq_szie);
    cudaMalloc(&d_len, batch_size * sizeof(int));
    cudaMemset(d_len, 0, batch_size * sizeof(int));
    cudaMalloc(&d_pre_sum, batch_size * sizeof(int));
    cudaMemset(d_pre_sum, 0, batch_size * sizeof(int));
    std::cout << "malloc2 execution time: " << GetTime() - t0 << " seconds, size " << batch_seq_szie + batch_size * sizeof(int) << std::endl;

    IndexParameters* d_index_para;
    cudaMalloc(&d_index_para, sizeof(IndexParameters));
     

    int* h_len = new int[batch_size];
    int* h_pre_sum = new int[batch_size + 1];
    char* h_seq = new char[batch_seq_szie];

    double gpu_cost = 0;
    size_t check_sum = 0;
    size_t size_tot = 0;
        
    for(int l_id = 0; l_id < records1.size(); l_id += batch_size) {

        int r_id = l_id + batch_size;
        if(r_id > records1.size()) r_id = records1.size();
        int s_len = r_id - l_id;
        //printf("[%d %d] -- %d\n", l_id, r_id, s_len);


        uint64_t tot_len = 0;
        h_pre_sum[0] = 0;
        for(int i = l_id; i < r_id; i++) {
            tot_len += records1[i].seq.length();
            h_len[i - l_id] = records1[i].seq.length();
            h_pre_sum[i + 1 - l_id] = h_pre_sum[i - l_id] + h_len[i - l_id];
        }

        //std::cout << "tot_len : " << tot_len << std::endl;
        //std::cout << "pre_sum : " << h_pre_sum[s_len] << std::endl;

        memset(h_seq, 0, tot_len);

        t0 = GetTime();
#pragma omp parallel for
        for(int i = l_id; i < r_id; i++) {
            memcpy(h_seq + h_pre_sum[i - l_id], records1[i].seq.c_str(), h_len[i - l_id]);
        }
        //std::cout << "host memcpy cost " << GetTime() - t0 << std::endl;

        t0 = GetTime();
        cudaMemcpy(d_seq, h_seq, tot_len, cudaMemcpyHostToDevice);
        cudaMemcpy(d_len, h_len, s_len * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pre_sum, h_pre_sum, s_len * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_index_para, &index_parameters, sizeof(IndexParameters), cudaMemcpyHostToDevice);
        //std::cout << "memcpy2 execution time: " << GetTime() - t0 << " seconds, size " << tot_len + s_len * sizeof(int) << std::endl;

        t0 = GetTime();
        for(int i = 0; i < s_len; i++) h_mpools[i].pos = 0;
        cudaMemcpy(d_mpools, h_mpools, s_len * sizeof(my_pool), cudaMemcpyHostToDevice);
        //std::cout << "buffer memcpy execution time: " << GetTime() - t0 << std::endl;

        t0 = GetTime();
        //std::cout << "read num " << s_len << std::endl;
        int threads_per_block = 32;
        int blocks_per_grid = (s_len + threads_per_block - 1) / threads_per_block;
        //std::cout << "blocks_per_grid " << blocks_per_grid << std::endl;
        gpu_step2<<<blocks_per_grid, threads_per_block>>>(d_randstrobes, d_randstrobe_start_indices, s_len, d_pre_sum, d_len, d_seq, d_index_para, a_randstrobe_sizes, a_hashes, d_mpools);
        cudaDeviceSynchronize();
        //std::cout << "GPU run cost " << GetTime() - t0 << std::endl;
        gpu_cost += GetTime() - t0;


        for (size_t i = 0; i < s_len; ++i) {
            size_tot += a_randstrobe_sizes[i];
        }
        //std::cout << "size tot " << size_tot << ", avg : " << 1.0 * size_tot / s_len << std::endl;
        for (size_t i = 0; i < 1; ++i) {
            int id = rand() % s_len;
            //int id = i;
            //std::cout << "Query " << id + l_id << ": Position " << a_randstrobe_sizes[id] << " " << a_hashes[id] <<  std::endl;
            check_sum += a_randstrobe_sizes[id];
        }
        //std::cout << "check sum is " << check_sum << std::endl;
    }


    for(int l_id = 0; l_id < records2.size(); l_id += batch_size) {

        int r_id = l_id + batch_size;
        if(r_id > records2.size()) r_id = records2.size();
        int s_len = r_id - l_id;
        //printf("[%d %d] -- %d\n", l_id, r_id, s_len);


        uint64_t tot_len = 0;
        h_pre_sum[0] = 0;
        for(int i = l_id; i < r_id; i++) {
            tot_len += records2[i].seq.length();
            h_len[i - l_id] = records2[i].seq.length();
            h_pre_sum[i + 1 - l_id] = h_pre_sum[i - l_id] + h_len[i - l_id];
        }

        //std::cout << "tot_len : " << tot_len << std::endl;
        //std::cout << "pre_sum : " << h_pre_sum[s_len] << std::endl;

        memset(h_seq, 0, tot_len);

        t0 = GetTime();
#pragma omp parallel for
        for(int i = l_id; i < r_id; i++) {
            memcpy(h_seq + h_pre_sum[i - l_id], records2[i].seq.c_str(), h_len[i - l_id]);
        }
        //std::cout << "host memcpy cost " << GetTime() - t0 << std::endl;

        t0 = GetTime();
        cudaMemcpy(d_seq, h_seq, tot_len, cudaMemcpyHostToDevice);
        cudaMemcpy(d_len, h_len, s_len * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pre_sum, h_pre_sum, s_len * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_index_para, &index_parameters, sizeof(IndexParameters), cudaMemcpyHostToDevice);
        //std::cout << "memcpy2 execution time: " << GetTime() - t0 << " seconds, size " << tot_len + s_len * sizeof(int) << std::endl;

        t0 = GetTime();
        for(int i = 0; i < s_len; i++) h_mpools[i].pos = 0;
        cudaMemcpy(d_mpools, h_mpools, s_len * sizeof(my_pool), cudaMemcpyHostToDevice);
        //std::cout << "buffer memcpy execution time: " << GetTime() - t0 << std::endl;

        t0 = GetTime();
        //std::cout << "read num " << s_len << std::endl;
        int threads_per_block = 32;
        int blocks_per_grid = (s_len + threads_per_block - 1) / threads_per_block;
        //std::cout << "blocks_per_grid " << blocks_per_grid << std::endl;
        gpu_step2<<<blocks_per_grid, threads_per_block>>>(d_randstrobes, d_randstrobe_start_indices, s_len, d_pre_sum, d_len, d_seq, d_index_para, a_randstrobe_sizes, a_hashes, d_mpools);
        cudaDeviceSynchronize();
        //std::cout << "GPU run cost " << GetTime() - t0 << std::endl;
        gpu_cost += GetTime() - t0;


        for (size_t i = 0; i < s_len; ++i) {
            size_tot += a_randstrobe_sizes[i];
        }
        //std::cout << "size tot " << size_tot << ", avg : " << 1.0 * size_tot / s_len << std::endl;
        for (size_t i = 0; i < 1; ++i) {
            int id = rand() % s_len;
            //int id = i;
            //std::cout << "Query " << id + l_id << ": Position " << a_randstrobe_sizes[id] << " " << a_hashes[id] <<  std::endl;
            check_sum += a_randstrobe_sizes[id];
        }
        //std::cout << "check sum is " << check_sum << std::endl;
    }

    std::cout << "gpu cost " << gpu_cost << std::endl;
    std::cout << check_sum << " " << size_tot << std::endl;

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

