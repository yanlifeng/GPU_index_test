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

    uint64_t tot_len = 0;
    int* h_len = new int[records1.size()];
    int* pre_sum = new int[records1.size() + 1];
    pre_sum[0] = 0;
    for(int i = 0; i < records1.size(); i++) {
        tot_len += records1[i].seq.length();
        h_len[i] = records1[i].seq.length();
        pre_sum[i + 1] = pre_sum[i] + h_len[i];
    }

    std::cout << "tot_len : " << tot_len << std::endl;
    std::cout << "pre_sum : " << pre_sum[records1.size()] << std::endl;

    char* h_seq = new char[tot_len];
    memset(h_seq, 0, tot_len);

    t0 = GetTime();
#pragma omp parallel for
    for(int i = 0; i < records1.size(); i++) {
        memcpy(h_seq + pre_sum[i], records1[i].seq.c_str(), h_len[i]);
    }
    std::cout << "host memcpy cost " << GetTime() - t0 << std::endl;

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

    t0 = GetTime();
    char* d_seq;
    int* d_len;
    cudaMalloc(&d_seq, tot_len);
    cudaMemset(d_seq, 0, tot_len);
    cudaMalloc(&d_len, records1.size() * sizeof(int));
    cudaMemset(d_len, 0, records1.size() * sizeof(int));
    std::cout << "malloc2 execution time: " << GetTime() - t0 << " seconds, size " << tot_len + records1.size() * sizeof(int) << std::endl;

    t0 = GetTime();
    cudaMemcpy(d_seq, h_seq, tot_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, h_len, records1.size() * sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "memcpy2 execution time: " << GetTime() - t0 << " seconds, size " << tot_len + records1.size() * sizeof(int) << std::endl;


    t0 = GetTime();
//    int threads_per_block = 256;
//    int blocks_per_grid = (records1.size() + threads_per_block - 1) / threads_per_block;
//    gpu_step2<<<blocks_per_grid, threads_per_block>>>(d_randstrobes, d_randstrobe_start_indices, records1.size());
//    cudaDeviceSynchronize();
    std::cout << "GPU run cost " << GetTime() - t0 << std::endl;


    size_t check_sum = 0;
    //for (size_t i = 0; i < 10; ++i) {
    //    int id = rand() % num_queries;
    //    std::cout << "Query " << id << ": Position " << positions[id] << std::endl;
    //    check_sum += positions[id];
    //}
    std::cout << "check sum is " << check_sum << std::endl;


    t0 = GetTime();
    cudaFree(d_randstrobes);
    cudaFree(d_randstrobe_start_indices);
    std::cout << "free execution time: " << GetTime() - t0 << " seconds" << std::endl;

    delete h_seq;
    delete h_len;
    delete pre_sum;
    return 0;
}

