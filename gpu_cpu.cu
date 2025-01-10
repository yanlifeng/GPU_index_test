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

inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}


using randstrobe_hash_t = uint64_t;
using bucket_index_t = size_t;

struct RefRandstrobe {
    randstrobe_hash_t hash;
    uint32_t offset1;
    uint32_t offset2;
};

int32_t read_int_from_istream(std::istream& is) {
    int32_t val;
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;

}

// Custom exception class for invalid index files
class InvalidIndexFile : public std::runtime_error {
    public:
        explicit InvalidIndexFile(const std::string& msg) : std::runtime_error(msg) {}
};

struct SyncmerParameters {
    const int k;
    const int s;
    const int t_syncmer;

    SyncmerParameters(int k, int s)
        : k(k)
          , s(s)
          , t_syncmer((k - s) / 2 + 1)
    {
        verify();
    }

    void verify() const {
        if (k <= 7 || k > 32) {
            //throw BadParameter("k not in [8,32]");
        }
        if (s > k) {
            //throw BadParameter("s is larger than k");
        }
        if ((k - s) % 2 != 0) {
            //throw BadParameter("(k - s) must be an even number to create canonical syncmers. Please set s to e.g. k-2, k-4, k-6, ...");
        }
    }

    bool operator==(const SyncmerParameters& other) const;
};

struct RandstrobeParameters {
    const int l;
    const int u;
    const uint64_t q;
    const int max_dist;
    const unsigned w_min;
    const unsigned w_max;

    RandstrobeParameters(int l, int u, uint64_t q, int max_dist, unsigned w_min, unsigned w_max)
        : l(l)
          , u(u)
          , q(q)
          , max_dist(max_dist)
          , w_min(w_min)
          , w_max(w_max)
    {
        verify();
    }

    bool operator==(const RandstrobeParameters& other) const;

    private:
    void verify() const {
        if (max_dist > 255) {
            //throw BadParameter("maximum seed length (-m <max_dist>) is larger than 255");
        }
        if (w_min > w_max) {
            //throw BadParameter("w_min is greater than w_max (choose different -l/-u parameters)");
        }
    }
};


/* Settings that influence index creation */
class IndexParameters {
    public:
        const size_t canonical_read_length;
        const SyncmerParameters syncmer;
        const RandstrobeParameters randstrobe;

        static const int DEFAULT = std::numeric_limits<int>::min();

        IndexParameters(size_t canonical_read_length, int k, int s, int l, int u, int q, int max_dist)
            : canonical_read_length(canonical_read_length)
              , syncmer(k, s)
              , randstrobe(l, u, q, max_dist, std::max(0, k / (k - s + 1) + l), k / (k - s + 1) + u)
    {
    }

        static IndexParameters read(std::istream& is) {
            size_t canonical_read_length = read_int_from_istream(is);
            int k = read_int_from_istream(is);
            int s = read_int_from_istream(is);
            int l = read_int_from_istream(is);
            int u = read_int_from_istream(is);
            int q = read_int_from_istream(is);
            int max_dist = read_int_from_istream(is);
            return IndexParameters(canonical_read_length, k, s, l, u, q, max_dist);
        }


        std::string filename_extension() const;
        void write(std::ostream& os) const;
        bool operator==(const IndexParameters& other) const;
        bool operator!=(const IndexParameters& other) const { return !(*this == other); }
};


// Helper function template: Read a vector from the input stream
template <typename T>
void read_vector(std::istream& is, std::vector<T>& v) {
    uint64_t size;
    v.clear();
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    v.resize(size);
    is.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
}

__global__ void gpu_find_single_thread(
        const RefRandstrobe* d_randstrobes,
        const bucket_index_t* d_randstrobe_start_indices,
        const randstrobe_hash_t* d_queries,
        size_t* d_positions,
        int num_queries,
        int bits
        ) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { 
        for (int tid = 0; tid < num_queries; ++tid) {
            randstrobe_hash_t key = d_queries[tid];
            const unsigned int top_N = key >> (64 - bits);
            bucket_index_t position_start = d_randstrobe_start_indices[top_N];
            bucket_index_t position_end = d_randstrobe_start_indices[top_N + 1];

            if (position_start == position_end) {
                d_positions[tid] = static_cast<size_t>(-1); // No match
                continue;
            }

            bool found = false;
            for (bucket_index_t i = position_start; i < position_end; ++i) {
                if (d_randstrobes[i].hash == key) {
                    d_positions[tid] = i;
                    found = true;
                    break;
                }
            }

            if (!found) {
                d_positions[tid] = static_cast<size_t>(-1); // No match
            }
        }
    }
}



// GPU kernel for find
__global__ void gpu_find_bs(
        const RefRandstrobe* d_randstrobes,
        const bucket_index_t* d_randstrobe_start_indices,
        const randstrobe_hash_t* d_queries,
        size_t* d_positions,
        int num_queries,
        int bits
        ) {
    int toffset = blockIdx.x * blockDim.x + threadIdx.x;
    int sz = gridDim.x * blockDim.x;
    for(int id = 0, tid = toffset; id < 32; id++, tid += sz) {
        if (tid < num_queries) {
            randstrobe_hash_t key = d_queries[tid];
            const unsigned int top_N = key >> (64 - bits);
            bucket_index_t position_start = d_randstrobe_start_indices[top_N];
            bucket_index_t position_end = d_randstrobe_start_indices[top_N + 1];

            //d_positions[tid] = position_end - position_start;

            if (position_start == position_end) {
                d_positions[tid] = static_cast<size_t>(-1); // No match
                continue;
                //return;
            }

            bool has_find = 0;
            for (bucket_index_t i = position_start; i < position_end; ++i) {
                if (d_randstrobes[i].hash == key) {
                    d_positions[tid] = i;
                    has_find = 1;
                    break;
                    //return;
                }
            }

            if(has_find == 0) d_positions[tid] = static_cast<size_t>(-1); // No match
        }
    }
}


__global__ void gpu_find(
        const RefRandstrobe* d_randstrobes,
        const bucket_index_t* d_randstrobe_start_indices,
        const randstrobe_hash_t* d_queries,
        size_t* d_positions,
        int num_queries,
        int bits
        ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_queries) {
        randstrobe_hash_t key = d_queries[tid];
        const unsigned int top_N = key >> (64 - bits);
        bucket_index_t position_start = d_randstrobe_start_indices[top_N];
        bucket_index_t position_end = d_randstrobe_start_indices[top_N + 1];

        //d_positions[tid] = position_end - position_start;

        if (position_start == position_end) {
            d_positions[tid] = static_cast<size_t>(-1); // No match
            return;
        }

        for (bucket_index_t i = position_start; i < position_end; ++i) {
            if (d_randstrobes[i].hash == key) {
                d_positions[tid] = i;
                return;
            }
        }

        d_positions[tid] = static_cast<size_t>(-1); // No match
    }
}

// Class representing the StrobemerIndex
class StrobemerIndex {
    public:
        std::vector<RefRandstrobe> randstrobes;  // Stores the randstrobes
        std::vector<bucket_index_t> randstrobe_start_indices;  // Start indices for hash buckets
        int bits;  // Number of bits for the hash table
        int filter_cutoff;  // Filter cutoff value

        // Read the index from a file
        void read(const std::string& filename) {
            errno = 0;
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs.is_open()) {
                throw InvalidIndexFile(filename + ": " + strerror(errno));
            }

            // Check magic number to verify file format
            union {
                char s[4];
                uint32_t v;
            } magic;
            ifs.read(magic.s, 4);
            if (magic.v != 0x01495453) {
                throw InvalidIndexFile("Index file has incorrect format (magic number mismatch)");
            }

            // Read file format version
            uint32_t file_format_version = read_int_from_istream(ifs);
            if (file_format_version != 2) {
                throw InvalidIndexFile("Unsupported index file format version");
            }

            // Skip over reserved chunk
            randstrobe_hash_t reserved_chunk_size;
            ifs.read(reinterpret_cast<char*>(&reserved_chunk_size), sizeof(reserved_chunk_size));
            ifs.seekg(reserved_chunk_size, std::ios_base::cur);

            // Read other parameters
            filter_cutoff = read_int_from_istream(ifs);
            bits = read_int_from_istream(ifs);
            const IndexParameters sti_parameters = IndexParameters::read(ifs);

            // Read randstrobes and start indices
            read_vector(ifs, randstrobes);
            read_vector(ifs, randstrobe_start_indices);
            if (randstrobe_start_indices.size() != (1u << bits) + 1) {
                throw InvalidIndexFile("randstrobe_start_indices vector is of the wrong size");
            }
        }

        // Find a key in the index
        size_t find(randstrobe_hash_t key) const {
            //constexpr int MAX_LINEAR_SEARCH = 4;
            const unsigned int top_N = key >> (64 - bits);
            bucket_index_t position_start = randstrobe_start_indices[top_N];
            bucket_index_t position_end = randstrobe_start_indices[top_N + 1];
            //return position_end - position_start;
            //std::cout << "top_N: " << top_N << ", range: " << position_start << " " << position_end << std::endl;

            if (position_start == position_end) {
                return static_cast<size_t>(-1); // No match
            }

            //if (position_end - position_start < MAX_LINEAR_SEARCH) {
            for (; position_start < position_end; ++position_start) {
                if (randstrobes[position_start].hash == key) return position_start;
                if (randstrobes[position_start].hash > key) break;
            }
            return static_cast<size_t>(-1); // No match
            //}

            //auto cmp = [](const RefRandstrobe& lhs, const randstrobe_hash_t rhs) { return lhs.hash < rhs; };
            //auto pos = std::lower_bound(randstrobes.begin() + position_start,
            //                            randstrobes.begin() + position_end,
            //                            key, cmp);

            //if (pos != randstrobes.end() && pos->hash == key) {
            //    return std::distance(randstrobes.begin(), pos);
            //}
            //return static_cast<size_t>(-1); // No match
        }


};


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


//#define use_managed

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <index_file_path>" << std::endl;
        return 1;
    }

    std::string index_file_path = argv[1];
    StrobemerIndex index;
    try {
        index.read(index_file_path);
    } catch (const std::exception& e) {
        std::cerr << "Error reading index file: " << e.what() << std::endl;
        return 1;
    }

	std::vector<randstrobe_hash_t> queries = readFileToVector("seed_info.txt");
    int num_queries = queries.size();
    printf("size %d, data[0] %lu\n", num_queries, queries[0]);
    double t0;

    size_t* d_positions;
    randstrobe_hash_t* d_queries;
    RefRandstrobe* d_randstrobes;
    bucket_index_t* d_randstrobe_start_indices;

    size_t* h_positions;
    size_t* positions = (size_t *)malloc(num_queries * sizeof(size_t));
    memset(positions, 0, num_queries * sizeof(size_t));
    randstrobe_hash_t* h_queries;

    cudaHostAlloc(&h_positions, num_queries * sizeof(size_t), cudaHostAllocDefault);
    cudaHostAlloc(&h_queries, num_queries * sizeof(randstrobe_hash_t), cudaHostAllocDefault);
    memcpy(h_queries, &(queries[0]), num_queries * sizeof(randstrobe_hash_t));

    t0 = GetTime();
    cudaMalloc(&d_positions, num_queries * sizeof(size_t));
    cudaMalloc(&d_queries, num_queries * sizeof(randstrobe_hash_t));
    cudaMalloc(&d_randstrobes, index.randstrobes.size() * sizeof(RefRandstrobe));
    cudaMalloc(&d_randstrobe_start_indices, index.randstrobe_start_indices.size() * sizeof(bucket_index_t));
    cudaMemset(d_positions, 0, num_queries * sizeof(size_t));
    cudaMemset(d_queries, 0, num_queries * sizeof(randstrobe_hash_t));
    cudaMemset(d_randstrobes, 0, index.randstrobes.size() * sizeof(RefRandstrobe));
    cudaMemset(d_randstrobe_start_indices, 0, index.randstrobe_start_indices.size() * sizeof(bucket_index_t));
    std::cout << "malloc execution time: " << GetTime() - t0 << " seconds, size " << num_queries * sizeof(size_t) + num_queries * sizeof(randstrobe_hash_t) + index.randstrobes.size() * sizeof(RefRandstrobe) + index.randstrobe_start_indices.size() * sizeof(bucket_index_t) << std::endl;

    t0 = GetTime();
    cudaMemcpy(d_randstrobes, index.randstrobes.data(), index.randstrobes.size() * sizeof(RefRandstrobe), cudaMemcpyHostToDevice);
    cudaMemcpy(d_randstrobe_start_indices, index.randstrobe_start_indices.data(), index.randstrobe_start_indices.size() * sizeof(bucket_index_t), cudaMemcpyHostToDevice);
    std::cout << "memcpy1 execution time: " << GetTime() - t0 << " seconds, size " << index.randstrobes.size() * sizeof(RefRandstrobe) + index.randstrobe_start_indices.size() * sizeof(bucket_index_t) << std::endl;

    std::thread gpu_thread([&]() {
        sleep(5);
        for(int i = 0; i < 8; i++) {
            t0 = GetTime();
            cudaMemcpy(d_queries, h_queries, num_queries * sizeof(randstrobe_hash_t), cudaMemcpyHostToDevice);
            std::cout << "memcpy2 execution time: " << GetTime() - t0 << " seconds, size " << num_queries * sizeof(randstrobe_hash_t) << std::endl;

            t0 = GetTime();
            int threads_per_block = 1 << (6 + i % 4);
            int blocks_per_grid = (num_queries + threads_per_block - 1) / threads_per_block;
            gpu_find<<<blocks_per_grid, threads_per_block>>>(d_randstrobes, d_randstrobe_start_indices, d_queries, d_positions, num_queries, index.bits);
            //gpu_find_bs<<<blocks_per_grid / 32, threads_per_block>>>(d_randstrobes, d_randstrobe_start_indices, d_queries, d_positions, num_queries, index.bits);
            cudaDeviceSynchronize(); // Ensure all threads are finished
            std::cout << "synchronize execution time: " << GetTime() - t0 << " seconds" << std::endl;

            t0 = GetTime();
            cudaMemcpy(h_positions, d_positions, num_queries * sizeof(size_t), cudaMemcpyDeviceToHost);
            std::cout << "memcpy back execution time: " << GetTime() - t0 << " seconds" << std::endl;
        }
    });

    t0 = GetTime();
#pragma omp parallel for num_threads(16)
    for (int i = 0; i < num_queries; ++i) {
        positions[i] = index.find(queries[i]);
    }
    std::cout << "CPU execution time: " << GetTime() - t0 << " seconds" << std::endl;



    size_t check_sum = 0;
    for (size_t i = 0; i < 10; ++i) {
        int id = rand() % num_queries;
        std::cout << "Query " << id << ": Position " << positions[id] << std::endl;
        check_sum += positions[id];
    }
    std::cout << "check sum is " << check_sum << std::endl;

    gpu_thread.join();
    t0 = GetTime();
    cudaFree(d_randstrobes);
    cudaFree(d_randstrobe_start_indices);
    cudaFree(d_queries);
    cudaFree(d_positions);
    std::cout << "free execution time: " << GetTime() - t0 << " seconds" << std::endl;

    return 0;
}
