#ifndef STROBEALIGN_RANDSTROBES_HPP
#define STROBEALIGN_RANDSTROBES_HPP

#include <vector>
#include <string>
#include <tuple>
#include <deque>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <inttypes.h>

#include "indexparameters.hpp"
#include "my_struct.hpp"
#include "hash.hpp"


using syncmer_hash_t = uint64_t;
using randstrobe_hash_t = uint64_t;

// a, A -> 0
// c, C -> 1
// g, G -> 2
// t, T, u, U -> 3
__device__ __host__ static unsigned char seq_nt4_table[256] = {
        0, 1, 2, 3,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};


__host__ __device__ static inline syncmer_hash_t syncmer_kmer_hash(uint64_t packed) {
    // return robin_hash(yk);
    // return yk;
    // return hash64(yk, mask);
    // return sahlin_dna_hash(yk, mask);
    return xxh64(packed);
}

__host__ __device__ static inline syncmer_hash_t syncmer_smer_hash(uint64_t packed) {
    // return ys;
    // return robin_hash(ys);
    // return hash64(ys, mask);
    return xxh64(packed);
}

__host__ __device__ static inline randstrobe_hash_t randstrobe_hash(syncmer_hash_t hash1, syncmer_hash_t hash2) {
    return hash1 + hash2;
}

struct RefRandstrobe {
    using packed_t = uint32_t;
    randstrobe_hash_t hash;
    uint32_t position;

    RefRandstrobe() { }

    RefRandstrobe(randstrobe_hash_t hash, uint32_t position, uint32_t packed)
        : hash(hash)
        , position(position)
        , m_packed(packed) { }

    bool operator< (const RefRandstrobe& other) const {
        if(hash == other.hash) return position < other.position;
        return hash < other.hash;
    }

    __host__ __device__ int reference_index() const {
        return m_packed >> bit_alloc;
    }

    __host__ __device__ int strobe2_offset() const {
        return m_packed & mask;
    }

private:
    static constexpr int bit_alloc = 8;
    static constexpr int mask = (1 << bit_alloc) - 1;
    packed_t m_packed; // packed representation of ref_index and strobe offset
};

struct QueryRandstrobe {
    randstrobe_hash_t hash;
    unsigned int start;
    unsigned int end;
    bool is_reverse;
};

std::ostream& operator<<(std::ostream& os, const QueryRandstrobe& randstrobe);

using QueryRandstrobeVector = std::vector<QueryRandstrobe>;

//QueryRandstrobeVector randstrobes_query(const std::string_view seq, const IndexParameters& parameters);

struct Randstrobe {
    randstrobe_hash_t hash;
    unsigned int strobe1_pos;
    unsigned int strobe2_pos;

    bool operator==(const Randstrobe& other) const {
        return hash == other.hash && strobe1_pos == other.strobe1_pos && strobe2_pos == other.strobe2_pos;
    }

    bool operator!=(const Randstrobe& other) const {
        return !(*this == other);
    }
};

std::ostream& operator<<(std::ostream& os, const Randstrobe& randstrobe);

struct Syncmer {
    syncmer_hash_t hash;
    size_t position;
    __host__ __device__ bool is_end() const {
        return hash == 0 && position == 0;
    }
};

/*
 * Iterate over randstrobes using a pre-computed vector of syncmers
 */
class RandstrobeIterator {
public:
    RandstrobeIterator(
        const std::vector<Syncmer>* syncmers,
        RandstrobeParameters parameters
    ) : syncmers(syncmers)
      , w_min(parameters.w_min)
      , w_max(parameters.w_max)
      , q(parameters.q)
      , max_dist(parameters.max_dist)
    {
        if (w_min > w_max) {
            throw std::invalid_argument("w_min is greater than w_max");
        }
    }

    Randstrobe next() {
        return get(strobe1_index++);
    }

    bool has_next() {
        return strobe1_index + w_min < (*syncmers).size();
    }

    __device__ RandstrobeIterator(
            const my_vector<Syncmer>* gpu_syncmers,
            RandstrobeParameters parameters
    ) : gpu_syncmers(gpu_syncmers)
            , w_min(parameters.w_min)
            , w_max(parameters.w_max)
            , q(parameters.q)
            , max_dist(parameters.max_dist)
    {
        if (w_min > w_max) {
            printf("w_min is greater than w_max\n");
			//            throw std::invalid_argument("w_min is greater than w_max");
		}
	}

	__device__ Randstrobe gpu_get(unsigned int strobe1_index) const {
		unsigned int w_end = (strobe1_index + w_max < (*gpu_syncmers).size() - 1) ? (strobe1_index + w_max) : (*gpu_syncmers).size() - 1;

		auto strobe1 = (*gpu_syncmers)[strobe1_index];
		auto max_position = strobe1.position + max_dist;
		unsigned int w_start = strobe1_index + w_min;

		uint64_t min_val = 0xFFFFFFFFFFFFFFFF;

		Syncmer strobe2 = strobe1;

		for (auto i = w_start; i <= w_end && (*gpu_syncmers)[i].position <= max_position; i++) {
			uint64_t hash_diff = (strobe1.hash ^ (*gpu_syncmers)[i].hash) & q;
			uint64_t res = 0;

			for (int j = 0; j < 64; j++) {
				res += ((hash_diff >> j) & 1);
			}

			if (res < min_val) {
				min_val = res;
				strobe2 = (*gpu_syncmers)[i];
			}
		}
        //Randstrobe r;
        //r.hash = randstrobe_hash(strobe1.hash, strobe2.hash);
        //r.strobe1_pos = static_cast<uint32_t>(strobe1.position);
        //r.strobe2_pos = static_cast<uint32_t>(strobe2.position);
        //printf("strobe1.position %u %lld %u\n", static_cast<uint32_t>(strobe1.position), strobe1.position, r.strobe1_pos);
        //return r;

		return Randstrobe{randstrobe_hash(strobe1.hash, strobe2.hash), static_cast<uint32_t>(strobe1.position), static_cast<uint32_t>(strobe2.position)};
	}



    __device__ Randstrobe gpu_next() {
        //Randstrobe r = gpu_get(strobe1_index++);
        //printf("strobe1.position %u\n", r.strobe1_pos);
        //return r;
        return gpu_get(strobe1_index++);
    }

    __device__ bool gpu_has_next() {
        return strobe1_index + w_min < (*gpu_syncmers).size();
    }

private:
    Randstrobe get(unsigned int strobe1_index) const;
    //__device__ Randstrobe gpu_get(unsigned int strobe1_index) const;
    const std::vector<Syncmer>* syncmers = nullptr;
    const my_vector<Syncmer>* gpu_syncmers = nullptr;
    const unsigned w_min = 0;
    const unsigned w_max = 0;
    const uint64_t q = 0;
    const unsigned int max_dist = 0;
    unsigned int strobe1_index = 0;
};

std::ostream& operator<<(std::ostream& os, const Syncmer& syncmer);

#define use_vector

#ifdef use_vector
class SyncmerIterator {
public:
    SyncmerIterator(const std::string_view* seq, SyncmerParameters parameters)
        : seq(seq), k(parameters.k), s(parameters.s), t(parameters.t_syncmer) { }

    Syncmer next();

    __device__ SyncmerIterator(my_vector<uint64_t>* vec, const char* gpu_seq, const size_t gpu_seq_len, SyncmerParameters parameters)
        : gpu_seq(gpu_seq), gpu_seq_len(gpu_seq_len), k(parameters.k), s(parameters.s), t(parameters.t_syncmer) {
        gpu_qs = vec;
        gpu_qs_front_pos = 0;
		}
	__device__ ~SyncmerIterator() {
//		if (gpu_qs != nullptr) {
//			free(gpu_qs);
//			gpu_qs = nullptr;
//		}
	}

//    __device__ Syncmer gpu_next();
    __device__ Syncmer gpu_next() {
        for ( ; i < gpu_seq_len; ++i) {
            int c = seq_nt4_table[(uint8_t) gpu_seq[i]];
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
                (*gpu_qs).push_back(hash_s);
                // not enough hashes in the queue, yet
                if ((*gpu_qs).size() - gpu_qs_front_pos < k - s + 1) {
                    continue;
                }
                if ((*gpu_qs).size() - gpu_qs_front_pos == k - s + 1) { // We are at the last s-mer within the first k-mer, need to decide if we add it
                    for (int j = gpu_qs_front_pos; j < (*gpu_qs).size(); j++) {
                        if ((*gpu_qs)[j] < qs_min_val) {
                            qs_min_val = (*gpu_qs)[j];
                            qs_min_pos = i - k + j - gpu_qs_front_pos + 1;
                        }
                    }
                }
                else {
                    // update queue and current minimum and position
                    gpu_qs_front_pos++;
                    if(gpu_qs_front_pos >= (*gpu_qs).size()) printf("size GG\n");

                    if (qs_min_pos == i - k) { // we popped the previous minimizer, find new brute force
                        qs_min_val = UINT64_MAX;
                        qs_min_pos = i - s + 1;
                        for (int j = (*gpu_qs).size() - 1; j >= gpu_qs_front_pos; j--) { //Iterate in reverse to choose the rightmost minimizer in a window
                            if ((*gpu_qs)[j] < qs_min_val) {
                                qs_min_val = (*gpu_qs)[j];
                                qs_min_pos = i - k + j - gpu_qs_front_pos + 1;
                            }
                        }
                    } else if (hash_s < qs_min_val) { // the new value added to queue is the new minimum
                        qs_min_val = hash_s;
                        qs_min_pos = i - s + 1;
                    }
                }
                if (qs_min_pos == i - k + t) { // occurs at t:th position in k-mer
                    uint64_t yk = xk[0] < xk[1] ? xk[0] : xk[1];
                    auto syncmer = Syncmer{syncmer_kmer_hash(yk), i - k + 1};
                    i++;
                    return syncmer;
                }
            } else {
                // if there is an "N", restart
                qs_min_val = UINT64_MAX;
                qs_min_pos = -1;
                l = xs[0] = xs[1] = xk[0] = xk[1] = 0;
                (*gpu_qs).clear();
                gpu_qs_front_pos = 0;
            }
        }
        return Syncmer{0, 0}; // end marker
    }

private:
    const std::string_view* seq = nullptr;
    const char* gpu_seq = nullptr;
    const size_t gpu_seq_len = 0;
    const size_t k = 0;
    const size_t s = 0;
    const size_t t = 0;

    const uint64_t kmask = (1ULL << 2*k) - 1;
    const uint64_t smask = (1ULL << 2*s) - 1;
    const uint64_t kshift = (k - 1) * 2;
    const uint64_t sshift = (s - 1) * 2;
    std::deque<uint64_t> *qs = nullptr;  // s-mer hashes
    my_vector<uint64_t> *gpu_qs = nullptr;
    int gpu_qs_front_pos = 0;
    uint64_t qs_min_val = UINT64_MAX;
    size_t qs_min_pos = -1;
    size_t l = 0;
    uint64_t xk[2] = {0, 0};
    uint64_t xs[2] = {0, 0};
    size_t i = 0;
};

#else

class SyncmerIterator {
public:
    SyncmerIterator(const std::string_view* seq, SyncmerParameters parameters)
        : seq(seq), k(parameters.k), s(parameters.s), t(parameters.t_syncmer) { }

    Syncmer next();

    __device__ SyncmerIterator(const char* gpu_seq, const size_t gpu_seq_len, SyncmerParameters parameters)
        : gpu_seq(gpu_seq), gpu_seq_len(gpu_seq_len), k(parameters.k), s(parameters.s), t(parameters.t_syncmer) {
            gpu_qs = (uint64_t*)malloc(128 * sizeof(uint64_t));
            gpu_qs_size = 0;
            gpu_qs_front_pos = 0;
		}
	__device__ ~SyncmerIterator() {
		if (gpu_qs != nullptr) {
			free(gpu_qs);
			gpu_qs = nullptr;
		}
	}

//    __device__ Syncmer gpu_next();
    __device__ Syncmer gpu_next() {
        for ( ; i < gpu_seq_len; ++i) {
            int c = seq_nt4_table[(uint8_t) gpu_seq[i]];
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
                gpu_qs[gpu_qs_size++] = hash_s;
                if(gpu_qs_size > 256) printf("size GG %d\n", gpu_qs_size);
                // not enough hashes in the queue, yet
                if (gpu_qs_size - gpu_qs_front_pos < k - s + 1) {
                    continue;
                }
                if (gpu_qs_size - gpu_qs_front_pos == k - s + 1) { // We are at the last s-mer within the first k-mer, need to decide if we add it
                    for (int j = gpu_qs_front_pos; j < gpu_qs_size; j++) {
                        if (gpu_qs[j] < qs_min_val) {
                            qs_min_val = gpu_qs[j];
                            qs_min_pos = i - k + j - gpu_qs_front_pos + 1;
                        }
                    }
                }
                else {
                    // update queue and current minimum and position
                    gpu_qs_front_pos++;
                    if(gpu_qs_front_pos >= gpu_qs_size) printf("size GG\n");

                    if (qs_min_pos == i - k) { // we popped the previous minimizer, find new brute force
                        qs_min_val = UINT64_MAX;
                        qs_min_pos = i - s + 1;
                        for (int j = gpu_qs_size - 1; j >= gpu_qs_front_pos; j--) { //Iterate in reverse to choose the rightmost minimizer in a window
                            if (gpu_qs[j] < qs_min_val) {
                                qs_min_val = gpu_qs[j];
                                qs_min_pos = i - k + j - gpu_qs_front_pos + 1;
                            }
                        }
                    } else if (hash_s < qs_min_val) { // the new value added to queue is the new minimum
                        qs_min_val = hash_s;
                        qs_min_pos = i - s + 1;
                    }
                }
                if (qs_min_pos == i - k + t) { // occurs at t:th position in k-mer
                    uint64_t yk = xk[0] < xk[1] ? xk[0] : xk[1];
                    auto syncmer = Syncmer{syncmer_kmer_hash(yk), i - k + 1};
                    i++;
                    return syncmer;
                }
            } else {
                // if there is an "N", restart
                qs_min_val = UINT64_MAX;
                qs_min_pos = -1;
                l = xs[0] = xs[1] = xk[0] = xk[1] = 0;
                gpu_qs_size = 0;
                gpu_qs_front_pos = 0;
            }
        }
        return Syncmer{0, 0}; // end marker
    }

private:
    const std::string_view* seq = nullptr;
    const char* gpu_seq = nullptr;
    const size_t gpu_seq_len = 0;
    const size_t k = 0;
    const size_t s = 0;
    const size_t t = 0;

    const uint64_t kmask = (1ULL << 2*k) - 1;
    const uint64_t smask = (1ULL << 2*s) - 1;
    const uint64_t kshift = (k - 1) * 2;
    const uint64_t sshift = (s - 1) * 2;
    std::deque<uint64_t> *qs = nullptr;  // s-mer hashes
    uint64_t *gpu_qs = nullptr;
    int gpu_qs_front_pos = 0;
    int gpu_qs_size = 0;
    uint64_t qs_min_val = UINT64_MAX;
    size_t qs_min_pos = -1;
    size_t l = 0;
    uint64_t xk[2] = {0, 0};
    uint64_t xs[2] = {0, 0};
    size_t i = 0;
};
#endif
/*
 * Iterate over randstrobes while generating syncmers on the fly
 *
 * Unlike RandstrobeIterator, this does not need a pre-computed vector
 * of syncmers and therefore uses less memory.
 */
//class RandstrobeGenerator {
//public:
//    RandstrobeGenerator(
//        const std::string& seq,
//        SyncmerParameters syncmer_parameters,
//        RandstrobeParameters randstrobe_parameters
//    ) : syncmer_iterator(SyncmerIterator(seq, syncmer_parameters))
//      , w_min(randstrobe_parameters.w_min)
//      , w_max(randstrobe_parameters.w_max)
//      , q(randstrobe_parameters.q)
//      , max_dist(randstrobe_parameters.max_dist)
//    { }
//
//    Randstrobe next();
//    Randstrobe end() const { return Randstrobe{0, 0, 0}; }
//
//private:
//    SyncmerIterator syncmer_iterator;
//    const unsigned w_min;
//    const unsigned w_max;
//    const uint64_t q;
//    const unsigned int max_dist;
//    std::deque<Syncmer> syncmers;
//};


//std::vector<Syncmer> canonical_syncmers(const std::string_view seq, SyncmerParameters parameters);

#endif

