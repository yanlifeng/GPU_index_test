#ifndef STROBEALIGN_NAM_HPP
#define STROBEALIGN_NAM_HPP

#include <vector>
#include <set>
#include <array>
#include "index.hpp"
#include "randstrobes.hpp"

// Non-overlapping approximate match
struct Nam {
    int nam_id;
    int query_start;
    int query_end;
    int query_prev_hit_startpos;
    int ref_start;
    int ref_end;
    int ref_prev_hit_startpos;
    int n_hits = 0;
    int ref_id;
    float score;
//    unsigned int previous_query_start;
//    unsigned int previous_ref_start;
    bool is_rc = false;

    __host__ __device__ int ref_span() const {
        return ref_end - ref_start;
    }

    __host__ __device__ int query_span() const {
        return query_end - query_start;
    }

    // TODO where use this <
//    bool operator < (const Nam& nn) const {
//        if(query_end != nn.query_end) return query_end < nn.query_end;
//        return nam_id < nn.nam_id;
//    }

    __host__ __device__ bool operator < (const Nam& nn) const {
        if(score != nn.score) return score > nn.score;
        if(query_end != nn.query_end) return query_end < nn.query_end;
        if(query_start != nn.query_start) return query_start < nn.query_start;
        if(ref_end != nn.ref_end) return ref_end < nn.ref_end;
        if(ref_start != nn.ref_start) return ref_start < nn.ref_start;
    }
};

std::pair<float, std::vector<Nam>> find_nams(
    const QueryRandstrobeVector &query_randstrobes,
    const StrobemerIndex& index
);

std::vector<Nam> find_nams_rescue(
    const QueryRandstrobeVector &query_randstrobes,
    const StrobemerIndex& index,
    unsigned int rescue_cutoff
);

std::ostream& operator<<(std::ostream& os, const Nam& nam);

#endif
