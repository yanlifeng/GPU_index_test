#ifndef MY_STRUCT_HPP
#define MY_STRUCT_HPP

#include <iostream>
#include <stdexcept>
#include <cassert>
#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "Utility.h"

using MemoryManagerType = MultiOuroVLPQ;


//#include <gallatin/allocators/global_allocator.cuh>

struct Hit {
    int query_start;
    int query_end;
    int ref_start;
    int ref_end;
    __host__ __device__ bool operator<(const Hit& other) const {
        if(query_start == other.query_start) return ref_start < other.ref_start;
        return query_start < other.query_start;
    }

    __host__ __device__  bool operator==(const Hit& other) const {
        return query_start == other.query_start &&
               query_end == other.query_end &&
               ref_start == other.ref_start &&
               ref_end == other.ref_end;
    }
};

struct RescueHit {
    size_t position;
    unsigned int count;
    unsigned int query_start;
    unsigned int query_end;
    __host__ __device__ bool operator<(const RescueHit& other) const {
        if (count != other.count) return count < other.count;
        if (query_start != other.query_start) return query_start < other.query_start;
        return query_end < other.query_end;
    }
};

//inline __device__ void* my_malloc(size_t size, MemoryManagerType* mm) {
//    void* ptr = gallatin::allocators::global_malloc(size);
//    return ptr;
//}
//
//inline __device__ void my_free(void* ptr, MemoryManagerType* mm) {
//    gallatin::allocators::global_free(ptr);
//}

__device__ void* my_malloc(size_t size, MemoryManagerType* mm);
__device__ void my_free(void* ptr, MemoryManagerType* mm);

__host__ void init_mm(uint64_t num_bytes, uint64_t seed);
__host__ void free_mm();

template <typename T>
struct my_vector {
    T* data = nullptr;
    int length;
    int capacity;
    MemoryManagerType* mm_pool;

//    __device__ my_vector() : data(nullptr), length(0), capacity(0), mm_pool(nullptr) {
//    }

    __device__ my_vector(MemoryManagerType* mm_, int N = 4) {
        mm_pool = mm_;
        capacity = N;
        length = 0;
        data = (T*)my_malloc(capacity * sizeof(T), mm_pool);
    }

    __device__ ~my_vector() {
        if (data != nullptr) {
            my_free(data, mm_pool);
        }
        data = nullptr;
    }

    __device__ void resize(int new_capacity) {
        T* new_data;
        new_data = (T*)my_malloc(new_capacity * sizeof(T), mm_pool);
        for (int i = 0; i < length; ++i) new_data[i] = data[i];
        if (data != nullptr) my_free(data, mm_pool);
        data = new_data;
        capacity = new_capacity;
    }

    __device__ void push_back(const T& value) {
        if (length == capacity) {
            resize(capacity == 0 ? 1 : capacity * 2);
        }
        data[length++] = value;
    }

    __device__ int size() const {
        return length;
    }

    __device__ void clear() {
        length = 0;
    }

    __device__ void release() {
        if (data != nullptr) {
            my_free(data, mm_pool);
//            free(data);
        }
        data = nullptr;
    }


    __device__ T& operator[](int index) {
//        if (index >= length) {
//            printf("Index out of range %d %d\n", index, length);
//            assert(false);
//        }
        return data[index];
    }

    __device__ const T& operator[](int index) const {
//        if (index >= length) {
//            printf("Index out of range %d %d\n", index, length);
//            assert(false);
//        }
        return data[index];
    }


};

template <typename T>
__device__ void my_swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

template <typename T>
__device__ T my_max(T a, T b) {
    return a > b ? a : b;
}

template <typename T>
__device__ T my_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T1, typename T2>
struct my_pair {
    T1 first;
    T2 second;

    __device__ my_pair() : first(T1()), second(T2()) {}
    __device__ my_pair(const T1& a, const T2& b) : first(a), second(b) {}

    __device__ bool operator<(const my_pair& other) const {
        if (first < other.first) return true;
        if (first > other.first) return false;
        return second < other.second;
    }
};


#endif // MY_STRUCT_HPP

