#ifndef MY_STRUCT_HPP
#define MY_STRUCT_HPP

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

#define vec_block_num_shift 3
#define vec_block_size_shift 16
#define vec_pre_block_size (1ll << vec_block_size_shift)
#define vec_block_size (1ll << (vec_block_size_shift + vec_block_num_shift))

struct my_pool {
    void* data;
    int size;
    int pos;
    int used[1 << vec_block_num_shift];
};

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


#define use_my_pool

#ifdef use_my_pool
template <typename T>
struct my_vector {
    T* data = nullptr;
    int length;
    int capacity;
    my_pool *mpool;
    int free_pos;

    __device__ my_vector() : data(nullptr), length(0), capacity(0), mpool(nullptr), free_pos(-1) {
        //printf("null vec %d\n", free_pos);
    }

    __device__ my_vector(my_pool* mpool_, int cap_ = vec_pre_block_size) {
        mpool = mpool_;
        free_pos = -1;
        for(int i = 0; i < (1 << vec_block_num_shift); i++) {
            if(mpool->used[i] == 0) {
                free_pos = i;
                mpool->used[i] = 1;
                break;
            }
        }
        data = (T*) (mpool->data + free_pos * cap_);
        length = 0;
        capacity = cap_ / sizeof(T);
        if(free_pos == -1) printf("mpool OOM\n");
        //else printf("get pos %d, ptr %p, size %d\n", free_pos, data, capacity);
    }

    __device__ ~my_vector() {
        //if(free_pos != -1) {
        //    mpool->used[free_pos] = 0;
        //    printf("free1 pos %d\n", free_pos);
        //}
        //free_pos = -1;
    }

    __device__ void release() {
        if(free_pos != -1) {
            mpool->used[free_pos] = 0;
            //printf("free2 pos %d\n", free_pos);
        }
        free_pos = -1;
    }

    __device__ void push_back(const T& value) {
        data[length++] = value;
        if(length >= capacity) {
            printf("OOM %d %d, T %d\n", length, capacity, sizeof(T));
//            exit(0);
        }
    }

    __device__ int size() const {
        return length;
    }

    __device__ void clear() {
        length = 0;
    }


    __device__ T& operator[](int index) {
        return data[index];
    }

    __device__ const T& operator[](int index) const {
        return data[index];
    }
};

#else

template <typename T>
struct my_vector {
    T* data = nullptr;
    int capacity = 64;
    int length = 0;
    void *d_buffer_ptr;

    __device__ my_vector(void *buffer_ptr, int N = 64) {
        capacity = N;
        length = 0;
        data = (T*)malloc(capacity * sizeof(T));
        memset(data, 0, capacity * sizeof(T));
    }

    __device__ ~my_vector() {
        if (data != nullptr) {
            free(data);
            //cudaFree(data);
        }
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


    __device__ T& operator[](int index) {
        if (index >= length) {
            // Custom error handling instead of throwing an exception
            printf("Index out of range %d %d\n", index, length);
            return data[0]; // Return some valid reference (this can be adjusted)
        }
        return data[index];
    }

    __device__ const T& operator[](int index) const {
        if (index >= length) {
            // Custom error handling instead of throwing an exception
            printf("Index out of range %d %d\n", index, length);
            return data[0]; // Return some valid reference (this can be adjusted)
        }
        return data[index];
    }

private:
    __device__ void resize(int new_capacity) {
        T* new_data;
        //cudaMalloc(&new_data, new_capacity * sizeof(T));
        new_data = (T*)malloc(new_capacity * sizeof(T));
        printf("resize %d\n", new_capacity);

        if (new_data == nullptr) {
            // Custom error handling for memory allocation failure
            printf("Memory allocation failed\n");
            return;
        }

        // Copy old data to new memory
        for (int i = 0; i < length; ++i) {
            new_data[i] = data[i];
        }

        // Free old memory
        if (data != nullptr) {
            free(data);
            //cudaFree(data);
        }

        // Update the vector with new data
        data = new_data;
        capacity = new_capacity;
    }
};
#endif


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

