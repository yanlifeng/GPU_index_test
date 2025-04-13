#ifndef MY_STRUCT_HPP
#define MY_STRUCT_HPP

#include <iostream>
#include <stdexcept>
#include <cassert>

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
        if(ref_id != nn.ref_id) return ref_id < nn.ref_id;
        return is_rc < nn.is_rc;
    }
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

__device__ void* my_malloc(size_t size);
__device__ void my_free(void* ptr);

__host__ void init_mm(uint64_t num_bytes, uint64_t seed);
__host__ void free_mm();
__host__ void print_mm();


template <typename T>
__host__ __device__ T my_max(T a, T b) {
    return a > b ? a : b;
}

template <typename T>
__host__ __device__ T my_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
__host__ __device__ T my_abs(T a) {
    return a < 0 ? -a : a;
}

struct my_string {
    char* data = nullptr;
    int slen;
    __host__ __device__ my_string() : data(nullptr), slen(0) {}
    __device__ my_string(char* str, int len) {
        slen = len;
        data = str;
    }
    __host__ __device__ ~my_string() {
        data = nullptr;
    }
    __device__ int length() const {
        return slen;
    }
    __device__ int size() const {
        return slen;
    }
    __device__ const char* c_str() const {
        return data;
    }
    __device__ char operator[](int index) const {
        return data[index];
    }
    __device__ char& operator[](int index) {
        return data[index];
    }
    __device__ my_string substr(int start, int len) const {
        int real_len = my_min(len, slen - start);
        return my_string(data + start, real_len);
    }
    __device__ int find(const my_string& str, int start = 0) const {
        for (int i = start; i <= slen - str.slen; ++i) {
            bool found = true;
            for (int j = 0; j < str.slen; ++j) {
                if (data[i + j] != str[j]) {
                    found = false;
                    break;
                }
            }
            if (found) return i;
        }
        return -1;
    }
    __device__ bool operator== (const my_string& other) const {
        if (slen != other.slen) return false;
        for (int i = 0; i < slen; ++i) {
            if (data[i] != other[i]) return false;
        }
        return true;
    }
};




template <typename T>
struct my_vector {
    T* data = nullptr;
    int length;
    int capacity;

//    __host__ my_vector() : data(nullptr), length(0), capacity(0) {}

    __device__ my_vector(int N = 4) {
        capacity = N;
        length = 0;
        data = (T*)my_malloc(capacity * sizeof(T));
    }

    __device__ void init(int N = 4) {
        capacity = N;
        length = 0;
        data = (T*)my_malloc(capacity * sizeof(T));
    }

    __device__ ~my_vector() {
        if (data != nullptr) my_free(data);
        data = nullptr;
    }

    __device__ void resize(int new_capacity) {
        T* new_data;
        new_data = (T*)my_malloc(new_capacity * sizeof(T));
        for (int i = 0; i < length; ++i) new_data[i] = data[i];
        if (data != nullptr) my_free(data);
        data = new_data;
        capacity = new_capacity;
    }

    __device__ void push_back(const T& value) {
        if (length == capacity) {
            resize(capacity == 0 ? 1 : capacity * 2);
        }
        data[length++] = value;
    }
    __device__ void emplace_back() {
        if (length == capacity) {
            resize(capacity == 0 ? 1 : capacity * 2);
        }
        data[length++] = T();
    }

    __device__ int size() const {
        return length;
    }

    __device__ void clear() {
        length = 0;
    }

    __device__ void release() {
        if (data != nullptr) my_free(data);
        data = nullptr;
        length = 0;
        capacity = 0;
    }

    __device__ T& operator[](int index) {
        return data[index];
    }

    __host__ __device__ const T& operator[](int index) const {
        return data[index];
    }

    __device__ T& back() {
        return data[length - 1];
    }

    __device__ void move_from(my_vector<T>& src) {
        if (data != nullptr) my_free(data);
        data = src.data;
        length = src.length;
        capacity = src.capacity;
        src.data = nullptr;
        src.length = 0;
        src.capacity = 0;
    }

    __device__ bool empty() const {
        return length == 0;
    }
};

template <typename T>
__device__ void my_swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}


template <typename T1, typename T2>
struct my_pair {
    T1 first;
    T2 second;
    __device__ bool operator<(const my_pair& other) const {
        if (first < other.first) return true;
        if (first > other.first) return false;
        return second < other.second;
    }
};


#endif // MY_STRUCT_HPP

