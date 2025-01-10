#ifndef MY_STRUCT_HPP
#define MY_STRUCT_HPP

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

// Device deque implementation using CUDA memory management (cudaMalloc and cudaFree)
template <typename T>
struct my_deque {
    T* data;          // Pointer to device memory holding the deque data
    size_t capacity;  // Capacity of the deque
    size_t size;      // Current number of elements in the deque
    size_t front_idx; // Front index of the deque (used for pop_front)
    size_t back_idx;  // Back index of the deque (used for push_back)

    // Constructor: Initialize the deque with zero capacity and size
    __device__ my_deque() : data(nullptr), capacity(0), size(0), front_idx(0), back_idx(0) {}

    // Destructor: Free the allocated device memory
    __device__ ~my_deque() {
        if (data != nullptr) {
            free(data);
            //cudaFree(data);
        }
    }

    // Resize the deque's internal memory if needed
    __device__ void resize(size_t new_capacity) {
        T* new_data;
        // Allocate new memory on device
        //cudaMalloc((void**)&new_data, new_capacity * sizeof(T));
        new_data = (T*)malloc(new_capacity * sizeof(T));
        if (new_data == nullptr) {
//            throw std::bad_alloc();
            printf("new_capacity * sizeof(T) %d, std::bad_alloc()\n", new_capacity * sizeof(T));
        }

        // Copy old elements to new memory (circular copy)
        for (size_t i = 0; i < size; ++i) {
            size_t index = (front_idx + i) % capacity;
//            cudaMemcpy(&new_data[i], &data[index], sizeof(T), cudaMemcpyDeviceToDevice);
//            memcpy(&new_data[i], &data[index], sizeof(T));
            new_data[i] = data[index];
        }

        // Free the old memory and update the pointers
        free(data);
        //cudaFree(data);
        data = new_data;
        capacity = new_capacity;
        front_idx = 0;
        back_idx = size;
    }

    // Push an element to the back of the deque
    __device__ void push_back(const T& value) {
        if (size == capacity) {
            resize(capacity == 0 ? 1 : capacity * 2); // Double the capacity if full
        }

        // Store the value at the back index
//        cudaMemcpy(&data[back_idx], &value, sizeof(T), cudaMemcpyHostToDevice);
        data[back_idx] = value;

        // Update the back index and size
        back_idx = (back_idx + 1) % capacity;
        size++;
    }

    // Pop an element from the front of the deque
    __device__ void pop_front() {
        if (size == 0) {
//            throw std::underflow_error("Deque is empty");
            printf("Deque is empty\n");
        }

        // Move the front index and decrease the size
        front_idx = (front_idx + 1) % capacity;
        size--;
    }

    // Get the size (number of elements) of the deque
    __device__ size_t get_size() const {
        return size;
    }

    // Clear the deque
    __device__ void clear() {
        size = 0;
        front_idx = 0;
        back_idx = 0;
    }

    // Access an element at the given index (with bounds checking)
    __device__ T& operator[](size_t index) {
        if (index >= size) {
//            throw std::out_of_range("Index out of bounds");
            printf("Index out of bounds\n");
        }

        size_t real_index = (front_idx + index) % capacity;
        return data[real_index];
    }

    // Const version of the operator[]
    __device__ const T& operator[](size_t index) const {
        if (index >= size) {
//            throw std::out_of_range("Index out of bounds");
            printf("Index out of bounds\n");
        }

        size_t real_index = (front_idx + index) % capacity;
        return data[real_index];
    }
};

struct my_pool {
    void* data;
    int size;
    int pos;
};

#define use_my_pool

#define vec_pre_block_size (1ll << 14)
#define vec_block_size (1ll << 16)

#ifdef use_my_pool
template <typename T>
struct my_vector {
    T* data = nullptr;
    int length;
    int capacity;
    my_pool *mpool;

    __device__ my_vector(my_pool* mpool_) {
        mpool = mpool_;
        data = (T*) (mpool->data + mpool->pos * vec_pre_block_size);
        mpool->pos++;
        if(mpool->pos * vec_pre_block_size >= vec_block_size) {
            printf("mpool->pos is %d, OOM\n", mpool->pos);
        }
        length = 0;
        capacity = vec_pre_block_size / sizeof(T);
    }

    __device__ ~my_vector() {
    }

    __device__ void push_back(const T& value) {
        data[length++] = value;
        if(length >= capacity) {
            printf("OOM %d %d\n", length, capacity);
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
#endif // MY_STRUCT_HPP

