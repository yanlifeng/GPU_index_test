#include "my_struct.hpp"

#include <gallatin/allocators/global_allocator.cuh>


__device__ void* my_malloc(size_t size, MemoryManagerType* mm) {
//    if (mm == nullptr) {
//        printf("Error: mm is nullptr in my_malloc!\n");
//        return nullptr;
//    }
    void* ptr = gallatin::allocators::global_malloc(size);
//    void* ptr = global_malloc(size);
//    void* ptr = mm->malloc(size);
//    void* ptr = malloc(size);
//    cudaError_t err;
//    void* ptr;
//    err = cudaMalloc(&ptr, size);
//    if (err != cudaSuccess) {
//        printf("device_malloc_free_test: cudaMalloc failed: %s\n", cudaGetErrorString(err));
//        return;
//    }
    if (ptr == nullptr) {
        printf("mm_memory allocation failed %d\n", size);
        assert(false);
    } else {
//        printf("mm_memory allocated %llu\n", size);
    }
    return ptr;
}

__device__ void my_free(void* ptr, MemoryManagerType* mm) {
//    if (mm == nullptr) {
//        printf("Error: mm is nullptr in my_malloc!\n");
//        return;
//    }
//    if(ptr != nullptr) mm->free(ptr);
//    free(ptr);
//    cudaError_t err = cudaFree(ptr);
//    if (err != cudaSuccess) {
//        printf("device_malloc_free_test: cudaFree failed: %s\n", cudaGetErrorString(err));
//        return;
//    }
    gallatin::allocators::global_free(ptr);
//    global_free(ptr);
}

__host__ void init_mm(uint64_t num_bytes, uint64_t seed) {
    gallatin::allocators::init_global_allocator(num_bytes, seed);
}

__host__ void free_mm() {
    gallatin::allocators::free_global_allocator();
}