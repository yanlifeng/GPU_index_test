#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

int main() {
    const size_t size = 1024ll * 1024 * 1024;  // 100 MB
    void *host_ptr, *device_ptr;

    // Allocate pageable host memory (default)
    //host_ptr = malloc(size);
    cudaHostAlloc(&host_ptr, size, cudaHostAllocDefault);
    memset(host_ptr, 0, size);
    cudaMalloc(&device_ptr, size);

    // Warm up
    cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);

    // Measure H2D bandwidth
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();
    double h2d_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Host-to-Device Bandwidth: " << (size / 1e9) / h2d_time << " GB/s" << std::endl;

    // Measure D2H bandwidth
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    double d2h_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Device-to-Host Bandwidth: " << (size / 1e9) / d2h_time << " GB/s" << std::endl;

    // Cleanup
    cudaFree(device_ptr);
    cudaFree(host_ptr);

    return 0;
}

