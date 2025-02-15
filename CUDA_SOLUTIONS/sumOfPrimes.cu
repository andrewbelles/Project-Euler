#include <cuda_runtime_api.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

__host__ __device__ static bool prime(int n) {
  if (n <= 1) return false;

  if (n <= 3) return true;

  if (n % 2 == 0 || n % 3 == 0) return false;

  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) return false;
  } 
  return true;
}

__global__ void primeKernel(int n, ::uint64_t *prime_sum) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  ::uint64_t value = static_cast<::uint64_t>(idx);
  if (idx > n) value = 0; 

  extern __shared__ ::uint64_t partial_sum[];

  if (!prime(value)) value = 0;
  partial_sum[threadIdx.x] = value;
  __syncthreads();

  for (int i = 256 / 2; i > 0; i /= 2) {
    if (threadIdx.x < i) {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + i];
    }
    __syncthreads();
  }

  
  if (threadIdx.x == 0) {
    atomicAdd(prime_sum, partial_sum[0]);
  }
}

__host__ uint64_t computeSum(int n) {
  ::uint64_t *result_ptr, result;
  cudaMallocManaged(&result_ptr, sizeof(::uint64_t));
  *result_ptr = 0;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  cudaMemPrefetchAsync(result_ptr, sizeof(::uint64_t), 0, NULL);
  primeKernel<<<blocks, threads, threads * sizeof(::uint64_t)>>>(n, result_ptr);
  cudaDeviceSynchronize();

  result = *result_ptr;
  cudaFree(result_ptr);
  return result;
}

__host__ int main(void) {
  int n = 2000000;
  ::uint64_t sum = 0;

  sum = computeSum(n);

  std::cout << "Sum of Prime Numbers under 2 million: " << sum << '\n';
}
