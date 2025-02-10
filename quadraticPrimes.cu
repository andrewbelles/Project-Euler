#include <cuda_runtime_api.h>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
  * We are maximizing the number of primes that a function returns consectively
  * 2D optimization. 
  * Is there some kind of matrix form that enables a more elegant optimization
*/
struct Table {
  int a;
  int b;
  int count;
};


__host__ __device__ static bool prime(int n) {
  if (n <= 1) return false;

  if (n <= 3) return true;

  if (n % 2 == 0 || n % 3 == 0) return false;

  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) return false;
  } 
  return true;
}

/**
 * Marks all values 0 to n that are prime then allocates and fills primes array with said values 
 */
__host__ void primeSieve(int **primes, int *n) {
  int prime_ct = 0, p = 0;
  int *prime_indices = (int *)calloc(*n, sizeof(int));

  // Count prime values on host 
  for (int i = 0; i < *n; ++i) {
    if (prime(i + 1)) {
      prime_indices[i] = 1;
      prime_ct++;
    }
  }

  // Shift prime values into array 
  cudaMallocManaged(primes, prime_ct * sizeof(int));
  for (int i = 0; i < *n; ++i)
    if (prime_indices[i] == 1) (*primes)[p++] = i + 1;

  *n = prime_ct;
}

__device__ static inline int function(int a, int b, int n) {
  return n * n + a * n + b;
}

__global__ static void consecPrimes(int *primes, Table *func_vals, const int n) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= (4*n*n)) return;

  // Calculate a and b coordinate values  
  unsigned int pair = idx / 4;

  unsigned int i = pair / n; 
  unsigned int j = pair % n;

  int dx = ((idx % 4) & 1) ? -1 : 1;
  int dy = ((idx % 4) & 2) ? -1 : 1;

  // Find four values. (a, b), (-a, b), (a, -b), (-a, -b)
  int a = primes[i] * dx;
  int b = primes[j] * dy;

  // Loops over 0 <= count until value isn't prime. 
  int k = 0;
  unsigned int count = 0;
  while (1) {
    int val = function(a, b, k);
    if (!prime(val) || val < 2) break;
    k++;
    count++;
  }

  // Set count for idx
  func_vals[idx].count = count;
  func_vals[idx].a     = a;
  func_vals[idx].b     = b;
}

/**
 * Host function to manage call to parallel computation of matrix/table
 */
__host__ void manageMaximization(int *primes, Table *func_vals, const int n) {
  int sq = 4*n*n;
  dim3 threads(256); 
  dim3 blocks((sq + threads.x - 1) / threads.x);

  // Provide each thread a single pair of primes to fill out
  consecPrimes<<<blocks, threads>>>(primes, func_vals, n);
  cudaDeviceSynchronize();
}


__host__ int main(void) {
  int n = 1000;
  int *primes;
  struct Table *func_vals;

  // Place all primes in order 
  primeSieve(&primes, &n);

  // Flat 2D table to hold key pair (a,b) and count associated 
  cudaMallocManaged(&func_vals, (4*n*n) * sizeof(Table));

  // Call to Kernel to manage GPU Kernel calls
  manageMaximization(primes, func_vals, n);

  struct Table max_val = func_vals[0];
  for (int i = 0; i < (4*n*n); ++i) {
    if (max_val.count < func_vals[i].count) max_val = func_vals[i];
  }

  // Value in 0th index is max a and b 
  std::cout << "a and b (a,b): (" << max_val.a << "," << max_val.b << "), count: " << max_val.count << '\n';
  std::cout << "Product of maximizing a and b: " << max_val.a * max_val.b << '\n'; 


  return 0;
}
