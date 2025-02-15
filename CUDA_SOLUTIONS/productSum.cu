#include <iostream>
#include <array>
#include <cuda_runtime.h>
#include <cstdlib>

#define CACHESIZE 15000

// BINARY TREE FOR NUMBER'S FACTOR
  // We want to init factors to the number of possible prime values of the largest value in cache 
  // Each value in exp array will be exponent of each prime 
struct factorCache {
  int *fac;
  int *exp;
};    // Exp and fac are 2d flat arrays going up to n * (# prime vals sub(n)) 

// Compute 
__host__ __device__ static bool prime(int n) {
  if (n <= 1) return false;

  if (n <= 3) return true;

  if (n % 2 == 0 || n % 3 == 0) return false;

  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) return false;
  } 
  return true;
}

// Implementation of Pollard's rho algorithm
__global__ void factorCacheKernel(factorCache *factors) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  if (idx >= CACHESIZE + 1) return;

  int value = idx;
}

// Naive check for prime value
constexpr static bool constPrime(int n) {
  if (n <= 1) return false;

  if (n <= 3) return true;

  if (n % 2 == 0 || n % 3 == 0) return false;

  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) return false;
  } 
  return true;
}

// Fetch number of prime numbers during compile time 
constexpr int precomputePrimeCount() {
  int count = 0;
  // Find all primes under or at cache size 
  for (int i = 1; i <= CACHESIZE; i++) {
    if (constPrime(i) != 1) continue;
    count++;
  }
  return count;
}

// Fetch array of prime numbers for cache during compile time 
constexpr auto precomputePrimeArray() {
  constexpr int prime_count = precomputePrimeCount();
  int ct = 0;
  std::array<int, prime_count> primes{};

  for (int i = 1; i <= CACHESIZE; i++) {
    if (constPrime(i) != 1) continue;
    primes[ct++] = i;
  }

  return primes;
} 

// Returns head to cached factor tree 
__host__ factorCache *createCache() {
  factorCache *factors;

  cudaMallocManaged(&factors, 1 * sizeof(factorCache));
  cudaMallocManaged(&factors->exp, (CACHESIZE ) * sizeof(int));

  // Dimension of thread call
  dim3 threads(256);
  dim3 blocks(((CACHESIZE + 1) + threads.x - 1) / threads.x);

  // Kernel call -> push cache onto GPU
  cudaMemPrefetchAsync(factors, (CACHESIZE + 1) * sizeof(factorCache), 0, NULL);
  factorCacheKernel<<<blocks, threads>>>(factors);
  cudaDeviceSynchronize();

  return factors;
}

__host__ int main(void) {
  int n = 12000;
  unsigned long long sum = 0;
  // compute prime cache at compile time 
  constexpr auto primes = precomputePrimeArray();
  int prime_count = primes.size();
  int *prime_cache = new int[prime_count];
  factorCache *factors;

  std::copy(primes.begin(), primes.end(), prime_cache);

  cudaMallocManaged(&factors, sizeof(factorCache));
  cudaMallocManaged(&factors->exp, CACHESIZE * prime_count * sizeof(int));
  cudaMallocManaged(&factors->fac, CACHESIZE * prime_count * sizeof(int));

  for (int i = 0; i < primes.size(); i++) {
    std::cout << i << "th prime: " << prime_cache[i] << '\n';
  }

  std::cout << "Cache Size: " << static_cast<long>((CACHESIZE * prime_count) * sizeof(int) / 1e6) << " MB\n";


  std::cout << "Sum of minimal product sum numbers: " << sum << '\n';
}
