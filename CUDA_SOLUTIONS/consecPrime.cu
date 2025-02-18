#include <iostream>
#include <array>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdlib>

#define CACHESIZE 125000

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
consteval bool constPrime(int n) {
  if (n <= 1) return false;

  if (n <= 3) return true;

  if (n % 2 == 0 || n % 3 == 0) return false;

  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) return false;
  } 
  return true;
}

// Fetch number of prime numbers during compile time 
consteval int precomputePrimeCount() {
  int count = 1;
  // Find all primes under or at cache size 
  for (int i = 3; i <= CACHESIZE; i += 2) {
    if (constPrime(i) != 1) continue;
    count++;
  }
  return count;
}

// Fetch array of prime numbers for cache during compile time 
consteval auto precomputePrimeArray() {
  constexpr auto prime_count = precomputePrimeCount();
  std::array<int, prime_count> primes{};
  int ct = 1;

  primes[ct] = 2;
  for (int i = 3; i <= CACHESIZE; i += 2) {
    if (constPrime(i) != 1) continue;
    if (ct < prime_count) {
      primes[ct++] = i;
    }
  }
  return primes;
} 

// Check cache for value (if its prime). Perform binary search 
__host__ bool checkCache(int *cache, int size, int value) { 
  int start_index = size / 2; 

  if (cache[start_index] == value) return true;

}

__host__ int main(void) {
  int n = 12000;
  unsigned long long sum = 0;
  // compute prime cache at compile time 
  constexpr auto primes = precomputePrimeArray();
  int prime_count = primes.size();
  int *prime_cache = new int[prime_count];
  factorCache *factors;

  bool end = false;

  std::copy(primes.begin(), primes.end(), prime_cache);

  for (int i = 1; i < primes.size() && !end; i++) {
    long sum = 0;
    for (int j = 1; j < i; j++) {
      sum += static_cast<long>(primes[j]);
      if (sum >= 1000000) {
        end = true;
        break;
      };
    }
    std::cout << "Sum of Primes Under " << primes[i] << " : " << sum << '\n'; 
  }


  // for (int i = 0; i < primes.size(); i++) {
  //   std::cout << i << "th prime: " << prime_cache[i] << '\n';
  // }


  std::cout << "Sum of minimal product sum numbers: " << sum << '\n';
}
