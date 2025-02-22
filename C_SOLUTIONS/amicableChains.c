#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 1000000

// Holds individual factors 
typedef struct Factor {
  int data[255];
  int count;
} Factor;
  
// Holds data for each amicable chain
typedef struct Chain {
  int smallest;
  int count;
} Chain;

// Return set of unique factors
static Factor factorize(int n) {
  Factor result;
  int i = 2;

  // Init first value to 1
  result.data[0] = 1;
  result.count   = 1;
  
  // Iterates up to sqrt n (if perfect square)
  while (i * i <= n) {
    // If divisible by i add
    if (n % i == 0) {
      result.data[result.count++] = i;
      // Check if value is perfect square or value itself to avoid adding
      if (i * i != n && n / i != n) {
        result.data[result.count++] = n / i;
      }
    }
    i++;
  }
  return result;
}

// Returns smallest member and size of chain in struct 
static Chain amicableChain(Chain chain_cache[SIZE], int n) {
  Chain chain = {
    .smallest = n,
    .count    = 0
  };

  int init = n;
  Factor factors, prevfac;
  int value = 0, preval = 0;

  // While true loop bad I know 
  while (1) {
    // return empty struct for n exceeding limit 
    if (n >= SIZE) return (Chain){
      .smallest = 0,
      .count    = 0
    };

    // Check for cache hit on current n and append values to chain struct 
    if (chain_cache[n].smallest != 0) {
      // Compare the cache'd smallest value with current value and swap if true
      if (chain.smallest > chain_cache[n].smallest) {
        chain.smallest = chain_cache[n].smallest;
      } 
      // Add running count with cache count 
      chain.count += chain_cache[n].count;
      return chain;
    }

    // Get digits from factorization 
    factors = factorize(n);  
    value = 0;

    // Add factors together 
    for (int i = 0; i < factors.count; i++) {
      value += factors.data[i]; 
    }

    if (value >= SIZE) {
      return (Chain){
        .smallest = 0,
        .count    = 0
      };
    }

    // If first iteration then smallest should be first value 
    if (chain.count == 0) chain.smallest = value;

    // Check for repeat values
    if (value == init) {
      
      prevfac = factorize(preval);
      preval = 0;
      for (int i = 0; i < prevfac.count; i++) {
        preval += prevfac.data[i]; 
      }

      if (preval == n) {
        chain_cache[init] = chain;
        return chain;
      } else {
        return (Chain){
          .smallest = 0,
          .count    = 0
        };
      }
    } 

    // Iterate and check for new min
    preval = n;
    n = value;
    chain.count++;
    if (chain.smallest > value) chain.smallest = value;
  }
  
  chain_cache[init] = chain;

  return chain;
}

int main(void) {
  // We want this on the heap to avoid 8mb on stack 
  Chain *cache = (Chain *)malloc(SIZE * sizeof(Chain));
  memset(cache, 0, SIZE * sizeof(Chain));
  Chain result, best = {
    .smallest = 0,
    .count    = 0
  }; 

  // Loop from 1 to 1e6
  for (int i = 1; i <= SIZE; i++) {
    result = amicableChain(cache, i); 
    // printf("I: %d\n  Small: %d\n  Count %d\n", i, result.smallest, result.count);
    if (best.count < result.count) {
      best = result;
      printf("Updated\n");
      printf("  S: %d\n  L: %d\n", best.smallest, best.count);
    }
  }

  printf("Smallest value %d in the largest chain length %d.\n", best.smallest, best.count);

  return 0;
}
