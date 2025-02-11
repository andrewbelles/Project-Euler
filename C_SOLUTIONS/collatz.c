#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 100000

// Counter functions 

static inline long even(long n) {
  return n / 2;
} 

static inline long odd(long n) {
  return 3 * n + 1;
}

// Added cache size n + 1 to place the count for the rest of a sequence at that value. 
long collatz(long cache[], long n) {
  long count = 0;
  while (n != 1) {
    if (n < MAX_SIZE) {
      if (cache[n] != 0)
        return cache[n];
    }

    n = (n % 2 == 0) ? even(n) : odd(n); 
    count++; 
  }
  cache[n] = count;

  return count;
}

// Simple memoization trick for collatz sequences 
int main(void) {
  long n = 1e6;
  long *cache = (long *)calloc(MAX_SIZE, sizeof(long));
  cache[1] = 1;

  long max = 0, start = 0;
  for (long i = 1; i <= n; ++i) {
    int res = collatz(cache, i);
    if (res > max) {
      max = res;
      start = i;
    }
  }

  printf("Maximum Chain on interval 0 -> %ld is %ld starting from %ld\n", n, max, start);
  
  return 0;
}
