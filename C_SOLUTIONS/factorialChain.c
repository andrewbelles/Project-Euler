#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000024

typedef struct Digits {
  unsigned long data[32];
  int count;
} Digits;


static long factorial(int n) {
  if (n == 0) return 1;
  long result = 1; 

  for (long i = 1; i <= n; i++) {
    result *= i;
  }

  return result;
}


static Digits splitValue(unsigned long input) {
  Digits result; 
  int i = 0;

  if (input == 0) {
    result.data[0] = 0;
    result.count = 1;
    return result;
  }

  unsigned value = 0;
  while (input > 0) {
    result.data[i] = input % 10;
    input /= 10;
    i++;
  }

  result.count = i;
  return result;
}

static int factChain(int cache[], int n) {

  // Check the resulting value with lookup. If it equals the values add 5 to them 
  // If they don't equal that in the lookup than you know that the value has been hit before
  // Store the count in cache and continue on
  int chain[64];
  int init = n;
  int i = 0, count = 1, done = 0;   // Start at a count of 1 for the first value in the chain 
  unsigned long value = 0;
  Digits split;

  while (1) {
    if (n < SIZE && cache[n] != 0) return cache[n] + count;

    if (count > 60) break; // We actually don't care about chains larger than 60

    // Split the current n into its principle digits 
    split = splitValue(n);
    value = 0;
    for (int j = 0; j < split.count; j++) {
      value += factorial(split.data[j]);
    }

    // Check lookup 
    for (int j = 0; j < count; j++) {
      if (chain[j] == value) return count;
    }

    // Add the value to the chain of previous elements and increment count 
    chain[count] = value; 

    n = value; 
    count++;
  }
  
  if (cache[init] == 0) cache[init] = count;  
  return count;
}


int main(void) {
  int n = 1e6; 
  int *cache = (int *)calloc(SIZE, sizeof(int));
  int count = 0, result = 0;

  for (int i = 1; i <= n; i++) {
    result = factChain(cache, i);
    // printf("Count for %d: %d\n", i, result);
    if (result == 60) count++;
  }

  printf("Number of Chains Matching Length 60: %d\n", count);
  free(cache);

  return 0;
}
