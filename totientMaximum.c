#include <stdio.h>

/**
  * Some things to note before beginning
  * Prime numbers have n - 1 relative primes
  * But the value will not be large since n/(n - 1) will have diminishing returns 
  *
  * Therefore the "totient maxs" will be values with many factors (therefore the value will be close to 1) 
*/

int primeFactorial(int n);
int totient(int value);

int main(void) { 
  int n = 1e6, res, primefac;
  double ratio = 0.0, max_ratio = 0.0;

  primefac = primeFactorial(n);
  printf("Prime Factorial of Primes With a Product under %d: %d\n", n, primefac);

  res = totient(primefac);
  printf("Totient for %d: %d\n", primefac, res);
  ratio = primefac/(float)res;


  printf("Max Ratio of %f from n: %d\n", ratio, primefac);

  return 0; 
}

static int prime(int n) {
  if (n <= 1) return 0;

  if (n <= 3) return 1;

  if (n % 2 == 0 || n % 3 == 0) return 0;

  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) return 0;
  } 
  return 1;
}

static inline int findNext(int n) {
  while (1) {
    n++;
    if (prime(n))
      return n;
  }
}

/**
 * Actually a super simple "recursive" function for this pattern. 
 * The max is passed to the factorial of primes 
 */
int primeFactorial(int n) {
  int product = 1;
  int next_prime = 0;

  while (product <= n) {
    // multiply by next prime number 
    next_prime = findNext(next_prime++);
    if (product * next_prime > n)
      break;

    product *= next_prime;
  }

  return product;
}


/**
  * Find the prime factors of value. Then use the formula n * (1 - 1/p1)(1 - 1/p2) ...
  */
int totient(int value) {
  if (value <= 0) return 0;
  int copy = 0, div = 0, tmp = 0, n = 0;

  for (int i = 1; i < value; i++) {
    
    copy = value;
    div = i;
    while (copy != 0) {
      tmp = copy;
      copy = div % copy;
      div = tmp;
    }
    if (div == 1) n++;
  }
  return n;
}
