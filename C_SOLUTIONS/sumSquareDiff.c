#include <stdio.h> 
#include <math.h>

int main(void) {

  int sq_sum = 0, sum_sq = 0;

  for (int i = 1; i <= 100; i++) {
    sq_sum += i;
  
    sum_sq += (i * i);
    //printf(">> I: %d\n  Sum Square: %d\n  Square Sum: %d\n", i, sum_sq, sq_sum);
  } 
  sq_sum *= sq_sum;

  int final = sq_sum - sum_sq;
  printf("Difference: %d\n", final);

  return 0;
}
