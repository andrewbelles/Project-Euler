#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 100024

int main(void) {
  FILE *fptr = fopen("words.txt", "r");
  if (fptr == NULL) {
    printf("File failure\n");
    return 1;
  }
  
  int *cache = (int *)calloc(TABLE_SIZE, sizeof(int));

  // find first thousand values
  for (int i = 0; i < 1000; ++i) {
    cache[i] = (int)((0.5 * i) * (i + 1));     // Will not be truncating
  }

  int value = 0;
  int count = 0;
  float interm = 0.0;

  char buffer[256];
  while (fscanf(fptr, "%s", buffer) != EOF) {
    int len = strlen(buffer);
    value = 0;
    for (int i = 0; i < len; ++i) {
      value += (buffer[i] - 'A' + 1);
    }
  }

  // Calculate some buffer of reference triangle values (?)
  // 2 times the nth triangle number is n^2 + n 
  // (1/2)sqrt(val) ~= (1/2)(sqrt(n)*n) = t
  //
  // Lookup table up to size t(100). If decomposition of val > t(100) calculate another 100
}
