#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 256

// Returns 1 if triangle number, 0 if not.
static int checkTriangle(int cache[], int value) {

  // Check against cached values for match
  int i = 0, hit = 0;
  int result = 0;

  for (i = 0; i < TABLE_SIZE; i++) {
    if (cache[i] == value) {
      hit = 1; 
      break;
    }
  }

  return hit; // If no cache hit or hit during padding phase 
}

int main(void) {
  FILE *fptr = fopen("words.txt", "r");
  if (fptr == NULL) {
    printf("File failure\n");
    return 1;
  }
  
  int *cache = (int *)calloc(TABLE_SIZE, sizeof(int));

  // find first thousand values
  for (int i = 0; i < TABLE_SIZE; ++i) {
    cache[i] = (int)((0.5 * i) * (i + 1));     // Will not be truncating
    // printf("Cache value: %d\n", cache[i]);
  }

  int value = 0;
  int count = 0;

  char buffer[32];
  while (fscanf(fptr, "%s", buffer) != EOF) {
    // printf("Word: %s\n", buffer);
    int len = strlen(buffer);
    value = 0;
    for (int i = 0; i < len; ++i) {
      value += (buffer[i] - 'A' + 1);
    }
    // printf("%d\n", value);

    // Call to triangle comparison between buffer, location of buffer ptr, 
    if (checkTriangle(cache, value) == 1) count++;
  }

  printf("Number of Triangle Words in file: %d\n", count);

  return 0;
}
