#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // Include string.h

int main(void) {
  FILE *fptr = fopen("words.txt", "r");
  FILE *wptr = fopen("out.txt", "w");

  char buffer[32768];
  if (fptr == NULL || wptr == NULL){
    perror("Error opening file");
    return 1;
  }
  
  if (fscanf(fptr, "%32767s", buffer) != 1) { // Read with size limit
    fclose(fptr);
    fclose(wptr);
    perror("Error reading file");
    return 1;
  }

  // Split buffer into null terminated strings and print to new file
  char *string = (char *)malloc(64 * sizeof(char));
  if (string == NULL) {
      perror("malloc");
      fclose(fptr);
      fclose(wptr);
      return 1;
  }

  int apoct = 0, j = 0;
  int buffer_len = strlen(buffer); // Get the actual length of the string in buffer

  for (int i = 0; i < buffer_len; i++) { // Iterate only up to the length of the string
    if (buffer[i] == '\"') {
      apoct++;
      continue;
    }

    if (apoct == 2) {
      apoct = 0;

      // Write to file
      fprintf(wptr, "%s\n", string);

      // Clear the string buffer correctly
      memset(string, 0, 64);  // Use memset to clear the buffer

      j = 0;

      continue;
    }

    // If here in code then values are valid chars to be read into out.txt
    string[j] = buffer[i];
    j++;

    // Check for potential buffer overflow in 'string'
    if (j >= 63) {
      printf("Warning: Word too long, truncating.\n");
      string[63] = '\0'; // Ensure null termination
      fprintf(wptr, "%s\n", string);
      memset(string, 0, 64);
      j = 0;
      apoct = 0;
    }
  }

  fclose(fptr);
  fclose(wptr);
  free(string);

  return 0;
}
