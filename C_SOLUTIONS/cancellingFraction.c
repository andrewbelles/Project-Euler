#include <stdio.h>

typedef struct {
  int a, b;
} Pair;

int main(void) {

  Pair values[4]; 
  int found = 0, val1 = 0, val2 = 0;
  float frac = 0.0, simple = 0.0;

  // Iterate over outside values, shared value 
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      for (int k = 0; k < 10; ++k) { 

        if (i == 0) continue;
        if (k == 0) continue;
        if (i == j && i == k) continue;

        val1 = i * 10 + k;
        val2 = k * 10 + j;

        if (val2 < val1) continue;

        frac   = (float)val1 / (float)val2;
        simple = (float)i / (float)j; 

        // If the naively simplified fraction equals the actual fraction fill array
        if (frac == simple) {
          values[found].a = val1;
          values[found].b = val2;
          found++;
          printf("Fraction: %d/%d\n", values[found - 1].a, values[found - 1].b);
        }
      }
    }
  }

  val1 = val2 = 1;
  for (int i = 0; i < 4; ++i) {
    val1 *= values[i].a;
    val2 *= values[i].b;
  }
  
  float denominator = (float)val2 / (float)val1;
  if (denominator == (float)(val2 / val1)) {
    printf("denominator: %f\n", denominator);
  } else {
    printf("do something else %f, %d/%d\n", denominator, val1, val2);
  }

  return 0;
}
