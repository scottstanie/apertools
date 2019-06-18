/*
 * Program to find the number of zero data elements in a file
 * of complex float data.
 * Used to quickly detect bad output (of .geo files, for example)
 */
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s filename\n", argv[0]);
    return EXIT_FAILURE;
  }

  float buf[2];
  char *filename = argv[1];

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failure to open %s. Exiting.\n", filename);
    return EXIT_FAILURE;
  }
  int nbytes = sizeof(float); // real, imag floats

  int num_elements = 0;
  int num_zeros = 0;
  while (fread(buf, nbytes, 2, fp) > 1) {
    ++num_elements;
    if (!buf[0] && !buf[1]) {
      ++num_zeros;
    }
  }
  printf("Percentage zeros for %s: %f\n", filename,
         (float)num_zeros / num_elements);

  return EXIT_SUCCESS;
}
