/**
 *  \file matrixDeterminantCols.c
 *
 *  \brief Problem name: Matrix Determinant Calculation With CUDA using column reduction.
 *
 *  The objective is to get the matrices within files and calculate their determinants,
 *  by utilizing the CUDA capabilities.
 *
 *  \author Mário Silva, Pedro Marques - June 2022
 */

#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "matrix_utils_col.h"
#include <unistd.h>

/**
 *  \brief Print results of the matrix determinant calculations.
 */
void printResults(char *filename, int numMatrices, int order, double *determinants);

/**
 *  \brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  \param cmdName string with the name of the command
 */
static void printUsage(char *cmdName);

/**
 *  \brief
 *
 *  Design and flow
 *
 *  1. Read and process the command line.
 *  2. Read the number of matrices in the file.
 *  3. Read the order of the matrices in the file.
 *  4. Initialize the array of matrices and determinants.
 *  5. Load all the matrices.
 *  6. Copy the matrices from the host to the GPU global memory.
 *  7. Process and calculate the matrices determinants.
 *  8. Retrieve the array of determinants from the GPU back to the host.
 *  9. Print results.
 *  10. For each matrix, calculate determinant using the CPU.
 *  11. Print total elapsed time for both CPU and GPU operations.
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */
int main(int argc, char **argv)
{
  printf("%s Starting...\n", argv[0]);

  // set up device
  int dev = 0; /* Device set up */
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev)); /* Show the current device's properties */
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  char *filenames[16]; /* array of file's names  */ /* array of file's names  */
  int fnip = 0;                                     /* filename insertion pointer */
  int opt;

  do
  {
    switch ((opt = getopt(argc, argv, "f:")))
    {
    case 'f': /* file name */
      if (optarg[0] == '-')
      {
        fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
        printUsage(basename(argv[0]));
        return EXIT_FAILURE;
      }
      if (fnip >= 16) /* at most 16 files */
      {
        fprintf(stderr, "%s: Too many files to unpack. At Most 16\n", basename(argv[0]));
        printUsage(basename(argv[0]));
        return EXIT_FAILURE;
      }

      filenames[fnip++] = optarg;
      break;

    case 'h': /* help mode */
      printUsage(basename(argv[0]));
      return EXIT_SUCCESS;
    case '?': /* invalid option */
      fprintf(stderr, "%s: invalid option\n", basename(argv[0]));
      printUsage(basename(argv[0]));
      return EXIT_FAILURE;
    case -1:
      break;
    }
  } while (opt != -1);

  if (argc == 1)
  {
    fprintf(stderr, "%s: invalid format\n", basename(argv[0]));
    printUsage(basename(argv[0]));
    return EXIT_FAILURE;
  }

  double iElaps = 0;    /* total elapsed time using CUDA */
  double iElapsCpu = 0; /* total elapsed time using CPU */
  for (int fileIndex = 0; fileIndex < fnip; fileIndex++)
  { /* process each file in filenames array */
    FILE *fp = fopen(filenames[fileIndex], "r");
    if (fp == NULL)
    {
      printf("Error: could not open file %s\n", filenames[fileIndex]);
      return EXIT_FAILURE;
    }
    int numMatrices;
    if (fread(&numMatrices, sizeof(int), 1, fp) == 0) /* Get the number of matrices in file */
    {
      printf("Error: could not read from file %s\n", filenames[fileIndex]);
      return EXIT_FAILURE;
    }

    int order;
    if (!fread(&order, sizeof(int), 1, fp)) /* Get the order of the matrices in file */
    {
      printf("Error: could not read from file %s\n", filenames[fileIndex]);
      return EXIT_FAILURE;
    }

    // malloc host memory
    double *matricesHost = (double *)malloc(sizeof(double) * numMatrices * order * order); /* allocate host memory for the matrices */
    double *determinantsHost = (double *)malloc(sizeof(double) * numMatrices);             /* allocate host memory for the determinants */

    // malloc device global memory all the matrices and the results array
    double *determinants;
    double *matricesDevice;
    CHECK(cudaMalloc((void **)&determinants, sizeof(double) * numMatrices));                   /* Device memory allocation for determinants array */
    CHECK(cudaMalloc((void **)&matricesDevice, sizeof(double) * numMatrices * order * order)); /* Device memory allocation for matrices */

    if (!fread(matricesHost, sizeof(double), numMatrices * order * order, fp)) /* Read all matrices to host array */
    {
      printf("Error: could not read from file %s\n", filenames[fileIndex]);
      return EXIT_FAILURE;
    }

    // transfer data from host to device
    CHECK(cudaMemcpy(matricesDevice, matricesHost, sizeof(double) * numMatrices * order * order, cudaMemcpyHostToDevice)); /* Set number of matrices at device's memory */

    // invoke kernel at host side
    dim3 grid(numMatrices, 1); /* Create a grid of one block per matrix */
    dim3 block(order, 1);      /* Create a thread per column for each block */

    double iStart = seconds();

    calcDeterminantsCols<<<grid, block>>>(matricesDevice, determinants); /* Calculate pivots for each column */
    CHECK(cudaDeviceSynchronize());

    iElaps += seconds() - iStart; /* sum processing time with CUDA Kernel */

    CHECK(cudaGetLastError()); /* check for a kernel error */

    CHECK(cudaMemcpy(determinantsHost, determinants, sizeof(double) * numMatrices, cudaMemcpyDeviceToHost)); /* copy kernel result back to host */

    // check device results
    printResults(filenames[fileIndex], numMatrices, order, determinantsHost); /* print determinant calculation results */

    /* free device global memory */
    CHECK(cudaFree(determinants));
    CHECK(cudaFree(matricesDevice));

    double iStartCpu = seconds();
    for (int matrixPointer = 0; matrixPointer < numMatrices; matrixPointer++)
    {                                                                  /* Calculate determinants using CPU */
      double *matrix = (matricesHost + order * order * matrixPointer); /* get matrix to process */
      double cpuDeterminant = getDeterminant(order, matrix);           /* calculate determinant  */
    }
    iElapsCpu += seconds() - iStartCpu; /* sum processing time with CPU */

    free(matricesHost);     /* free the array of matrices at the host */
    free(determinantsHost); /* free the array of determinants at the host */

    // reset device
    CHECK(cudaDeviceReset()); /* reset device */
  }
  printf("\nGPU Elapsed time = %.6f s\n", iElaps);    /* Elapsed Time Using the Cuda Kernel */
  printf("\nCPU Elapsed time = %.6f s\n", iElapsCpu); /* Elapsed Time Using the CPU */
  exit(EXIT_SUCCESS);
}

/**
 *  \brief Print results of the matrix detemrinant calculations
 */
void printResults(char *filename, int numMatrices, int order, double *determinants)
{
  printf("\nMatrix File  %s\n", filename);
  printf("Number of Matrices  %d\n", numMatrices);
  printf("Order of the matrices  %d\n", order);

  for (int i = 0; i < numMatrices; i++)
  {
    printf("\tMatrix %d Result: Determinant = %.3e \n", i + 1, determinants[i]);
  }
}

/**
 *  \brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  \param cmdName string with the name of the command
 */
static void printUsage(char *cmdName)
{
  fprintf(stderr, "\nSynopsis: %s OPTIONS [filename]\n"
                  "  OPTIONS:\n"
                  "  -h      --- print this help\n"
                  "  -f      --- filename to process\n",
          cmdName);
}