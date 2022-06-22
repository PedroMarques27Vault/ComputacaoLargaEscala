#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "matrix_utils_col.h"
#include <unistd.h>

/**
 *  \brief Print results of the matrix determinant calculations.
 */
void printResults(char * filename, int numMatrices, int order, double * determinants);

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
 *  1 - Read and process the command line.
 *  2 - For every file:
 *    2.1 - Read the number of matrices in file
 *    2.2 - Read the order of the matrices in file
 *    2.3 - Initialize the file's array of matrices and determinants
 *    2.4 - Load all the matrices into the array
 *    2.5 - Copy the matrices from the host to the kernel's memory
 *    2.6 - Process and Calculate the Matrix' Determinant:
 *        2.6.1 - Each block has a thread per Column
 *        2.6.2 - For each Column:
 *                2.6.2.1 - Calculate pivots
 *                2.6.2.2 - Subtract pivot from Column
 *    2.7 - Retrieve results from kernel back to host
 *    2.8 - Print results
 *    2.8 - For each matrix, calculate determinant using the CPU
 *  3 - Print total elapsed time for both CPU and Kernel operations
 *  4 - Finalize.
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
  int dev = 0;                                                                                          /* Device set up */
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));                                                     /* Show the current device's properties */
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  char *filenames[16];                                                                                  /* array of file's names  */                                            /* array of file's names  */
  int fnip = 0;                                                                                         /* filename insertion pointer */
  int opt;  
  
  do  
    {
      switch ((opt = getopt(argc, argv, "f:")))
      {
      case 'f':                                                                                         /* file name */
        if (optarg[0] == '-')
        {
          fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
          printUsage(basename(argv[0]));
          return EXIT_FAILURE;
        }
        if (fnip>=16)                                                                                   /* at most 16 files */                                    
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
   
    
  double iElaps = 0;                                                                                    /* total elapsed time using CUDA */
  double iElapsCpu = 0;                                                                                 /* total elapsed time using CPU */
  for (int fileIndex = 0;fileIndex<fnip;fileIndex++){                                                   /* process each file in filenames array */                                                                     
    FILE *fp = fopen(filenames[fileIndex], "r");
    if (fp == NULL)
    {
      printf("Error: could not open file %s\n", filenames[fileIndex]);
      return EXIT_FAILURE;
    }
    int numMatrices;
    if (fread(&numMatrices, sizeof(int), 1, fp) == 0)                                                   /* Get the number of matrices in file */                        
    {
      printf("Error: could not read from file %s\n", filenames[fileIndex]);
      return EXIT_FAILURE;
    }

    int order;
    if (!fread(&order, sizeof(int), 1, fp))                                                             /* Get the order of the matrices in file */                                  
    {
      printf("Error: could not read from file %s\n", filenames[fileIndex]);
      return EXIT_FAILURE;
    }

    // malloc host memory
    double *matricesHost = (double *)malloc(sizeof(double) * numMatrices * order * order);              /* allocate host memory for the matrices */
    double *determinantsHost = (double *)malloc(sizeof(double) * numMatrices);                          /* allocate host memory for the determinants */

    // malloc device global memory the order and all the matrices
    int *orderDevice;
    int *currentCol;
    double *determinants;
    double *matricesDevice;
    CHECK(cudaMalloc((void **)&orderDevice, sizeof(int)));                                              /* Device memory allocation for matrix order */
    CHECK(cudaMalloc((void **)&currentCol, sizeof(int)));                                               /* Device memory allocation of current column being processed */
    CHECK(cudaMalloc((void **)&determinants, sizeof(double) * numMatrices));                            /* Device memory allocation for determinants array */
    CHECK(cudaMalloc((void **)&matricesDevice, sizeof(double) * numMatrices * order * order));          /* Device memory allocation for matrices */

    if (!fread(matricesHost, sizeof(double), numMatrices * order * order, fp))                          /* Read all matrices to host array */
    { 
      printf("Error: could not read from file %s\n", filenames[fileIndex]);
      return EXIT_FAILURE;
    }

    // transfer data from host to device
    CHECK(cudaMemcpy(orderDevice, &order, sizeof(int), cudaMemcpyHostToDevice));                        /* Set matrix order at device's memory */
    CHECK(cudaMemcpy(matricesDevice, matricesHost, sizeof(double) * numMatrices*order*order, cudaMemcpyHostToDevice));  /* Set number of matrices at device's memory */
 
    // invoke kernel at host side 
    dim3 grid(numMatrices, 1);                                                                          /* Create a grid of one block per matrix */
    dim3 block(order, 1);                                                                               /* Create a thread per column for each block */

    double iStart = seconds();
    for (int iteration = 0; iteration < order; iteration++)
    {
      // update the currentCol value on the device
      CHECK(cudaMemcpy(currentCol, &iteration, sizeof(int), cudaMemcpyHostToDevice));                   /* update the currentCol value on the device */

      calcPivots<<<grid, block>>>(matricesDevice, orderDevice, determinants, currentCol);               /* Calculate pivots for each column */
      CHECK(cudaDeviceSynchronize());                                                                   /* Wait for every pivot calculation */

      subtractPivots<<<grid, block>>>(matricesDevice, orderDevice, determinants, currentCol);           /* Subtract pivot from each column */
      CHECK(cudaDeviceSynchronize());                                                                   /* Wait for every subtraction to finish */
    }
    iElaps += seconds() - iStart;                                                                       /* sum processing time with CUDA Kernel */

    CHECK(cudaGetLastError());                                                                          /* check for a kernel error */

    CHECK(cudaMemcpy(determinantsHost, determinants, sizeof(double) * numMatrices, cudaMemcpyDeviceToHost));  /* copy kernel result back to host */

    // check device results
    printResults(filenames[fileIndex], numMatrices, order, determinantsHost);                           /* print determinant calculation results */

    CHECK(cudaFree(orderDevice));                                                                        /* free device global memory */
    CHECK(cudaFree(currentCol));
    CHECK(cudaFree(determinants));
    CHECK(cudaFree(matricesDevice));


    double iStartCpu = seconds();
    for (int matrixPointer = 0; matrixPointer<numMatrices;matrixPointer++){                             /* Calculate determinants using CPU */
        double *matrix = (matricesHost+order*order*matrixPointer);                                      /* get matrix to process */
        double cpuDeterminant = getDeterminant(order,matrix);                                           /* calculate determinant  */
    }
    iElapsCpu += seconds() - iStartCpu;                                                                 /* sum processing time with CPU */


    free(matricesHost);                                                                                 /* free the array of matrices at the host */
    free(determinantsHost);                                                                             /* free the array of determinants at the host */

    // reset device
    CHECK(cudaDeviceReset());                                                                           /* reset device */

    printf("\nGPU Elapsed time = %.6f s\n", iElaps);                                                    /* Elapsed Time Using the Cuda Kernel */ 
    printf("\nCPU Elapsed time = %.6f s\n", iElapsCpu);                                                 /* Elapsed Time Using the CPU */
  } 
  exit(EXIT_SUCCESS);
}

/**
 *  \brief Print results of the matrix detemrinant calculations
 */
void printResults(char * filename, int numMatrices, int order, double * determinants)
{
  printf("\nMatrix File  %s\n", filename);
  printf("Number of Matrices  %d\n", numMatrices);
  printf("Order of the matrices  %d\n", order);

  for (int i=0; i<numMatrices; i++)
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