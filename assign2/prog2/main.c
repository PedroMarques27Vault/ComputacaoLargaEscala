/**
 *  \file main.c 
 *
 *  \brief  Problem name: Matrix Determinant Calculation With Multithreading.
 * 
 *  The objective is to matrices within files and calculate their determinant.
 *
 *  The main thread is responsible for reading the matrices from files and providing them to
 *  the shared region. Aferwards, worker threads should retrieve them and calculate the determinant.
 *  Then, these results are saved in the shared region where the main thread can retrieve them and 
 *  present them. 
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Generator thread of the intervening entities.
 *
 *  \author Pedro Marques - April 2022
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "probConst.h"
#include "matrixutils.h"
#include "sharedregion.h"
#include <stdbool.h>
#include <libgen.h>
#include <libgen.h>
#include <string.h>
#include <mpi.h>



/** \brief prints explanation of how to run code */
static void printUsage(char *cmdName);


/**
 *  \brief Main thread.
 *
 *  Design and flow of the main thread:
 *
 *  1 - Process the arguments from the command line.
 *
 *  2 - Initialize the shared region with the necessary structures.
 *
 *  3 - Create the worker threads.
 * 
 *  4 - Continuously provide matrices to the shared region, for the worker to process
 *
 *  5 - Wait for the worker threads to terminate.
 *
 *  6 - Print final results.
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */

int main(int argc, char *argv[])
{
  int N = DN;                                                                             /* number of worker threads */
  int K = M;                                                                  /*Size of FIFO Queue in Shared Region */

 
  char *filenames[10];                                                                     /* array of file's names  */
  int fnip = 0;                                                                        /* filename insertion pointer */
  int opt;  
  
                                                                                        /* selected option */
  int rank, size;

  // MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* This program requires at least 2 processes */
  if (size < 2)
  {
    fprintf(stderr, "Requires at least two processes.\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  if (rank == 0){
    int WORKSTATUS = 1;
    // argument handling
    do  
    {
      switch ((opt = getopt(argc, argv, "f:k:")))
      {
      case 'f': /* file name */
        if (optarg[0] == '-')
        {
          fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
          printUsage(basename(argv[0]));
          return EXIT_FAILURE;
        }
        if (fnip>=10) /* at most 10 files */                                    
        {
          fprintf(stderr, "%s: Too many files to unpack. At Most 10\n", basename(argv[0]));
          printUsage(basename(argv[0]));
          return EXIT_FAILURE;
        }

        filenames[fnip++] = optarg;
        break;
      case 'k': /* numeric argument */
        if (atoi(optarg) < 1)
        {
          fprintf(stderr, "%s: size of the queue must be greater or equal than 1\n", basename(argv[0]));
          printUsage(basename(argv[0]));
          return EXIT_FAILURE;
        }
        K = (int)atoi(optarg);
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

    matrices = malloc(sizeof(struct matrixData) * K);                              /* initialize FIFO/matrices Queue  */
    files = (struct matrixFile *)malloc(fnip * sizeof(struct matrixFile));       /* initialize files array  */
                                              

    

    for (int fCk = 0;fCk<fnip;fCk++){                                          /* process each file in filenames array */

      FILE *fp = fopen(filenames[fCk], "r");

      if (fp == NULL)
      {
          printf("Error: could not open file %s", filenames[fCk]);
          return 1;
      }

      int numMatrix;
      fread(&numMatrix, 4, 1, fp);                                               /* get number of matrices in the file */
      
      int order;
      fread(&order, 4, 1, fp);                                                /* get order of the matrices in the file */
      

      int incMCount = 0;         

      (files+fCk)->filename = file.filename;
      (files+fCk)->order = order;
      (files+fCk)->nMatrix = numMatrix;
      (files+fCk)->matrixDeterminants = (double *)malloc(numMatrix * sizeof(double));
      int nProcesses = 0;
      
      int rest = numMatrix%size;
      int iterations = (numMatrix-rest)/size;

      for (int iter = 0; iter < iterations + 1; iter++){
        int nMatrixToRead = size;
        if (iter == iterations) nMatrixToRead = rest;
        for (int nProc = 1; nProc<nMatrixToRead+1; nProc++){
            double *matrix = (double *)malloc(order * order * sizeof(double));        /* memory allocation of the matrix */
            fread(matrix, 8, order*order, fp);                                     /* read full matrix from file */
            MPI_Send(WORKSTATUS, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD); 
            MPI_Send(order, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD);        /* Send order*/
            MPI_Send(incMCount, 1, MPI_UNSIGNED_INT, nProc, 0, MPI_COMM_WORLD);/* Send matrix index*/
            MPI_Send(matrix, order*order, MPI_DOUBLE, nProc, 0, MPI_COMM_WORLD);
        }

        for (int nProc = 1; nProc<nMatrixToRead+1; nProc++){
            int curMatrixNumber;
            double determinant;
            MPI_Recv(&curMatrixNumber, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&determinant, 1, MPI_DOUBLE, nProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* update struct with new results */
            ((filesData + fCk)->matrixDeterminants+curMatrixNumber) = determinant;
        }
      }
    }
    WORKSTATUS = 0;
    for (int nProc = 1; nProc<size+1; nProc++){
          MPI_Send(WORKSTATUS, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD); 
    }
  
  }else{
    
    while(true){
      int curWorkStatus;
      MPI_Recv(&curWorkStatus, maxBytesPerChunk, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (curWorkStatus ==0) break;

      int order, matrixIndex;
      double *matrix = (double *)malloc(order * order * sizeof(double)); 
      MPI_Recv(&order, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&matrixIndex, 1, MPI_UNSIGNED_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(matrix, order*order, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    

      
      double det = getDeterminant(curMatrix->order,curMatrix->matrix);                     /* calculate determinant  */

      MPI_Send(matrixIndex, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(determinantV, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                                                                     /* free allocated memory */
    }
  }

  

  exit (EXIT_SUCCESS);

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
  fprintf(stderr, "\nSynopsis: %s OPTIONS [filename / number of threads / size of the FIFO queue]\n"
                  "  OPTIONS:\n"
                  "  -h      --- print this help\n"
                  "  -f      --- filename to process\n"
                  "  -n      --- number of threads\n"
                  "  -k      --- size of the FIFO queue in the monitor\n",
          cmdName);
}

