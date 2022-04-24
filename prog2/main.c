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


/** \brief shared region initialization */
extern void initialization(int _totalFileCount, int _K);

/** \brief prints explanation of how to run code */
static void printUsage(char *cmdName);

/** \brief worker threads return status array */
int *statusWorker;

/** \brief worker life cycle routine */
static void *worker(void *id);


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
  int opt;                                                                                        /* selected option */


  // argument handling
  do  
  {
    switch ((opt = getopt(argc, argv, "f:n:k:")))
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
    case 'n': /* numeric argument */
      if (atoi(optarg) < 1)
      {
        fprintf(stderr, "%s: number of threads must be greater or equal than 1\n", basename(argv[0]));
        printUsage(basename(argv[0]));
        return EXIT_FAILURE;
      }
      N = (int)atoi(optarg);
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

  pthread_t tIdCons[N];                                                        /* consumers internal thread id array */
  unsigned int cons[N];                                             /* consumers application defined thread id array */
  int *status_p;                                                                      /* pointer to execution status */                            
  
  struct timespec start, finish;                                                                      /* time limits */

  clock_gettime (CLOCK_MONOTONIC_RAW, &start);                                          /* begin of time measurement */
  
  statusWorker = malloc(sizeof(int) * N);                       /* memory allocation of worker's return status array */

  for (int i = 0; i < N; i++)                                                          /* incremental id attribution */
    cons[i] = i;


  initialization(fnip,K);                                                     /* initialization of the shared region */

  for (int i = 0; i < N; i++)                                                             /* worker htreads creation */
    if (pthread_create (&tIdCons[i], NULL, worker, &cons[i]) != 0)                                  /* thread worker */
       { perror ("error on creating thread consumer");
         exit (EXIT_FAILURE);
       }

  
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
    
    
    struct matrixFile curFile;                                         /* initialize structure with file information */
    curFile.filename = filenames[fCk];
    curFile.processedMatrixCounter = 0;
    curFile.order = order;
    curFile.nMatrix = numMatrix;


    putFileData (curFile);                    /* insert the current file's info into the shared region's files array */


    int incMCount = 0;                                                                 /* incremental matrix counter */
    while(incMCount!=numMatrix){                                     /* iterate over each matrix in the current file */
   
      struct matrixData  curMatrix;                               /* initialize structure with current matrix's info */
      curMatrix.fileIndex = fCk;
      curMatrix.matrixNumber = incMCount;
      curMatrix.order = order;
      curMatrix.determinant = 0;

      curMatrix.matrix = (double *)malloc(order * order * sizeof(double));        /* memory allocation of the matrix */
  
      fread(curMatrix.matrix, 8, order*order, fp);                                     /* read full matrix from file */
     
      putMatrixInFifo (curMatrix);                        /* add matrix to the shared region's FIFO processing queue */

      incMCount++;
    }
    
  
  }
  
  /* waiting for the termination of the intervening worker threads */
  for (int i = 0; i < N; i++)
  { if (pthread_join (tIdCons[i], (void *) &status_p) != 0)                                       
       { perror ("error on waiting for thread customer");
         exit (EXIT_FAILURE);
       }
    printf ("thread consumer, with id %u, has terminated: ", i);
    printf ("its status was %d\n", *status_p);
  }


  clock_gettime (CLOCK_MONOTONIC_RAW, &finish);   
  
  for (int g=0; g<fnip; g++) {                                                     /* printing results for each file */
    struct matrixFile *file = getFileData();                                     /* retrieve file from shared region */
    
    printf("\nMatrix File  %s\n", file->filename);
    printf("Number of Matrices  %d\n", file->nMatrix);
    printf("Order of the matrices  %d\n", file->order);

    for (int o =0;o<file->nMatrix; o++){
      printf("\tMatrix %d Result: Determinant = %.3e \n", o+1,file->matrixDeterminants[o]);
    }
        
  }
                                             /* end of measurement */
  printf ("\nElapsed time = %.6f s\n",  (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);

  exit (EXIT_SUCCESS);

}
/**
 *  \brief Function worker.
 *  Worker's life cycle
 * 
 *  Its role is to process a matrix and calculate its determinant.
 *
 *  \param wid pointer to application defined worker identification
 */

static void *worker(void *wid)
{
  unsigned int id = *((unsigned int *)wid); /* worker id */

  while(true){
      struct matrixData  *curMatrix = (struct matrixData *)malloc(sizeof(struct matrixData));   /* matrix to be processed
                                                                                                  memory alocation   */
      int contin = getSingleMatrixData(id, curMatrix);                          /* retrive matrix from shared region */
      if (contin == -1) {                                        /* if all files have been processed, end life cycle */
        break;
      }
      double det = getDeterminant(curMatrix->order,curMatrix->matrix);                     /* calculate determinant  */

      putResults(id,det, curMatrix->fileIndex, curMatrix->matrixNumber);      /* insert results in the shared region */
    
      free(curMatrix);                                                                      /* free allocated memory */
  }
 
  statusWorker[id] = EXIT_SUCCESS;
  pthread_exit (&statusWorker[id]);
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

