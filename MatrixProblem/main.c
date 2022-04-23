/**
 *  \file main.c
 *
 *  \brief
 *
 TODO
 *
 *  \author 
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "matrixutils.h"
#include "sharedregion.h"
#include <stdbool.h>
#include <libgen.h>
#include <libgen.h>
#include <string.h>

extern void initialization(int _totalFileCount, int _K);

static void printUsage(char *cmdName);
/** \brief worker threads return status array */
int *statusWorker;


/** \brief worker life cycle routine */
static void *worker(void *id);


/**
 *  \brief Main thread.
 *
 *  TODO
 */

int main(int argc, char *argv[])
{
  int N = 8;                                                                /* number of worker threads. Default = 4 */
  int K = 12;                                                                   /*Size of FIFO Queue in Shared Region*/

  int *status_p;                                                                   
  char *filenames[10];


  int fnip = 0;
  int opt;                                       /* selected option */

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

  struct timespec start, finish;                                                                      /* time limits */

  clock_gettime (CLOCK_MONOTONIC_RAW, &start);                              /* begin of measurement */
  statusWorker = malloc(sizeof(int) * N);


  for (int i = 0; i < N; i++)
    cons[i] = i;


  initialization(fnip,K);

  for (int i = 0; i < N; i++)
    if (pthread_create (&tIdCons[i], NULL, worker, &cons[i]) != 0)                             /* thread consumer */
       { perror ("error on creating thread consumer");
         exit (EXIT_FAILURE);
       }

  /* waiting for the termination of the intervening entities threads */
  for (int fCk = 0;fCk<fnip;fCk++){

    FILE *fp = fopen(filenames[fCk], "r");

    if (fp == NULL)
    {
        printf("Error: could not open file %s", filenames[fCk]);
        return 1;
    }
    // number of matrices in the file
    int numMatrix;
    fread(&numMatrix, 4, 1, fp);
    
    // order of matrices
    int order;
    fread(&order, 4, 1, fp);
    
    struct matrixFile curFile;
    curFile.filename = filenames[fCk];
    curFile.processedMatrixCounter = 0;

    curFile.order = order;
    curFile.nMatrix = numMatrix;


    putFileData (curFile);

    // iterate over each matrix
    int incMCount = 0;
    while(incMCount!=numMatrix){
   


      struct matrixData  curMatrix;
      curMatrix.fileIndex = fCk;
      curMatrix.matrixNumber = incMCount;
      curMatrix.order = order;
      curMatrix.determinant = 0;
      curMatrix.processed = 0;

      curMatrix.matrix = (double *)malloc(order * order * sizeof(double));
  
      fread(curMatrix.matrix, 8, order*order, fp);
     

      putMatrixInFifo (curMatrix);

      incMCount++;
    }
    
  
  }
 
  for (int i = 0; i < N; i++)
  { if (pthread_join (tIdCons[i], (void *) &status_p) != 0)                                       /* thread consumer */
       { perror ("error on waiting for thread customer");
         exit (EXIT_FAILURE);
       }
    printf ("thread consumer, with id %u, has terminated: ", i);
    printf ("its status was %d\n", *status_p);
  }
  
  for (int g=0; g<fnip; g++) {
    struct matrixFile *file = getFileData();
    printf("\nMatrix File  %s\n", file->filename);

    printf("Number of Matrices  %d\n", file->nMatrix);
    printf("Order of the matrices  %d\n", file->order);

    for (int o =0;o<file->nMatrix; o++){
      printf("\tMatrix %d Result: Determinant = %.3f \n", o+1,file->matrixDeterminants[o]);
    }
        
  }

  clock_gettime (CLOCK_MONOTONIC_RAW, &finish);                                /* end of measurement */
  printf ("\nElapsed time = %.6f s\n",  (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);

  exit (EXIT_SUCCESS);

}

static void *worker(void *wid)
{
  unsigned int id = *((unsigned int *)wid); /* worker id */

  while(true){
    
      struct matrixData  *curMatrix = (struct matrixData *)malloc(sizeof(struct matrixData));
      int contin = getSingleMatrixData(id, curMatrix);
      if (contin == -1) {
        signalWaitingConsumers(id);
        break;
      }
      double det = getDeterminant(curMatrix->order,curMatrix->matrix);

      putResults(id,det, curMatrix->fileIndex, curMatrix->matrixNumber);
      curMatrix->processed =1;
      
  }
 

 
  statusWorker[id] = EXIT_SUCCESS;
  pthread_exit (&statusWorker[id]);
  return 0;
}


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

