/**
 *  \file main.c
 *
 *  \brief
 *
 *  Initializes the shared region with the structures
 *  (including file names to be processed).
 *
 *  Creates the worker threads.
 *
 *  Wait for the worker threads to terminate.
 *
 *  Print final results.
 *
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  \author Mário Silva - April 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <libgen.h>

#include "sharedRegion.h"
#include "textProcFunctions.h"

/** \brief worker threads return status array */
int *statusWorker;

/** \brief number of files to process */
int numFiles;

/** \brief maximum number of bytes per chunk */
int maxBytesPerChunk;

extern void initialData(char *fileNames[]);

static void printUsage(char *cmdName);

/** \brief worker life cycle routine */
static void *worker(void *id);

/**
 *  \brief Main thread.
 *
 *  Its role is starting the simulation by generating the intervening entities threads (workers) and
 *  waiting for their termination.
 */

int main(int argc, char *argv[])
{
  struct timespec start, finish; /* time limits */

  // timer starts
  clock_gettime(CLOCK_MONOTONIC_RAW, &start); /* begin of measurement */

  int i; /* counting variable */

  // defaults to 600 bytes per worker
  maxBytesPerChunk = 600; /* maximum number of bytes each worker will process at a time */

  // defaults to 8 worker threads
  int N = 8; /* number of worker threads */

  // files to be processed (maximum of 10)
  char *fileNames[10];

  // number of files passed as argument
  numFiles = 0;

  int opt; /* selected option */

  do
  {
    switch ((opt = getopt(argc, argv, "f:n:m:")))
    {
    case 'f': /* file name */
      if (optarg[0] == '-')
      {
        fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
        printUsage(basename(argv[0]));
        return EXIT_FAILURE;
      }
      fileNames[numFiles++] = optarg;
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
    case 'm': /* numeric argument */
      if (atoi(optarg) < 11)
      {
        fprintf(stderr, "%s: number of bytes must be greater or equal than 11\n", basename(argv[0]));
        printUsage(basename(argv[0]));
        return EXIT_FAILURE;
      }
      maxBytesPerChunk = (int)atoi(optarg);
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

  statusWorker = malloc(sizeof(int) * N); /* workers status */
  pthread_t tIdWorker[N];                 /* workers internal thread id array */
  unsigned int workerId[N];               /* workers application defined thread id array */
  int *status_p;                          /* pointer to execution status */

  /* set up structures to be used on the monitor and shared regions */
  initialData(fileNames);

  /* generation of worker threads */
  for (i = 0; i < N; i++)
  {
    workerId[i] = i;

    if (pthread_create(&tIdWorker[i], NULL, worker, &workerId[i]) != 0) /* thread worker */
    {
      perror("error on creating thread worker");
      exit(EXIT_FAILURE);
    }
  }

  /* waiting for the termination of the worker threads */
  for (i = 0; i < N; i++)
  {
    if (pthread_join(tIdWorker[i], (void *)&status_p) != 0)
    {
      perror("error on waiting for worker thread");
      exit(EXIT_FAILURE);
    }
  }
  // timer ends
  clock_gettime(CLOCK_MONOTONIC_RAW, &finish); /* end of measurement */

  printResults();

  // calculate the elapsed time
  printf("\nElapsed time = %.6f s\n", (finish.tv_nsec - start.tv_nsec) / 1000000000.0);

  exit(EXIT_SUCCESS);
}

static void *worker(void *wid)
{
  unsigned int id = *((unsigned int *)wid); /* worker id */

  /* structure that has file's chunk to process and the results of that processing */
  struct filePartialData *partialData = (struct filePartialData *)malloc(sizeof(struct filePartialData));
  partialData->chunk = (unsigned char *)malloc(maxBytesPerChunk * sizeof(unsigned char));

  while (true)
  {
    getData(id, partialData); /* retrieve data to be processed */

    if (partialData->fileName == NULL) /* no more data to be processed */
      break;

    processChunk(partialData);

    putData(id, partialData); /* put processed data */

    // clear up structure
    partialData->fileName = NULL;
    memset(partialData->chunk, 0, maxBytesPerChunk * sizeof(unsigned char));
  }

  free(partialData);

  statusWorker[id] = EXIT_SUCCESS;
  pthread_exit(&statusWorker[id]);
  return 0;
}

static void printUsage(char *cmdName)
{
  fprintf(stderr, "\nSynopsis: %s OPTIONS [filename / number of threads / maximum number of bytes per thread]\n"
                  "  OPTIONS:\n"
                  "  -h      --- print this help\n"
                  "  -f      --- filename to process\n"
                  "  -n      --- number of threads\n"
                  "  -m      --- maximum number of bytes per thread\n",
          cmdName);
}