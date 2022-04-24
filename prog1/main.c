/**
 *  \file main.c
 *
 *  \brief Problem name: Text Processing with Multithreading.
 *
 *  The main objective of this program is to process files in order to obtain
 *  the number of words, and the number of words starting with a vowel and ending in
 *  a consonant.
 *
 *  It is optimized by splitting the work between worker threads which after obtaining
 *  the chunk of the file from the shared region, perform the calculations and then save
 *  the processing results.
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
#include "textProcUtils.h"
#include "probConst.h"

/** \brief worker threads return status array */
int *statusWorker;

/** \brief number of files to process */
int numFiles;

/** \brief maximum number of bytes per chunk */
int maxBytesPerChunk;

static void printUsage(char *cmdName);

/** \brief worker life cycle routine */
static void *worker(void *id);

/**
 *  \brief Main thread.
 *
 *  Design and flow of the main thread:
 *
 *  1 - Process the arguments from the command line.
 *
 *  2 - Initialize the shared region with the necessary structures (by passing the filenames).
 *
 *  3 - Create the worker threads.
 *
 *  4 - Wait for the worker threads to terminate.
 *
 *  5 - Print final results.
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */
int main(int argc, char *argv[])
{
  struct timespec start, finish; /* time limits */

  /* timer starts */
  clock_gettime(CLOCK_MONOTONIC_RAW, &start); /* begin of measurement */

  /* process command line arguments and set up variables */

  int i;                 /* counting variable */
  maxBytesPerChunk = DB; /* maximum number of bytes each worker will process at a time */
  int N = DN;            /* number of worker threads */
  char *fileNames[M];    /* files to be processed (maximum of M) */
  numFiles = 0;          /* number of files to process */
  int opt;               /* selected option */
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
      if (numFiles == M)
      {
        fprintf(stderr, "%s: can only process %d files at a time\n", basename(argv[0]), M);
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
      if (atoi(optarg) < MIN)
      {
        fprintf(stderr, "%s: number of bytes must be greater or equal than %d\n", basename(argv[0]), MIN);
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

  putInitialData(fileNames);

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

  /* timer ends */
  clock_gettime(CLOCK_MONOTONIC_RAW, &finish); /* end of measurement */

  /* print the results of the text processing */
  printResults();

  /* calculate the elapsed time */
  printf("\nElapsed time = %.6f s\n", (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);

  exit(EXIT_SUCCESS);
}

/**
 *  \brief Function worker.
 *
 *  Its role is to perform text processing on chunks of data, after obtaining
 *  the chunk from the shared region, and then save the processing results.
 *
 *  \param wid pointer to application defined worker identification
 */
static void *worker(void *wid)
{
  unsigned int id = *((unsigned int *)wid); /* worker id */

  /* structure that has file's chunk to process and the results of that processing */
  struct filePartialData *partialData = (struct filePartialData *)malloc(sizeof(struct filePartialData));
  partialData->chunk = (unsigned char *)malloc(maxBytesPerChunk * sizeof(unsigned char));

  while (true) /* work until no more data is available */
  {
    getData(id, partialData); /* retrieve data from the shared region to process */

    if (partialData->finished) /* no more data available */
      break;

    processChunk(partialData); /* perform text processing on the chunk */

    savePartialResults(id, partialData); /* save results on the shared region */

    /* reset structures */
    partialData->finished = true;
    partialData->nWords = 0;
    partialData->nWordsBV = 0;
    partialData->nWordsEC = 0;
    memset(partialData->chunk, 0, maxBytesPerChunk * sizeof(unsigned char));
  }

  free(partialData); /* deallocate the structure memory */

  statusWorker[id] = EXIT_SUCCESS;
  pthread_exit(&statusWorker[id]);
  return 0;
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
  fprintf(stderr, "\nSynopsis: %s OPTIONS [filename / number of threads / maximum number of bytes per chunk]\n"
                  "  OPTIONS:\n"
                  "  -h      --- print this help\n"
                  "  -f      --- filename to process\n"
                  "  -n      --- number of threads\n"
                  "  -m      --- maximum number of bytes per chunk\n",
          cmdName);
}