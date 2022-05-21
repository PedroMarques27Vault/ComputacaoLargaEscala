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

#include "textProcUtils.h"
#include "probConst.h"

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

  if (rank == 0)
  {
    struct timespec start, finish; /* time limits */

    /* timer starts */
    clock_gettime(CLOCK_MONOTONIC_RAW, &start); /* begin of measurement */

    /** \brief maximum number of bytes per chunk */
    int maxBytesPerChunk = DB;

    /* process command line arguments and set up variables */
    int i;              /* counting variable */
    char *fileNames[M]; /* files to be processed (maximum of M) */
    int numFiles = 0;   /* number of files to process */
    int opt;            /* selected option */
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

    // allocating memory for numFiles of file structs
    struct fileData *filesData = (struct fileData *)malloc(numFiles * sizeof(struct fileData));

    for (int i = 0; i < numFiles; i++)
    {
      (filesData + i)->fileName = fileNames[i];
      FILE *fp = fopen(fileNames[i], "rb"); /* get the file pointer */
      if (fp == NULL)
      {
        printf("Error: could not open file %s\n", fileNames[i]);
        exit(EXIT_FAILURE);
      }

      
      int fileProcessed = 0;
      // while file is processing
      while (!fileProcessed)
      {
        int nProcesses = 0; // number of processes that got chunks

        // Send a chunk of data to each worker process for processing
        for (int nProc = 1; nProc < size; nProc++)
        {
          if (fileProcessed)
            break;

          nProcesses++;

          fileProcessed = getChunk(file, buff, chunkSize);

          /*Warn workers that the work is not over and give them the buffer*/
          exitWork = FILESINPROCESSING;
          MPI_Send(&exitWork, 1, MPI_UNSIGNED, size-1, 0, MPI_COMM_WORLD);
          MPI_Send(buff, chunkSize, MPI_CHAR, size-1, 0, MPI_COMM_WORLD);
        }

        /*
        Receive data From each process
        Assemble the partial data received with the data that was stored in the final info
        */
        for (int nProc = 1; nProc < nProcesses + 1; nProc++)
        { // for all processes
          int rows;

          /*Receive number of words and number of rows*/
          MPI_Recv(&nWords, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          /*Sum total number of words*/
          finalInfo[i].nwords += nWords;

          for (int col = 0; col < row + 1; col++)
          { // col represents number of consonants (if row = 2, col will get the value of 0, 1 and 2)
            finalInfo[i].data[row][col] += data[col];
          }
        }
      }
    }

    //TODO: BROADCAST BETTER MAYBE....
    /* inform workers that all files are process and they can exit */
    for (int nProc = 1 ; nProc < size ; nProc++)
      MPI_Send (&ALLFILESPROCESSED, 1, MPI_UNSIGNED, nProc, 0, MPI_COMM_WORLD);

    /* timer ends */
    clock_gettime(CLOCK_MONOTONIC_RAW, &finish); /* end of measurement */

    /* print the results of the text processing */
    printResults();

    /* calculate the elapsed time */
    printf("\nElapsed time = %.6f s\n", (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);
  }
  else
  {
    char buff[1050];
    struct PartialInfo partialInfo;
    int exitWork;

    while (true) {

        MPI_Recv (&exitWork, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);     
        if (exitWork == NOMOREWORK) 
            break;

        MPI_Recv(buff,chunkSize,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);    //receive buffer

        int i = 0;
        char ch;
        int consonants = 0;
        int inword = 0;
        int numchars = 0;
        
        partialInfo.data = (int**)malloc(sizeof(int*));
        partialInfo.nwords = 0;

        processDataString(buff);

        /* Send the processing results to the dispatcher */
        MPI_Send (&partialInfo.nwords, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);    //send number of words

        free(partialInfo.data);
    }
  }

  MPI_Finalize ();
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