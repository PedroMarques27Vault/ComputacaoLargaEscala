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
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <libgen.h>
#include <mpi.h>

#include <pthread.h>

#include "sharedRegion.h"
#include "textProcUtils.h"
#include "probConst.h"

/**
 *  \brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  \param cmdName string with the name of the command
 */
static void printUsage(char *cmdName);

/**
 *  \brief Print results of the text processing.
 *
 *  Operation carried out by the dispatcher process.
 */
void printResults(struct fileData *filesData, int numFiles);

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
    int nFile, nProc;   /* counting variables */
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

    /* allocating memory for numFiles of fileData structs */
    struct fileData *filesData = (struct fileData *)malloc(numFiles * sizeof(struct fileData));
    int nProcesses = 0; /* number of processes that got chunks */
    int previousCh = 0; /* last character read of the previous chunk */

    for (nFile = 0; nFile < numFiles; nFile++)
    {
      (filesData + nFile)->fileName = fileNames[nFile];
      /* get the file pointer */
      if (((filesData + nFile)->fp = fopen(fileNames[nFile], "rb")) == NULL)
      {
        printf("Error: could not open file %s\n", fileNames[nFile]);
        exit(EXIT_FAILURE);
      }

      /* allocating memory for the chunk buffer */
      /* TODO: usar apenas um chunk buffer talvez ou libertar o anterior dps... */
      (filesData + nFile)->chunk = (unsigned char *)malloc(maxBytesPerChunk * sizeof(unsigned char));

      /* while file is processing */
      while (!((filesData + nFile)->finished))
      {
        nProcesses = 0; // number of processes that got chunks

        /* Send a chunk of data to each worker process for processing */
        for (nProc = 1; nProc < size; nProc++)
        {
          if ((filesData + nFile)->finished) {
            fclose((filesData + nFile)->fp); /* close the file pointer */
            break;
          }

          previousCh = (filesData + nFile)->previousCh;

          (filesData + nFile)->chunkSize = fread((filesData + nFile)->chunk, 1, maxBytesPerChunk - 7, (filesData + nFile)->fp);
          /*
            if the chunk read is smaller than the value expected
            it means the current file has reached the end
          */
          if ((filesData + nFile)->chunkSize < maxBytesPerChunk - 7)
            (filesData + nFile)->finished = true;
          /*
            - reads bytes from the file until it reads a full UTF8 encoded character
            - adds the bytes read to the chunk
            - updates the chunk size
            - updates the last character of this chunk as the previous character
          */
          else
            getChunkSizeAndLastChar(filesData + nFile);

          if ((filesData + nFile)->previousCh == EOF) /* checks the last character was the EOF */
            (filesData + nFile)->finished = true;

          /*
            send to the worker:
            - a flag saying there work to do
            - the chunk buffer
            - the size of the chunk
            - the character of the previous chunk
          */
          int workStatus = FILESINPROCESSING;
          MPI_Send(&workStatus, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD);
          MPI_Send((filesData + nFile)->chunk, maxBytesPerChunk, MPI_UNSIGNED_CHAR, nProc, 0, MPI_COMM_WORLD);
          MPI_Send(&(filesData + nFile)->chunkSize, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD);
          MPI_Send(&previousCh, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD);
          memset((filesData + nFile)->chunk, 0, maxBytesPerChunk * sizeof(unsigned char));
          nProcesses++;
        }

        /*
          Receive the processing results from each worker process
        */
        for (nProc = 1; nProc < nProcesses + 1; nProc++)
        {
          /*Receive number of words and number of rows*/
          // TODO: change this to an array of results
          int nWords = 0;
          int nWordsBV = 0;
          int nWordsEC = 0;
          MPI_Recv(&nWords, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&nWordsBV, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&nWordsEC, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          /* update struct with new results */
          (filesData + nFile)->nWords += nWords;
          (filesData + nFile)->nWordsBV += nWordsBV;
          (filesData + nFile)->nWordsEC += nWordsEC;
        }
      }
    }
    // TODO: BROADCAST BETTER MAYBE....
    /* inform workers that all files are process and they can exit */
    int workStatus = ALLFILESPROCESSED;
    for (int nProc = 1; nProc < size; nProc++)
      MPI_Send(&workStatus, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD);

    /* timer ends */
    clock_gettime(CLOCK_MONOTONIC_RAW, &finish); /* end of measurement */

    /* print the results of the text processing */
    printResults(filesData, numFiles);

    /* calculate the elapsed time */
    printf("\nElapsed time = %.6f s\n", (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);
  }
  else
  {
    int maxBytesPerChunk = DB;
    /* allocating memory for the file data structure */
    struct fileData* data = (struct fileData *) malloc(sizeof(struct fileData));
    /* allocating memory for the chunk buffer */
    data->chunk = (unsigned char *)malloc(maxBytesPerChunk * sizeof(unsigned char));
    int workStatus;

    while (true)
    {
      MPI_Recv(&workStatus, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (workStatus == ALLFILESPROCESSED)
        break;
      
      MPI_Recv(data->chunk, maxBytesPerChunk, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive buffer
      MPI_Recv(&data->chunkSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive buffer
      MPI_Recv(&data->previousCh, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive buffer
      
      /* perform text processing on the chunk */
      processChunk(data);
      /* Send the processing results to the dispatcher */
      MPI_Send(&data->nWords, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&data->nWordsBV, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&data->nWordsEC, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      /* reset structures */
      data->nWords = 0;
      data->nWordsBV = 0;
      data->nWordsEC = 0;
    }
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);
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

/**
 *  \brief Print results of the text processing.
 *
 *  Operation carried out by the dispatcher process.
 */
void printResults(struct fileData *filesData, int numFiles)
{
  for (int i = 0; i < numFiles; i++)
  {
    printf("\nFile name: %s\n", (filesData + i)->fileName);
    printf("Total number of words = %d\n", (filesData + i)->nWords);
    printf("N. of words beginning with a vowel = %d\n", (filesData + i)->nWordsBV);
    printf("N. of words ending with a consonant = %d\n", (filesData + i)->nWordsEC);
  }
}