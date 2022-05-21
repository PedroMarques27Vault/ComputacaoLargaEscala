/**
 *  \file processingSharedRegion.c (implementation file)
 *
 *  \brief Shared Region for the text processing with multithreading problem.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 *
 *  This shared region will utilize the array of structures initialized by
 *  the main thread.
 * 
 *  Workers can access the shared region to obtain data to process from that
 *  array of structures. They can also store the partial results of the
 *  processing done.
 * 
 *  There is also a function to print out the final results, that should be
 *  used after there is no more data to be processed.
 * 
 *  Monitored Methods:
 *     \li getData - operation carried out by worker threads.
 *     \li savePartialResults - operation carried out by worker threads.
 *
 *  Unmonitored Methods:
 *     \li putInitialData - operation carried out by the main thread.
 *     \li printResults - operation carried out by the main thread.
 *
 *  \author Mário Silva - April 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>

#include "sharedRegion.h"
#include "textProcUtils.h"

/** \brief worker threads return status array */
extern int *statusWorker;

/** \brief number of files to process */
extern int numFiles;

/** \brief maximum number of bytes per chunk */
extern int maxBytesPerChunk;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief storage region */
struct fileData *filesData;

/** \brief current file index being processed */
static int currFileIndex = 0;

/**
 *  \brief Initialization of the data transfer region.
 *
 *  Allocates the memory for an array of structures with the files passed
 *  as argument and initializes it with their names.
 *
 *  \param fileNames contains the names of the files to be stored
 */
void putInitialData(char *fileNames[])
{
  // allocating memory for numFiles of file structs
  filesData = (struct fileData *)malloc(numFiles * sizeof(struct fileData));

  for (int i = 0; i < numFiles; i++)
  {
    (filesData + i)->fileName = fileNames[i];
    (filesData + i)->fp = NULL;
  }
}

/**
 *  \brief Get data to process from the data transfer region.
 *
 *  Operation carried out by the workers.
 *
 *  \param workerId worker identification
 *  \param partialData structure that will store the chunk of chars to process
 */
void getData(unsigned int workerId, struct filePartialData *partialData)
{
  if ((statusWorker[workerId] = pthread_mutex_lock(&accessCR)) != 0) /* enter monitor */
  {
    errno = statusWorker[workerId]; /* save error in errno */
    perror("error on entering monitor(CF)");
    statusWorker[workerId] = EXIT_FAILURE;
    pthread_exit(&statusWorker[workerId]);
  }

  if (numFiles != currFileIndex) /* if files have not all been processed yet */
  {

    /* obtain the current file to process */

    struct fileData *fileToProcess = (filesData + currFileIndex);

    if (fileToProcess->fp == NULL) /* check if the file pointer has been initialized */
    {
      fileToProcess->fp = fopen(fileToProcess->fileName, "rb"); /* store the file pointer */
      if (fileToProcess->fp == NULL)
      {
        printf("Error: could not open file %s\n", fileToProcess->fileName);
        exit(EXIT_FAILURE);
      }
    }

    /* obtain partial data to be sent to the worker */

    partialData->finished = false;                       /* data is available to process */
    partialData->fileIndex = currFileIndex;              /* file index on the shared region array structure */
    partialData->previousCh = fileToProcess->previousCh; /* last character of the previous chunk */

    /*
      stores in a buffer the a chunk with {maxBytesPerChunk-7} bytes
      also obtains the size of the chunk that was read from the file
    */
    partialData->chunkSize = fread(partialData->chunk, 1, maxBytesPerChunk - 7, fileToProcess->fp);

    /*
      if the chunk read is smaller than the value expected
      it means the current file has reached the end
    */
    if (partialData->chunkSize < maxBytesPerChunk - 7)
    {
      currFileIndex++;           /* update the current file being processed index */
      fclose(fileToProcess->fp); /* close the file pointer */
    }
    else
    {

      /*
        - reads bytes from the file until it reads a full UTF8 encoded character
        - adds the bytes read to the chunk of partialData
        - updates the chunk size
        - stores the last character of this chunk as the previous character
        of the file to process
      */
      getChunkSizeAndLastChar(fileToProcess, partialData);

      if (partialData->previousCh == EOF) /* checks the last character was the EOF */
      {
        currFileIndex++;           /* update the current file being processed index */
        fclose(fileToProcess->fp); /* close the file pointer */
      }
    }
  }

  if ((statusWorker[workerId] = pthread_mutex_unlock(&accessCR)) != 0) /* exit monitor */
  {
    errno = statusWorker[workerId]; /* save error in errno */
    perror("error on exiting monitor(CF)");
    statusWorker[workerId] = EXIT_FAILURE;
    pthread_exit(&statusWorker[workerId]);
  }
}

/**
 *  \brief Store the results of text processing in the data transfer region.
 *
 *  Operation carried out by the workers.
 *
 *  \param workerId worker identification
 *  \param partialData structure with the results to be stored
 */
void savePartialResults(unsigned int workerId, struct filePartialData *partialData)
{
  if ((statusWorker[workerId] = pthread_mutex_lock(&accessCR)) != 0) /* enter monitor */
  {
    errno = statusWorker[workerId]; /* save error in errno */
    perror("error on entering monitor(CF)");
    statusWorker[workerId] = EXIT_FAILURE;
    pthread_exit(&statusWorker[workerId]);
  }

  /*
    add partial results to the final results
    on the correct structure from the shared region
  */
  (filesData + partialData->fileIndex)->nWords += partialData->nWords;
  (filesData + partialData->fileIndex)->nWordsBV += partialData->nWordsBV;
  (filesData + partialData->fileIndex)->nWordsEC += partialData->nWordsEC;

  if ((statusWorker[workerId] = pthread_mutex_unlock(&accessCR)) != 0) /* exit monitor */
  {
    errno = statusWorker[workerId]; /* save error in errno */
    perror("error on exiting monitor(CF)");
    statusWorker[workerId] = EXIT_FAILURE;
    pthread_exit(&statusWorker[workerId]);
  }
}

/**
 *  \brief Print results of the text processing.
 *
 *  Operation carried out by the main thread.
 */
void printResults()
{
  for (int i = 0; i < numFiles; i++)
  {
    printf("\nFile name: %s\n", (filesData + i)->fileName);
    printf("Total number of words = %d\n", (filesData + i)->nWords);
    printf("N. of words beginning with a vowel = %d\n", (filesData + i)->nWordsBV);
    printf("N. of words ending with a consonant = %d\n", (filesData + i)->nWordsEC);
  }
}