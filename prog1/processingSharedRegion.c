/**
 *  \file monitor.c (implementation file)
 *
 *  \brief
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Shared region implemented as a monitor.
 *
 *  It keeps one array of fileData structures:
 *      file: char[] fileName, FILE fp, int bytesRead, double totalBytes, bool processed, int nWords (etc);
 *  One for storing data associated to reading the file, and the other for storing the results.
 *
 *  Monitored Functions:
 *    A worker can execute the function getData which will:
 *      Get a File from the list of Files to be processed.
 *      Read the last part of the chunk until it reaches a white space, separator or punctuation.
 *      Store the number of bytes read in addition to the chunksize.
 *      Increment by one the the current file index if the file has been fully read.
 *
 *      Return to the worker the fileName, the initial byte to start processing and the size to read.
 *
 *    A worker also publishes results with the function savePartialResults:
 *      Stores results associated to a File.
 *
 *  There is another function, getResults, which will return the results of each file.
 *
 *
 *  \author Mário Silva - April 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>

#include "sharedRegion.h"
#include "textProcFunctions.h"

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
void initialData(char *fileNames[])
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
