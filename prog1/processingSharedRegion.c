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
 *  It keeps two list of file structures:
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
 *    A worker also publishes results with the function putData:
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

/** \brief producer threads return status array */
extern int *statusWorker;

/** \brief number of files to process */
extern int numFiles;

/** \brief maximum number of bytes per chunk */
extern int maxBytesPerChunk;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief storage region */
struct fileData * filesData;

/** \brief current file index being processed */
static int currFileIndex = 0;

/**
 *  \brief Initialization of the data transfer region.
 *
 *  Internal monitor operation.
 */
void initialData (char *fileNames[])
{
  // allocating memory for numFiles of file structs
  filesData = (struct fileData *)malloc(numFiles * sizeof(struct fileData));
  
  for (int i = 0; i < numFiles; i++)
  {
    (filesData + i)->fileName = fileNames[i];
    (filesData + i)->fp = NULL;
  }
}


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
    struct fileData *fileToProcess = (filesData + currFileIndex);

    if (fileToProcess->fp == NULL)
    {
      fileToProcess->fp = fopen( fileToProcess->fileName, "rb");
      if (fileToProcess->fp == NULL) {
        printf("Error: could not open file %s\n", fileToProcess->fileName);
        exit(EXIT_FAILURE);
      }
    }

    partialData->fileName = fileToProcess->fileName;
    partialData->previousCh = fileToProcess->previousCh;
    partialData->nWords = 0;
    partialData->nWordsBV = 0;
    partialData->nWordsEC = 0;
    partialData->fileIndex = currFileIndex;

    partialData->chunkSize = fread(partialData->chunk, 1, maxBytesPerChunk - 7, fileToProcess->fp);

    getChunkSizeAndLastChar(fileToProcess, partialData);

    if (partialData->chunkSize < maxBytesPerChunk-7 || partialData->previousCh == EOF) {
      currFileIndex++;
      fclose(fileToProcess->fp);
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
