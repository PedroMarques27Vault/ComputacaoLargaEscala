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

// TODO:
// ver se nao era igual ao prog1_2 se fizer fread e dps usar um buffer
// ter dois ficheiros com shared regions diferentes (get e put)
// mudar o nome de char para outra cena tipo utf8 encoding
// apagar ficheiros desnecessarios e organizar as funcoes e imports e assim
// meter comentarios e tal

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>

#include "sharedRegion.h"

/** \brief producer threads return status array */
extern int *statusWorker;

/** \brief number of files to process */
extern int numFiles;

/** \brief storage region */
extern struct fileData *filesData;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessPR = PTHREAD_MUTEX_INITIALIZER;

// save partial data of processed file
void putData(unsigned int workerId, struct filePartialData *partialData)
{
  if ((statusWorker[workerId] = pthread_mutex_lock(&accessPR)) != 0) /* enter monitor */
  {
    errno = statusWorker[workerId]; /* save error in errno */
    perror("error on entering monitor(CF)");
    statusWorker[workerId] = EXIT_FAILURE;
    pthread_exit(&statusWorker[workerId]);
  }

  (filesData + partialData->fileIndex)->nWords += partialData->nWords;
  (filesData + partialData->fileIndex)->nWordsBV += partialData->nWordsBV;
  (filesData + partialData->fileIndex)->nWordsEC += partialData->nWordsEC;

  if ((statusWorker[workerId] = pthread_mutex_unlock(&accessPR)) != 0) /* exit monitor */
  {
    errno = statusWorker[workerId]; /* save error in errno */
    perror("error on exiting monitor(CF)");
    statusWorker[workerId] = EXIT_FAILURE;
    pthread_exit(&statusWorker[workerId]);
  }
}

// function to print the processing results of the files
void printResults()
{
  for (int i = 0; i < numFiles; i++)
  {
    printf("File name = %s\n", (filesData + i)->fileName);
    printf("Total number of words = %d\n", (filesData + i)->nWords);
    printf("N. of words beginning with a vowel = %d\n", (filesData + i)->nWordsBV);
    printf("N. of words ending with a consonant = %d\n\n", (filesData + i)->nWordsEC);
  }
}