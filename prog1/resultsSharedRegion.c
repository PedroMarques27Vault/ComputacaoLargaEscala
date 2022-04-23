/**
 *  \file resultsSharedRegion.c (implementation file)
 *
 *  \brief Monitor for the text processing results.
 *
 *
 *
 *
 *  \author Mário Silva - April 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>

#include "sharedRegion.h"

/** \brief worker threads return status array */
extern int *statusWorker;

/** \brief number of files to process */
extern int numFiles;

/** \brief storage region */
extern struct fileData *filesData;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/**
 *  \brief Store the results of text processing in the data transfer region.
 *
 *  Operation carried out by the workers.
 *
 *  \param workerId worker identification
 *  \param data structure with the results to be stored
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