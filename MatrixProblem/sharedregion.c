
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include "sharedregion.h"
#include <math.h>
#include "matrixutils.h"
#include <stdbool.h>
 #include  <time.h>


int statusProd;

/** \brief consumer threads return status array */
extern int *statusWorker;

/** \brief storage region */
static struct matrixData *matrices;


static struct matrixFile * files;

/** \brief insertion pointer */
static unsigned int ii;


/** \brief file insertion pointer */
static unsigned int fip;

/** \brief file retrieval pointer */
static unsigned int frp;

/** \brief total number of files */
static unsigned int fCounter;

static unsigned int totalFileCount;

static unsigned int K;

/** \brief retrieval pointer */
static unsigned int ri;

/** \brief flag signaling the data transfer region is full */
static bool full;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;


/** \brief producers synchronization point when the data transfer region is full */
static pthread_cond_t fifoFull;

/** \brief consumers synchronization point when the data transfer region is empty */
static pthread_cond_t fifoEmpty;


void initialization(int _totalFileCount, int _K)
{
  K = _K;
  totalFileCount = _totalFileCount;
  matrices = malloc(sizeof(struct matrixData) * _K);
  files = (struct matrixFile *)malloc(_totalFileCount * sizeof(struct matrixFile));

  

                                                                                   /* initialize FIFO in empty state */
  ii = ri = fip = 0;                                        /* FIFO insertion and retrieval pointers set to the same value */
  fCounter = 0;

  full = false;                                                                                  /* FIFO is not full */

  pthread_cond_init (&fifoFull, NULL);                                 /* initialize producers synchronization point */
  pthread_cond_init (&fifoEmpty, NULL);                                /* initialize consumers synchronization point */
}


void putFileData (struct matrixFile matrix)
{

  if ((statusProd = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
     { errno = statusProd;                                                            /* save error in errno */
       perror ("error on entering monitor(CF)");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }
     /* internal data initialization */
    (files+fip)->filename = matrix.filename;
    (files+fip)->processedMatrixCounter = matrix.processedMatrixCounter;
    (files+fip)->order = matrix.order;
    (files+fip)->nMatrix = matrix.nMatrix;
    (files+fip)->matrixDeterminants = (double *)malloc(matrix.nMatrix * sizeof(double));


  fip++;

  
  if ((statusProd = pthread_mutex_unlock (&accessCR)) != 0)                                  /* exit monitor */
     { errno = statusProd;                                                            /* save error in errno */
       perror ("error on exiting monitor(CF)");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }
}

struct matrixFile * getFileData ()
{
  struct matrixFile * toRetrieve;

    toRetrieve = (files+frp);

    frp = (frp + 1) % totalFileCount;
    
  
 
  return toRetrieve;
}

void putMatrixInFifo (struct matrixData matrix)
{
  if ((statusProd = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
     { errno = statusProd;                                                            /* save error in errno */
       perror ("error on entering monitor(CF)");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }                                         

  while (full)                                                           /* wait if the data transfer region is full */
  { if ((statusProd = pthread_cond_wait (&fifoFull, &accessCR)) != 0)
       { errno = statusProd;                                                          /* save error in errno */
         perror ("error on waiting in fifoFull");
         statusProd = EXIT_FAILURE;
         pthread_exit (&statusProd);
       }
  }

  matrices[ii] = matrix;                                                                          /* store value in the FIFO */
  ii = (ii + 1) % K;
  full = (ii == ri);

  if ((statusProd = pthread_cond_signal (&fifoEmpty)) != 0)      /* let a consumer know that a value has been
                                                                                                               stored */
     { errno = statusProd;                                                             /* save error in errno */
       perror ("error on signaling in fifoEmpty");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }

  if ((statusProd = pthread_mutex_unlock (&accessCR)) != 0)                                  /* exit monitor */
     { errno = statusProd;                                                            /* save error in errno */
       perror ("error on exiting monitor(CF)");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }
}


int areFilesAvailable(unsigned int consId)
{                                                                        /* retrieved value */
  int val = 1;
  if ((statusWorker[consId] = pthread_mutex_lock (&accessCR)) != 0)     {                              /* enter monitor */
     { errno = statusWorker[consId];                                                            /* save error in errno */
       perror ("error on entering monitor(CF)");
       statusWorker[consId] = EXIT_FAILURE;
       pthread_exit (&statusWorker[consId]);
     }                                          /* internal data initialization */}


  if (fCounter == totalFileCount){ 
    val = 0;
    pthread_cond_broadcast(&fifoEmpty);
  }

  if ((statusWorker[consId] = pthread_mutex_unlock (&accessCR)) != 0)                                   /* exit monitor */
     { errno = statusWorker[consId];                                                             /* save error in errno */
       perror ("error on exiting monitor(CF)");
       statusWorker[consId] = EXIT_FAILURE;
       pthread_exit (&statusWorker[consId]);
     }

  return val;
}

struct matrixData getSingleMatrixData(unsigned int consId)
{
  struct matrixData val;                                                                           /* retrieved value */

  if ((statusWorker[consId] = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
     { errno = statusWorker[consId];                                                            /* save error in errno */
       perror ("error on entering monitor(CF)");
       statusWorker[consId] = EXIT_FAILURE;
       pthread_exit (&statusWorker[consId]);
     }                                          /* internal data initialization */

  while ((ii == ri) && !full && (fCounter!=totalFileCount))                                           /* wait if the data transfer region is empty */
    { 
      if ((statusWorker[consId] = pthread_cond_wait (&fifoEmpty, &accessCR)) != 0)
        { errno = statusWorker[consId];                                                          /* save error in errno */
          perror ("error on waiting in fifoEmpty");
          statusWorker[consId] = EXIT_FAILURE;
          pthread_exit (&statusWorker[consId]);
        }
    }

    val = matrices[ri];          
    ri = (ri + 1) % K;

    if (((struct matrixFile *)(files+val.fileIndex))->processedMatrixCounter == ((struct matrixFile *)(files+val.fileIndex))->nMatrix-1){
      fCounter++;
    }

    ((struct matrixFile *)(files+val.fileIndex))->processedMatrixCounter++;
    full = false;
    if ((statusWorker[consId] = pthread_cond_signal (&fifoFull)) != 0)       /* let a producer know that a value has been
                                                                                                              retrieved */
      { errno = statusWorker[consId];                                                             /* save error in errno */
        perror ("error on signaling in fifoFull");
        statusWorker[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorker[consId]);
      }
  
  
  
    

  if ((statusWorker[consId] = pthread_mutex_unlock (&accessCR)) != 0)                                   /* exit monitor */
     { errno = statusWorker[consId];                                                             /* save error in errno */
       perror ("error on exiting monitor(CF)");
       statusWorker[consId] = EXIT_FAILURE;
       pthread_exit (&statusWorker[consId]);
     }

  return val;
}


void putResults(unsigned int consId,double determinant,int fileIndex,int matrixNumber)
{
    (*((((struct matrixFile *)(files+fileIndex))->matrixDeterminants) + matrixNumber)) = determinant; 
}

