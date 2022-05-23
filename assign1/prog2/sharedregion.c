
/**
 *  \file sharedregion.c (implementation file)
 *
 *  \brief Problem name: Matrix Determinant Calculation With Multithreading.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 *  Includes 2 main data regions:
 *     \li files is an array of matrixFile structures, containing information about a processed file
 *     \li matrices is the FIFO Queue of matrices to be processed by the workers
 * 
 *  
 *  Definition of the operations carried out by the workers and main thread:
 *  Executed by the main thread:
 *     \li putFileData - Inserts a file info in the shared region
 *     \li putMatrixInFifo - Inserts a Matrix in the FIFO Queue for processing
 *     \li getFileData  - Returns a file's information from the files array
 * Executed by the worker threads:
 *     \li getSingleMatrixData - Retrieved a single matrix from the FIFO Queue
 *     \li putResults - Inserts results of the processed matrix into the correct file in the files array
 *
 *  \author Pedro Marques - April 2022
 */




#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include "sharedregion.h"
#include <math.h>
#include "matrixutils.h"
#include <stdbool.h>
 #include  <time.h>


/** \brief main thread's return status */
int statusProd;

/** \brief worker threads return status array */
extern int *statusWorker;

/** \brief matrix to process storage region - FIFO Queue */
static struct matrixData *matrices;


/** \brief storage region of all files */
static struct matrixFile * files;

/** \brief insertion pointer in FIFO Queue*/
static unsigned int ii;

/** \brief retrieval pointer */
static unsigned int ri;

/** \brief file insertion pointer */
static unsigned int fip;

/** \brief file retrieval pointer */
static unsigned int frp;

/** \brief counter of processed files */
static unsigned int fCounter;

/** \brief total number of files */
static unsigned int totalFileCount;


/** \brief dimension of matrices array */
static unsigned int K;


/** \brief flag signaling the data transfer region is full */
static bool full;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief main process file insertion producers synchronization point when the data transfer region is full */
static pthread_cond_t fifoFull;

/** \brief workers synchronization point when the data transfer region is empty */
static pthread_cond_t fifoEmpty;


/**
 *  \brief 
 *
 *  Initialization of the shared region variables
 *  Memory allocation for the FIFO Queue and files array
 *  
 *  \param _totalFileCount total number of files to be processed
 *  \param _K size of the FIFO Queue
 *
 */
void initialization(int _totalFileCount, int _K)
{
  K = _K;
  totalFileCount = _totalFileCount;                                               

  matrices = malloc(sizeof(struct matrixData) * _K);                              /* initialize FIFO/matrices Queue  */
  files = (struct matrixFile *)malloc(_totalFileCount * sizeof(struct matrixFile));       /* initialize files array  */

  

                                                                                 
  ii = ri = 0;                                       /* FIFO insertion and retrieval pointers  set to the same value */
  fip = 0;                                                                     /* File insertion pointer initialized */
  fCounter = 0;                                                             /* initialize number of processed files  */

  full = false;                                                                                  /* FIFO is not full */

  pthread_cond_init (&fifoFull, NULL);                                 /* initialize producers synchronization point */
  pthread_cond_init (&fifoEmpty, NULL);                                /* initialize consumers synchronization point */
}


/**
 *  \brief 
 *
 *  Insert a file's data info in the files array
 *  Executed by the main thread
 *  \param file file's data matrixFile structure
 *
 */
void putFileData (struct matrixFile file)
{

  if ((statusProd = pthread_mutex_lock (&accessCR)) != 0)                                           /* enter monitor */
     { errno = statusProd;                                                                    /* save error in errno */
       perror ("error on entering monitor(CF)");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }
  
  /* saving the file's data into the files array */
  (files+fip)->filename = file.filename;
  (files+fip)->processedMatrixCounter = file.processedMatrixCounter;
  (files+fip)->order = file.order;
  (files+fip)->nMatrix = file.nMatrix;
  (files+fip)->matrixDeterminants = (double *)malloc(file.nMatrix * sizeof(double));

  fip++;                                                                         /* increment file insertion pointer */

  if ((statusProd = pthread_mutex_unlock (&accessCR)) != 0)                                          /* exit monitor */
     { errno = statusProd;                                                                    /* save error in errno */
       perror ("error on exiting monitor(CF)");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }
}

/**
 *  \brief 
 *
 *  Retrieve file's data info from the shared region
 *  Executed by the main thread
 *  \return file's data info, matrixFile structure
 *
 */
struct matrixFile * getFileData ()
{
  struct matrixFile * toRetrieve;

  toRetrieve = (files+frp);                                                /* retrieve file at frp position in files */

  frp = (frp + 1) % totalFileCount;                                         /* increase file retrieval pointer value */
 
  return toRetrieve;
}

/**
 *  \brief 
 *  
 *  Insert matrix into the FIFO Queue matrices to be processed
 *  If the Queue is full, wait until space is available to insert.
 *  Afterwards, signal a consumer about the existance of a new matrix in the queue
 *  Executed by the main thread
 *  \param matrix to be added, matrixData structure
 *
 */
void putMatrixInFifo (struct matrixData matrix)
{
  if ((statusProd = pthread_mutex_lock (&accessCR)) != 0)                                           /* enter monitor */
     { errno = statusProd;                                                                    /* save error in errno */
       perror ("error on entering monitor(CF)");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }                                         

  while (full)                                                    /* wait if the data transfer region (FIFO) is full */
  { if ((statusProd = pthread_cond_wait (&fifoFull, &accessCR)) != 0)
       { errno = statusProd;                                                                  /* save error in errno */
         perror ("error on waiting in fifoFull");
         statusProd = EXIT_FAILURE;
         pthread_exit (&statusProd);
       }
  }

  matrices[ii] = matrix;                                                             /* insert matrix at ii position */
  ii = (ii + 1) % K; 
  full = (ii == ri);

  if ((statusProd = pthread_cond_signal (&fifoEmpty)) != 0)        /* let a worker know that a value has been stored */
     { errno = statusProd;                                                                    /* save error in errno */
       perror ("error on signaling in fifoEmpty");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }

  if ((statusProd = pthread_mutex_unlock (&accessCR)) != 0)                                          /* exit monitor */
     { errno = statusProd;                                                                    /* save error in errno */
       perror ("error on exiting monitor(CF)");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }
}


/**
 *  \brief 
 *
 *  Retrieve matrix info from the FIFO Queue matrices to be processed
 *  If the Queue is empty, wait until a matrix is inserted.
 *  Afterwards, signal the main  about the existance of a new matrix in the queue
 *  
 *  \param val matrixData structure to be filled with the retrieved matrix's info
 *  \param consId worker thread's id
 *  \return if there are still files to be processed returns 0, otherwise returns -1
 *
 */
int getSingleMatrixData(unsigned int consId, struct matrixData *val)
{
  int toReturn = -1;                                                                         /* value to be returned */
  if ((statusWorker[consId] = pthread_mutex_lock (&accessCR)) != 0)                                 /* enter monitor */
     { errno = statusWorker[consId];                                                          /* save error in errno */
       perror ("error on entering monitor(CF)");
       statusWorker[consId] = EXIT_FAILURE;
       pthread_exit (&statusWorker[consId]);
     }                                         

  while ((ii == ri) && !full && (fCounter!=totalFileCount))         /* wait if the data transfer region is empty and not 
                                                                                       all files have been processed */
    { 
      if ((statusWorker[consId] = pthread_cond_wait (&fifoEmpty, &accessCR)) != 0)
        { errno = statusWorker[consId];                                                       /* save error in errno */
          perror ("error on waiting in fifoEmpty");
          statusWorker[consId] = EXIT_FAILURE;
          pthread_exit (&statusWorker[consId]);
        }
    }


  if (fCounter != totalFileCount){                       /* retrieve matrix if not all files have been processed yet */
      val->fileIndex = matrices[ri].fileIndex;
      val->matrixNumber = matrices[ri].matrixNumber;
      val->order = matrices[ri].order;
      val->determinant = matrices[ri].determinant;
      val->matrix = matrices[ri].matrix;
      
    
      ri = (ri + 1) % K;

      /* if all matrixes have been processed, increment processed file counter */
      if (((struct matrixFile *)(files+val->fileIndex))->processedMatrixCounter == ((struct matrixFile *)(files+val->fileIndex))->nMatrix-1){
        fCounter++;
      }

      ((struct matrixFile *)(files+val->fileIndex))->processedMatrixCounter++;                  /* increment counter of 
                                                                                                  processed matrices */
      full = false;                                                                       /* queue is no longer full */
      
      if ((statusWorker[consId] = pthread_cond_signal (&fifoFull)) != 0)             /* let the main thread know that a
                                                                                            value has been retrieved */
        { errno = statusWorker[consId];                                                       /* save error in errno */
          perror ("error on signaling in fifoFull");
          statusWorker[consId] = EXIT_FAILURE;
          pthread_exit (&statusWorker[consId]);
        }
      toReturn = 0;                                                                   
  }

  if ((statusWorker[consId] = pthread_mutex_unlock (&accessCR)) != 0)                                /* exit monitor */
     { errno = statusWorker[consId];                                                          /* save error in errno */
       perror ("error on exiting monitor(CF)");
       statusWorker[consId] = EXIT_FAILURE;
       pthread_exit (&statusWorker[consId]);
     }

  return toReturn;
}

/**
 *  \brief 
 *
 *  Insert processed matrix's results into the file's determinants array
 *  If all files have been processed and no matrices are left to be processed, signal all waiting workers 
 *  to end lifecycle.

 *  \param consId worker thread's id
 *  \param determinant determinant of the processed matrix
 *  \param fileIndex index of processed matrix's file in the files array
 *  \param matrixNumber index of the matrix in its file
 *
 */
void putResults(unsigned int consId,double determinant,int fileIndex,int matrixNumber)
{
  if ((statusWorker[consId] = pthread_mutex_lock (&accessCR)) != 0)                                 /* enter monitor */
     { errno = statusWorker[consId];                                                          /* save error in errno */
       perror ("error on entering monitor(CF)");
       statusWorker[consId] = EXIT_FAILURE;
       pthread_exit (&statusWorker[consId]);
     }     
  (*((((struct matrixFile *)(files+fileIndex))
    ->matrixDeterminants) + matrixNumber)) = determinant;        /* add determinant in the file's determinants array */
  
  if (fCounter == totalFileCount){ 
    pthread_cond_broadcast(&fifoEmpty);                                               /* signal all  waiting workers */
  }

  if ((statusWorker[consId] = pthread_mutex_unlock (&accessCR)) != 0)                                /* exit monitor */
  { errno = statusWorker[consId];                                                             /* save error in errno */
    perror ("error on exiting monitor(CF)");
    statusWorker[consId] = EXIT_FAILURE;
    pthread_exit (&statusWorker[consId]);
  }
}

