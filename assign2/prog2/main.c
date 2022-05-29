/**
 *  \file main.c 
 *
 *  \brief  Problem name: Matrix Determinant Calculation With MPI.
 * 
 *  The objective is to get the matrices within files and calculate their determinant.
 *
 *  The dispatcher is responsible for reading the matrices from files and sending them to
 *  the worker processes. Aferwards, these workers should retrieve them and calculate the determinant.
 *  Then, these results are sent back to the dispatcher in order to display them
 *
 *
 *  \author Pedro Marques - May 2022
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "probConst.h"
#include "matrixutils.h"
#include <stdbool.h>
#include <libgen.h>
#include <libgen.h>
#include <string.h>
#include <mpi.h>



/** \brief prints explanation of how to run code */
static void printUsage(char *cmdName);

/** \brief indicates if all files have been processed */
# define ALLFILESPROCESSED 0

/** \brief indicates there are still files to be processed */
# define PROCESSINGFILES 1

/**
 *  \brief Main thread.
 *
 *
 *  \param argc number of words of the command line
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */

int main(int argc, char *argv[])
{
  int t  = 20;
  double time = 0;
  double results_array[t];


  char *filenames[16];                                                                                          /* array of file's names  */
  int fnip = 0;                                                                                                 /* filename insertion pointer */
  int opt;  
  
                                                                                        
  int rank, size;

  // MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int fmp = 0; fmp<t;fmp++){
  
  if (size < 2)                                                                                                 /* Requires at least 2 processes */
  {
    fprintf(stderr, "Requires at least two processes.\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  if (rank == 0){                       
    // argument handling
    do  
    {
      switch ((opt = getopt(argc, argv, "f:")))
      {
      case 'f':                                                                                                 /* file name */
        if (optarg[0] == '-')
        {
          fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
          printUsage(basename(argv[0]));
          return EXIT_FAILURE;
        }
        if (fnip>=16)                                                                                           /* at most 16 files */                                    
        {
          fprintf(stderr, "%s: Too many files to unpack. At Most 16\n", basename(argv[0]));
          printUsage(basename(argv[0]));
          return EXIT_FAILURE;
        }

        filenames[fnip++] = optarg;
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
   

   
      struct timespec start, finish;                                                                            /* time limits */

    clock_gettime (CLOCK_MONOTONIC_RAW, &start);                                                                /* begin of time measurement */  
    struct matrixFile * files = (struct matrixFile *)malloc(fnip * sizeof(struct matrixFile));                  /* initialize files array  */
                                              

    for (int fCk = 0;fCk<fnip;fCk++){                                                                           /* process each file in filenames array */

      FILE *fp = fopen(filenames[fCk], "r");

      if (fp == NULL)
      {
          printf("Error: could not open file %s", filenames[fCk]);
          return 1;
      }
      
      int numMatrix;
      fread(&numMatrix, 4, 1, fp);                                                                              /* get number of matrices in the file */
      
      int order; 
      fread(&order, 4, 1, fp);                                                                                  /* get order of the matrices in the file */
      
       
 
      (files+fCk)->filename = filenames[fCk];                                                                   /* save current file's data */
      (files+fCk)->order = order;                                                                               /* save order of the matrices */
      (files+fCk)->nMatrix = numMatrix;                                                                         /* save total number of matrices */
      (files+fCk)->matrixDeterminants = (double *)malloc(numMatrix * sizeof(double));                           /* allocate memory for determinants */

      
      int rest = numMatrix%(size-1);
      int iterations = floor((numMatrix-rest)/(size-1));                                                        /* deal with odd number of workers */
      int incMCount = 0;  
      if (rest>0) iterations+=1;

      for (int iter=0; iter<iterations;iter++){                                                                 /* read and get determinant of each matrix in file */
        int toRead = size;
        if (iter == iterations-1 && rest != 0) toRead = rest+1;                                                 /* deal with odd number of workers */

        for (int nProc = 1; nProc<toRead; nProc++){
          double *matrix = (double *)malloc(order * order * sizeof(double));                                    /* memory allocation of the matrix */
          fread(matrix, 8, order*order, fp);                                                                    /* read full matrix from file */
          
          int WORKSTATUS = PROCESSINGFILES;
          MPI_Send(&WORKSTATUS, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD);                                          /* Send current worker status (PROCESSINGFILES) */
          MPI_Send(&order, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD);                                               /* Send order*/
          MPI_Send(&incMCount, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD);                                           /* Send matrix index*/
          MPI_Send(matrix, order*order, MPI_DOUBLE, nProc, 0, MPI_COMM_WORLD);                                  /* send matrix */
          incMCount++;
         
      }

      for (int nProc = 1; nProc<toRead; nProc++){                                                               /* receive results form all workers */
          int curMatrixNumber;
          double determinant;
          MPI_Recv(&curMatrixNumber, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);                  /* receive the matrix index from the nProc worker */
          MPI_Recv(&determinant, 1, MPI_DOUBLE, nProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);                   /* receive the determinant from the nProc worker*/

          /* update struct with new results */
          (*((((struct matrixFile *)(files+fCk))->matrixDeterminants) + curMatrixNumber)) = determinant;        /* save calculated determinant */

          }
      }
    }


    for (int nProc = 1; nProc<size; nProc++){                                        /* End worker Processes */
      int ws = ALLFILESPROCESSED;
      MPI_Send(&ws, 1, MPI_INT, nProc, 0, MPI_COMM_WORLD); 
    }

  

    clock_gettime (CLOCK_MONOTONIC_RAW, &finish);   
    for (int g=0; g<fnip; g++) {                                                     /* printing results for each file */
      struct matrixFile *file = ((struct matrixFile *)(files+g));                    
      
      printf("\nMatrix File  %s\n", file->filename);
      printf("Number of Matrices  %d\n", file->nMatrix);
      printf("Order of the matrices  %d\n", file->order);

      for (int o =0;o<file->nMatrix; o++){
        printf("\tMatrix %d Result: Determinant = %.3e \n", o+1,file->matrixDeterminants[o]);
      }
        
    }
    double _time =  (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    time+=_time;
    results_array[fmp] = _time;
    printf ("\nElapsed time = %.6f s\n",  _time);

  
   }else{                                                                                 /* Worker Processes, rank!=0 */
    
    while(true){
      int curWorkStatus;
      MPI_Recv(&curWorkStatus, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);      /* receive current worker status  */
      if (curWorkStatus == ALLFILESPROCESSED) {                                           /* finalize if ALLFILESPROCESSED  */
        break;
      }
      

      int order, matrixIndex;
      MPI_Recv(&order, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);              /* receive matrix order */
      double *matrix = (double *)malloc(order * order * sizeof(double));                  /* matrix memory allocation */
      MPI_Recv(&matrixIndex, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);        /* receive matrix index */
      MPI_Recv(matrix, order*order, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); /* receive matrix */
    

      double det = getDeterminant(order,matrix);                                          /* calculate determinant  */
      free(matrix);                                                                       /* free memory used by malloc  */
      MPI_Send(&matrixIndex, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);                           /* send matrix index back to dispatcher  */
      MPI_Send(&det, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);                                /* send matrix determinant to dispatcher  */
    }

  }
  }

  printf("%f\n", time/t);
  double _sum = 0;
  for (int k = 0; k<t;k++){
    _sum+=  ( results_array[k]-(time/t))*( results_array[k]-(time/t));
  }
  printf("Standard Deviation %f\n", sqrt(_sum/t));
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
  fprintf(stderr, "\nSynopsis: %s OPTIONS [filename]\n"
                  "  OPTIONS:\n"
                  "  -h      --- print this help\n"
                  "  -f      --- filename to process\n",
          cmdName);
}

