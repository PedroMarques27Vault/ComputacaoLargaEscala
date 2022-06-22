/**
 *  \file matrix_utils.c
 *
 *  \brief Problem name: Matrix Determinant Calculation With CUDA.

 *  Utility functions to calculate the determinant of a matrix
 *
 *  \author Mário Silva, Pedro Marques - June 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>


/**
 *  \brief 
 *  Calculates the determinant of a given matrix
 *  \param matrix the matrix to be processed
 *  \param argv order of the matrix
 *
 *  \return the determinant of the matrix
 */
double getDeterminant(int order, double *matrix){
    int i,j,k;
    int swaps = 0;
    for(i=0;i<order-1;i++){
        //Partial Pivoting
        for(k=i+1;k<order;k++){
            //If diagonal element(absolute vallue) is smaller than any of the terms below it
            if(fabs(*((matrix+i*order) + i))<fabs(*((matrix+k*order) + i))){
                //Swap the rows
                swaps++;
                for(j=0;j<order;j++){                
                    double temp;
                    temp=*((matrix+j*order) + i);
                    *((matrix+j*order) + i)=*((matrix+j*order) + k);
                    *((matrix+j*order) + k)=temp;
                }
            }
        }
        //Begin Gauss Elimination
        for(k=i+1;k<order;k++){
            double  term=*((matrix+i*order) + k)/ *((matrix+i*order) + i);
            for(j=0;j<order;j++){
                *((matrix+j*order) + k)=*((matrix+j*order) + k)-term*(*((matrix+j*order) + i));
            }
        }
    }
    
    double det = 1;
    for(int i=0; i<order; i++){
        det *= (*((matrix+i*order) + i));
    }
	return pow(-1,swaps)*det;;    
}

/**
 *  \brief 
 *  For a given column, calculates the pivot and continuously updates the determinant's value
 *  \param matricesDevice pointer to the matrix in the device
 *  \param orderDevice pointer to the order of the matrix in the device
 *  \param determinants pointer to the array of determinants in the device
 *  \param currentCol current column matrix index whose pivot is being determined
 * 
 */
__global__ void calcPivots(double *matricesDevice, int *orderDevice, double *determinants, int *currentCol)
{
  int iteration = *currentCol;
  if (threadIdx.x == iteration)
  {
    int order = *orderDevice;
    bool switchedCols = false;
    double *matrix = matricesDevice + blockIdx.x * order * order;

    double pivot =  *(matrix + iteration * order + iteration);
    // finding the pivot
    if (pivot == 0.0)
    {
      for (int k = iteration + 1; k < order; k++)
      {
        if ((matrix + iteration*order + k) != 0)
        {
          // Swap the two cols
          for (int j = 0; j < order; j++)
          {
            double temp = *(matrix + iteration * order + iteration);
            *(matrix + j * order + iteration) = *(matrix + j * order + k);
            *(matrix + j * order + k) = temp;
          }
          switchedCols = true;
          break;
        }
      }
    }

    pivot = *(matrix + iteration * order + iteration);

    // calculate the determinants
    if (iteration == 0)
      determinants[blockIdx.x] = pivot;
    else
      determinants[blockIdx.x] *= pivot;

    if (switchedCols)
      determinants[blockIdx.x] *= -1;
  }
}

/**
 *  \brief 
 *  For a given column, subtracts the pivot calculated with calcPivots, executing Gauss Elimination
 *  \param matricesDevice pointer to the matrix in the device
 *  \param orderDevice pointer to the order of the matrix in the device
 *  \param determinants pointer to the array of determinants in the device
 *  \param currentCol current column matrix index to be subtracted the pivot
 * 
 */
__global__ void subtractPivots(double *matricesDevice, int *orderDevice, double *determinants, int *currentCol)
{
  int iteration = *currentCol;
  if (threadIdx.x > iteration) {
    int order = *orderDevice;
    double *matrix = matricesDevice + blockIdx.x * order * order;
    double *col = matrix + threadIdx.x;
    double *pivotCol = matrix + iteration;
    double pivot = *(pivotCol + iteration*order);

    double scale = col[iteration*order] / pivot;
    // Begin Gauss Elimination
    for(int k=iteration+1; k<order; k++)
    {
      col[k*order] -= scale * pivotCol[k*order];
    }
  }
}