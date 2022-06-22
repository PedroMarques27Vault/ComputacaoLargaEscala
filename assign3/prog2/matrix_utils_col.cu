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
double getDeterminant(int order, double *matrix)
{
  int i, j, k;
  int swaps = 0;
  for (i = 0; i < order - 1; i++)
  {
    // Partial Pivoting
    for (k = i + 1; k < order; k++)
    {
      // If diagonal element(absolute vallue) is smaller than any of the terms to the right of it
      if (fabs(*((matrix + i * order) + i)) < fabs(*((matrix + i * order) + k)))
      {
        // Swap the cols
        swaps++;
        for (j = 0; j < order; j++)
        {
          double temp;
          temp = *((matrix + i * order) + j);
          *((matrix + i * order) + j) = *((matrix + j * order) + k);
          *((matrix + j * order) + k) = temp;
        }
      }
    }
    // Begin Gauss Elimination
    for (k = i + 1; k < order; k++)
    {
      double term = *((matrix + i * order) + k) / *((matrix + i * order) + i);
      for (j = 0; j < order; j++)
      {
        *((matrix + j * order) + k) = *((matrix + j * order) + k) - term * (*((matrix + j * order) + i));
      }
    }
  }
  double det = 1;
  for (int i = 0; i < order; i++)
  {
    det *= (*((matrix + i * order) + i));
  }
  return pow(-1, swaps) * det;
  ;
}



/**
 *  \brief
 *  Calculates the determinant of each matrix in a provided array of matrix  and returns them in an array. 
 *  Each thread calculates the pivot for each column. The threads synchronize and then Gaussian Elimination
 *  begins by subtracted the pivot to each column
 *  \param matricesDevice array of matrices
 *  \param orderDevice order of the matrices
 *  \param determinants array of determinants for each matrix
 */
__global__ void calcDeterminants(double *matricesDevice, int *orderDevice, double *determinants)
{
  int order = *orderDevice;
  double *matrix = matricesDevice + blockIdx.x * order * order;
  double pivot;
  bool switchedCols;
  double *col = matrix + threadIdx.x * order;
  double *pivotCol;
  double scale;

  int iteration, k, j, temp;
  for (iteration = 0; iteration < order; iteration++)
  {
    if (threadIdx.x == iteration)
    {
      switchedCols = false;

      pivot = *(matrix + iteration * order + iteration);
      // finding the pivot
      if (pivot == 0.0)
      {
        for (k = iteration + 1; k < order; k++)
        {
          if ((matrix + k * order + iteration) != 0)
          {
            // Swap the two rows
            for (j = 0; j < order; j++)
            {
              temp = *(matrix + j * order + k);
              *(matrix + j * order + k) = *(matrix + j * order + iteration);
              *(matrix + j * order + iteration) = temp;
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

    __syncthreads();

    if (threadIdx.x > iteration)
    {
      pivotCol = matrix + iteration * order;
      pivot = *(pivotCol + iteration);

      scale = col[iteration] / pivot;
      // Begin Gauss Elimination
      for (k = iteration + 1; k < order; k++)
      {
        col[k] -= scale * pivotCol[k];
      }
    }

    __syncthreads();
  }
}