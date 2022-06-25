/**
 *  \file matrix_utils_col.c
 *
 *  \brief Problem name: Matrix Determinant Calculation With CUDA.
 *
 *  Utility functions to calculate the determinant of a matrix.
 *
 *  \author Mário Silva, Pedro Marques - June 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include <time.h>
/**
 *  \brief
 *  Calculates the determinant of a given matrix using column reduction.
 * 
 *  \param matrix the matrix to be processed
 *  \param argv order of the matrix
 *  \return the determinant of the matrix
 */
double getDeterminant(int order, double *matrix)
{
  int i, j, k; 
  int swaps = 0;
  for (i = 0; i < order - 1; i++)
  {
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
          temp = *((matrix + j * order) + i);
          *((matrix + j * order) + i) = *((matrix + j * order) + k);
          *((matrix + j * order) + k) = temp;
        }
      }
    }

    // Begin Gauss Elimination
    for (k = i + 1; k < order; k++)
    {
      double term = *((matrix + i * order) + k) / *((matrix + i * order) + i);
      for (j = i + 1; j < order; j++)
      {
        *((matrix + j * order) + k) -= term * (*((matrix + j * order) + i));
      }
    }
  }
  
 
  double det = 1;
  for (int i = 0; i < order; i++)
  {
    det *= (*((matrix + i * order) + i));
  }

  return pow(-1, swaps) * det;
}

/**
 *  \brief
 *  Calculates the determinant of each matrix in a provided array of matrices
 *  and returns them in an array.
 *
 *  1. The thread corresponding to the current iteration calculates the pivot,
 *  and, multiplies the pivot to the determinant of that matrix.
 *  2. Threads Synchronize.
 *  3. Each thread, that is responsible for a column to the right of the current pivot’s column,
 *  does the Gaussian Elimination on its column only.
 *  4. Threads Synchronize.
 *
 *  \param matricesDevice array of matrices
 *  \param determinants array of determinants for each matrix
 */
__global__ void calcDeterminantsCols(double *matricesDevice, double *determinants)
{
  int order = blockDim.x;
  double *matrix = matricesDevice + blockIdx.x * order * order;
  double pivot;
  bool switchedCols;
  double *col = matrix + threadIdx.x;
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

    // synchronize threads
    __syncthreads();

    // if the thread's column is to the right of the current pivot's column
    if (threadIdx.x > iteration)
    {
      pivotCol = matrix + iteration;
      pivot = *(pivotCol + iteration*order);

      scale = col[iteration*order] / pivot;
      // Begin Gauss Elimination
      for (k = iteration + 1; k < order; k++)
      {
        col[k*order] -= scale * pivotCol[k*order];
      }
    }

    // synchronize threads
    __syncthreads();

    if (threadIdx.x < iteration)
      return;
  }
}