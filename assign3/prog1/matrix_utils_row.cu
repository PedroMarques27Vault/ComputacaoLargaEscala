/**
 *  \file matrix_utils_row.c
 *
 *  \brief Problem name: Matrix Determinant Calculation With CUDA using row reduction.
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

/**
 *  \brief
 *  Calculates the determinant of a given matrix using row reduction.
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
    // Partial Pivoting
    for (k = i + 1; k < order; k++)
    {
      // If diagonal element(absolute vallue) is smaller than any of the terms below it
      if (fabs(*((matrix + i * order) + i)) < fabs(*((matrix + k * order) + i)))
      {
        // Swap the rows
        swaps++;
        for (j = 0; j < order; j++)
        {
          double temp;
          temp = *((matrix + i * order) + j);
          *((matrix + i * order) + j) = *((matrix + k * order) + j);
          *((matrix + k * order) + j) = temp;
        }
      }
    }
    // Begin Gauss Elimination
    for (k = i + 1; k < order; k++)
    {
      double term = *((matrix + k * order) + i) / *((matrix + i * order) + i);
      for (j = i + 1; j < order; j++)
      {
        *((matrix + k * order) + j) = *((matrix + k * order) + j) - term * (*((matrix + i * order) + j));
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
 *  Calculates the determinant of each matrix in a provided array of matrix  and returns them in an array. 
 *
 *  1. The thread corresponding to the current iteration calculates the pivot,
 *  and, multiplies the pivot to the determinant of that matrix.
 *  2. Threads Synchronize.
 *  3. Each thread, that is responsible for a row below of the current pivot’s row,
 *  does the Gaussian Elimination on its row only.
 *  4. Threads Synchronize.
 *
 *  \param matricesDevice array of matrices
 *  \param determinants array of determinants for each matrix
 */
__global__ void calcDeterminantsRows(double *matricesDevice, double *determinants)
{
  int order = blockDim.x;
  double *matrix = matricesDevice + blockIdx.x * order * order;
  double pivot;
  bool switchedRows;
  double *row = matrix + threadIdx.x * order;
  double *pivotRow;
  double scale;
  int iteration, k, j, temp;
  for (iteration = 0; iteration < order; iteration++)
  {
    if (threadIdx.x == iteration)
    {
      switchedRows = false;

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
              temp = *(matrix + k * order + j);
              *(matrix + k * order + j) = *(matrix + iteration * order + j);
              *(matrix + iteration * order + j) = temp;
            }
            switchedRows = true;
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

      if (switchedRows)
        determinants[blockIdx.x] *= -1;
    }

    // synchronize threads
    __syncthreads();

    // if the thread's row is bellow the current pivot's row
    if (threadIdx.x > iteration)
    {
      pivotRow = matrix + iteration * order;
      pivot = *(pivotRow + iteration);

      scale = row[iteration] / pivot;
      // Begin Gauss Elimination
      for (k = iteration + 1; k < order; k++)
      {
        row[k] -= scale * pivotRow[k];
      }
    }

    // synchronize threads
    __syncthreads();
    if (threadIdx.x < iteration)
      return;
  }
}