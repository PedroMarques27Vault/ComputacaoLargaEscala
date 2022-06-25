/**
 *  \file matrix_utils_row.h (interface file)
 *
 *  \brief Problem name: Matrix Determinant Calculation With CUDA.
 *
 *  Utility functions to calculate the determinant of a matrix.
 *
 *  \author Mário Silva, Pedro Marques - June 2022
 */
#ifndef MATRIXUTILSROW_H
# define MATRIXUTILSROW_H

/**
 *  \brief
 *  Calculates the determinant of a given matrix using row reduction.
 * 
 *  \param matrix the matrix to be processed
 *  \param argv order of the matrix
 *  \return the determinant of the matrix
 */
extern double getDeterminant(int order, double *matrix);             


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
extern __global__ void calcDeterminantsRows(double *matricesDevice, double *determinants);

#endif