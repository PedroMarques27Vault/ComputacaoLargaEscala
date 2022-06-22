/**
 *  \file matrix_utils.h (interface file)
 *
 *  \brief Problem name: Matrix Determinant Calculation With Multithreading.
 *  Header file of matrixutils.c
 *
 *  \author Mário Silva, Pedro Marques - June 2022
 */
#ifndef MATRIXUTILSCOL_H
# define MATRIXUTILSCOL_H

/** \brief get the determinant of given matrix */
extern double getDeterminant(int order, double *matrix);             

/**
 *  \brief For a given column, calculates the pivot and continuously updates the determinant's value
 */
extern __global__ void calcPivots(double *matricesDevice, int *orderDevice, double *determinants, int *currentCol);

/**
 *  \brief For a given column, subtracts the pivot calculated with calcPivots, executing Gauss Elimination
 */
extern __global__ void subtractPivots(double *matricesDevice, int *orderDevice, double *determinants, int *currentCol);
#endif