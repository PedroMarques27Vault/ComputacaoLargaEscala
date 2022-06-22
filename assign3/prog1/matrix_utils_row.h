/**
 *  \file matrix_utils.h (interface file)
 *
 *  \brief Problem name: Matrix Determinant Calculation With Multithreading.
 *  Header file of matrixutils.c
 *
 *  \author Mário Silva, Pedro Marques - June 2022
 */
#ifndef MATRIXUTILSROW_H
# define MATRIXUTILSROW_H

/** \brief get the determinant of given matrix */
extern double getDeterminant(int order, double *matrix);             

/** \brief For a given row, calculates the pivot and continuously updates the determinant's value */
extern __global__ void calcPivots(double *matricesDevice, int *orderDevice, double *determinants, int *currentRow);

/** \brief For a given row, subtracts the pivot calculated with calcPivots, executing Gauss Elimination */
extern __global__ void subtractPivots(double *matricesDevice, int *orderDevice, double *determinants, int *currentRow);
#endif