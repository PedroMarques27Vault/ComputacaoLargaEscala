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


/** \brief Calculates the determinant for each matrix  and returns them in an array */
extern __global__ void calcDeterminants(double *matricesDevice, int *orderDevice, double *determinants);

#endif