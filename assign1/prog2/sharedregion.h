/**
 *  \file sharedregion.h (interface file)
 *
 *  \brief Problem name: Matrix Determinant Calculation With Multithreading.
 *  Header file of sharedregion.c
 *
 *  \author Pedro Marques - April 2022
 */
#ifndef SHAREDREGION_H
# define SHAREDREGION_H

/** \brief structure with matrix information */
struct matrixData
{
  unsigned int fileIndex;                                                          /** file where the matrix is from */
  unsigned int matrixNumber;                                                             /** index of matrix in file */
  unsigned int order;                                                                        /** order of the matrix */
  double determinant;                                                                  /** determinant of the matrix */
  double *matrix;                                                                         /** array of matrix values */
};

/** \brief structure with file information */
struct matrixFile
{
  char *filename;                                                                       /** name of the current file */
  double *matrixDeterminants;                                               /** array of determinants of each matrix */
  unsigned int processedMatrixCounter;                                      /** number of matrices already processed */
  unsigned int order;                                                                      /** order of the matrices */
  unsigned int nMatrix;                                                         /** total number of matrices in file */
};


/** \brief retrive single matrix structure from queue */
extern int getSingleMatrixData(unsigned int consId, struct matrixData *val);

/** \brief insert file information */
extern void putFileData (struct matrixFile matrix);

/** \brief retrive one matrixFile object from files array */
extern struct matrixFile * getFileData ();

/** \brief insert matrix in FIFO Queue */
extern void putMatrixInFifo (struct matrixData matrix);

/** \brief insert results in file's determinant array */
extern void putResults(unsigned int consId,double determinant,int fileIndex,int matrixNumber);

#endif