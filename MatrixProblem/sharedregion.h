
#ifndef SHAREDREGION_H
# define SHAREDREGION_H

struct matrixData
{
  unsigned int fileIndex;
  unsigned int matrixNumber;
  unsigned int order;
  double determinant;
  double *matrix;
  int processed;
};

struct matrixFile
{
  char *filename;
  double *matrixDeterminants;
  unsigned int processedMatrixCounter;
  unsigned int order;
  unsigned int nMatrix;
};

extern int getSingleMatrixData(unsigned int consId, struct matrixData *val);

extern void putFileData (struct matrixFile matrix);
extern struct matrixFile * getFileData ();
extern void putMatrixInFifo (struct matrixData matrix);
extern int signalWaitingConsumers(unsigned int consId);
extern void putResults(unsigned int consId,double determinant,int fileIndex,int matrixNumber);

#endif