/**
 *  \file sharedRegion.h (interface file)
 *
 *  \brief Problem name: Producers / Consumers.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 *
 *  Definition of the operations carried out by the producers / consumers:
 *     \li putVal
 *     \li getVal.
 *
 *  \author Mário Silva - April 2022
 */

#ifndef MONITOR_H
#define MONITOR_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

struct fileData
{
  char *fileName;
  FILE *fp;
  int nWords;
  int nWordsBV;
  int nWordsEC;
  int previousCh;
};

struct filePartialData
{
  char *fileName;
  int fileIndex;
  unsigned char *chunk;
  int chunkSize;
  int nWords;
  int nWordsBV;
  int nWordsEC;
  int previousCh;
};

/**
 *  \brief Store a value in the data transfer region.
 *
 *  Operation carried out by the producers.
 *
 *  \param prodId producer identification
 *  \param val value to be stored
 */
extern void putData(unsigned int workerId, struct filePartialData *data);

/**
 *  \brief Get a value from the data transfer region.
 *
 *  Operation carried out by the consumers.
 *
 *  \param consId consumer identification
 *
 *  \return value
 */

extern void getData(unsigned int workerId, struct filePartialData *partialData);

extern void printResults();

#endif /* MONITOR_H */
