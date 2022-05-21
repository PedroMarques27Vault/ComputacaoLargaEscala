/**
 *  \file probConst.h (interface file)
 *
 *  \brief Problem name: Text Processing with Multithreading.
 *
 *  Problem parameters.
 *
 *  \author Mário Silva - April 2022
 */

#ifndef PROBCONST_H_
#define PROBCONST_H_

/* Generic parameters */

/** \brief maximum number of files */
#define M 10

/** \brief minimum number of bytes each chunk must have */
#define MIN 11

/** \brief default maximum number of bytes each chunk has */
#define DB 2500

/** \brief indicates if all files have been processed */
# define ALLFILESPROCESSED 0

/** \brief indicates there are still files to be processed */
# define FILESINPROCESSING 1

#endif /* PROBCONST_H_ */