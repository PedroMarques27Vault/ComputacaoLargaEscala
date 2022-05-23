/**
 *  \file matrixutils.c
 *
 *  \brief Problem name: Matrix Determinant Calculation With Multithreading.

 *  Utility functions to calculate the determinant of a matrix
 *
 *  \author Pedro Marques - April 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>


/**
 *  \brief 
 *  Calculates the determinant of a given matrix
 *  \param matrix the matrix to be processed
 *  \param argv order of the matrix
 *
 *  \return the determinant of the matrix
 */
double getDeterminant(int order, double *matrix){
    int i,j,k;
    int swaps = 0;
    for(i=0;i<order-1;i++){
        //Partial Pivoting
        for(k=i+1;k<order;k++){
            //If diagonal element(absolute vallue) is smaller than any of the terms below it
            if(fabs(*((matrix+i*order) + i))<fabs(*((matrix+k*order) + i))){
                //Swap the rows
                swaps++;
                for(j=0;j<order;j++){                
                    double temp;
                    temp=*((matrix+i*order) + j);
                    *((matrix+i*order) + j)=*((matrix+k*order) + j);
                    *((matrix+k*order) + j)=temp;
                }
            }
        }
        //Begin Gauss Elimination
        for(k=i+1;k<order;k++){
            double  term=*((matrix+k*order) + i)/ *((matrix+i*order) + i);
            for(j=0;j<order;j++){
                *((matrix+k*order) + j)=*((matrix+k*order) + j)-term*(*((matrix+i*order) + j));
            }
        }
    }
    
    double det = 1;
    for(int i=0; i<order; i++){
        det *= (*((matrix+i*order) + i));
    }
	return pow(-1,swaps)*det;;

             
}


