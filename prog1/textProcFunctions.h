#include "sharedRegion.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef TEXT_PROC_Funct_H
#define TEXT_PROC_Funct_H

int isAlpha(int ch);

int isVowel(int ch);

int isConsonant(int ch);

int isWhiteSpace(int ch);

int isSeparation(int ch);

int isPunctuation(int ch);

int isMergeChar(int ch);

int isUnderscore(int ch);

int isNumeric(int ch);

int handleSpecialChars(int ch);

void extractAChar(FILE *fp, int *charUTF8Bytes);

void processChunk(struct filePartialData *partialData);

void getChunkSizeAndLastChar(struct fileData *data, struct filePartialData *partialData);

#endif /* TEXT_PROC_Funct_H */