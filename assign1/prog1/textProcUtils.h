/**
 *  \file textProcUtils.h (interface file)
 *
 *  \brief Problem name: Text Processing with Multithreading.
 *
 *  Functions used for text processing.
 *
 *  \author Mário Silva - April 2022
 */
#include <stdlib.h>
#include <stdio.h>

#include "sharedRegion.h"

#ifndef TEXT_PROC_Funct_H
#define TEXT_PROC_Funct_H

/**
 *  \brief Checks if the given character is a alpha character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is alpha or not.
 */
int isAlpha(int ch);

/**
 *  \brief Checks if the given character is a vowel.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is a vowel or not.
 */
int isVowel(int ch);

/**
 *  \brief Checks if the given character is a consonant.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is consonant or not.
 */
int isConsonant(int ch);

/**
 *  \brief Checks if the given character is a white space character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is white space or not.
 */
int isWhiteSpace(int ch);

/**
 *  \brief Checks if the given character is a separation character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is separation or not.
 */
int isSeparation(int ch);

/**
 *  \brief Checks if the given character is a punctuation character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is punctuation or not.
 */
int isPunctuation(int ch);

/**
 *  \brief Checks if the given character is a merge character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is merge or not.
 */
int isMergeChar(int ch);

/**
 *  \brief Checks if the given character is a underscore character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is underscore or not.
 */
int isUnderscore(int ch);

/**
 *  \brief Checks if the given character is a numeric character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is numeric or not.
 */
int isNumeric(int ch);

/**
 *  \brief Transforms some special characters to a more general character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return general representation of the given character.
 */
int handleSpecialChars(int ch);

/**
 *  \brief Extracts a character in UTF8 Encoding from a unsigned char buffer.
 *
 *   It also counts the number of bytes read to obtain the character.
 *
 *  \param buffer buffer to read bytes from
 *  \param charUTF8Bytes array that will be filled with the first element
 *  the UTF8 character obtained and the second element the number of bytes read
 */
void extractAChar(unsigned char *buffer, int index, int charUTF8Bytes[2]);

/**
 *  \brief Performs text processing of a chunk.
 *
 *  Counts the number of words, words starting with a vowel and words ending with a consonant.
 *
 *  Needs to know the previous character to see if the previous chunk was inside a word
 *  and also if it was and the word ends with the next character, to see if it was a consonant.
 *
 *  Operation executed by workers.
 *
 *  \param partialData structure that contains the data needed to process
 *  and will be filled with the results obtained
 */
void processChunk(struct filePartialData *partialData);

/**
 *  \brief Reads bytes from the file until it reads a full UTF8 encoded character.
 *
 *  Adds the bytes read to the given buffer.
 *  Updates the chunk size of the given structure.
 *  Obtains the last character of the given chunk as the previous character.
 *  Operation executed by workers.
 *
 *  \param data fileData structure that has the file pointer to read from and will
 *  be updated with the last character of the given chunk.
 *  \param partialData filePartialData structure that contains the chunk and chunk size.
 */
void getChunkSizeAndLastChar(struct fileData *data, struct filePartialData *partialData);

#endif /* TEXT_PROC_Funct_H */