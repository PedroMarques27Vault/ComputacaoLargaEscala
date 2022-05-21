/**
 *  \file textProcUtils.h (interface file)
 *
 *  \brief Problem name: Text Processing with Multithreading.
 *
 *  Functions used for text processing.
 *
 *  \author Mário Silva - April 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "sharedRegion.h"

/**
 *  \brief Checks if the given character is a alpha character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is alpha or not.
 */
int isAlpha(int ch)
{
  return ((65 <= ch && ch <= 90) || (97 <= ch && ch <= 122));
}

/**
 *  \brief Checks if the given character is a vowel.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is a vowel or not.
 */
int isVowel(int ch)
{
  return ((ch == 65) || (ch == 69) || (ch == 73) || (ch == 79) ||
          (ch == 85) || (ch == 97) || (ch == 101) || (ch == 105) ||
          (ch == 111) || (ch == 117));
}

/**
 *  \brief Checks if the given character is a consonant.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is consonant or not.
 */
int isConsonant(int ch)
{
  return (isAlpha(ch) && !isVowel(ch));
}

/**
 *  \brief Checks if the given character is a white space character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is white space or not.
 */
int isWhiteSpace(int ch)
{
  return ((ch == 9) || (ch == 10) || (ch == 13) || (ch == 32));
}

/**
 *  \brief Checks if the given character is a separation character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is separation or not.
 */
int isSeparation(int ch)
{
  return ((ch == 34) || (ch == 40) || (ch == 41) || (ch == 45) ||
          (ch == 91) || (ch == 93) || (ch == 171) || (ch == 187) ||
          (ch == 8211) || (ch == 8220) || (ch == 8221));
}

/**
 *  \brief Checks if the given character is a punctuation character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is punctuation or not.
 */
int isPunctuation(int ch)
{
  return ((ch == 33) || (ch == 44) || (ch == 46) || (ch == 58) ||
          (ch == 59) || (ch == 63) || (ch == 8212) || (ch == 8230));
}

/**
 *  \brief Checks if the given character is a merge character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is merge or not.
 */
int isMergeChar(int ch)
{
  return ((ch == 39) || (ch == 8216) || (ch == 8217));
}

/**
 *  \brief Checks if the given character is a underscore character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is underscore or not.
 */
int isUnderscore(int ch)
{
  return (ch == 95);
}

/**
 *  \brief Checks if the given character is a numeric character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return character is numeric or not.
 */
int isNumeric(int ch)
{
  return (48 <= ch && ch <= 57);
}

/**
 *  \brief Transforms some special characters to a more general character.
 *
 *  \param ch UTF8 encoded character
 *
 *  \return general representation of the given character.
 */
int handleSpecialChars(int ch)
{
  if ((192 <= ch && ch <= 196) || (224 <= ch && ch <= 228))
    return 97;
  if ((200 <= ch && ch <= 203) || (232 <= ch && ch <= 235))
    return 101;
  if ((204 <= ch && ch <= 207) || (236 <= ch && ch <= 239))
    return 105;
  if ((210 <= ch && ch <= 214) || (242 <= ch && ch <= 246))
    return 111;
  if ((217 <= ch && ch <= 220) || (249 <= ch && ch <= 252))
    return 117;
  if (ch == 199 || ch == 231)
    return 99;
  return ch;
}

/**
 *  \brief Extracts a character in UTF8 Encoding from a unsigned char buffer.
 *
 *   It also counts the number of bytes read to obtain the character.
 *
 *  \param buffer buffer to read bytes from
 *  \param charUTF8Bytes array that will be filled with the first element
 *  the UTF8 character obtained and the second element the number of bytes read
 */
void extractAChar(unsigned char *buffer, int index, int charUTF8Bytes[2])
{
  int ch = buffer[index++];

  /* if its the EOF or not a multi byte sequence */
  if (ch == EOF || !(ch & 0x80))
  {
    charUTF8Bytes[0] = ch;
    charUTF8Bytes[1] = 1;
    return;
  }

  int seq_len = 1;
  int fn = ch;

  /* find out the number of bytes to read */
  while (ch & (0x80 >> seq_len))
  {
    /*
      shift to add 6 zeros on the right of the final char
      and use the 6 most representative bits of the read char
    */
    fn = (fn << 6) | (buffer[index++] & 0x3F);
    seq_len++;
  }
  /* add the initial bits after the sequence length identifier bits */
  ch = fn & ((1 << ((7 - seq_len) + 6 * (seq_len - 1))) - 1);

  /* update the array with the results */
  charUTF8Bytes[0] = handleSpecialChars(ch);
  charUTF8Bytes[1] = seq_len;
}

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
void processChunk(struct filePartialData *partialData)
{
  // final results variables
  int nWords = 0;
  int nWordsBV = 0;
  int nWordsEC = 0;

  int previousCh = partialData->previousCh;

  // current read character
  int ch = 0;
  // indicates when the chars read belong to a word
  bool inWord;
  // check if the previous char is a char that belongs to words
  inWord = (isMergeChar(previousCh) || isAlpha(previousCh) ||
            isNumeric(previousCh) || isUnderscore(previousCh));

  int charUTF8Bytes[2];
  int numChars = 0;

  while (numChars <= partialData->chunkSize)
  {
    /* extract a UTF8 encoded character from the buffer */
    extractAChar(partialData->chunk, numChars, charUTF8Bytes);

    ch = charUTF8Bytes[0];

    numChars += charUTF8Bytes[1];

    /*
      if its not processing any word
      and the char is alfa, numeric or a underscore
    */
    if (!inWord &&
        (isAlpha(ch) || isNumeric(ch) || isUnderscore(ch)))
    {
      /* check if the start of the word is a vowel */
      if (isVowel(ch))
        nWordsBV++;
      /* processing a new word */
      inWord = true;
      /* update previous read char */
      previousCh = ch;
      /* increase one word */
      nWords++;
    }
    /* if its processing a word */
    else if (inWord)
    {
      /*
        if its a char that merges words
        update the previous read char value
        and continue reading next chars
      */
      if (isMergeChar(ch) || isAlpha(ch) ||
          isNumeric(ch) || isUnderscore(ch))
        previousCh = ch;
      /*
        else if its a white space, separation, punctuation char
        or the end of the file
        it means the current word has ended
      */
      else if (isWhiteSpace(ch) ||
               isSeparation(ch) ||
               isPunctuation(ch) ||
               ch == EOF)
      {
        /* checking if the last previous read char value is a consonant */
        if (isConsonant(previousCh))
          nWordsEC++;

        /* end processing the word */
        inWord = false;
      }
    }
  }

  /* update the structure with the results */
  partialData->nWords = nWords;
  partialData->nWordsBV = nWordsBV;
  partialData->nWordsEC = nWordsEC;
}

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
void getChunkSizeAndLastChar(struct fileData *data, struct filePartialData *partialData)
{
  /* read the next unsigned char from the file and add it to partial data */
  int ch = fgetc(data->fp);
  (partialData->chunk)[partialData->chunkSize++] = ch;

  /*
    while in middle of multi byte sequence read a unsigned char from the file
    add to the byte read to the chunk of the partial data
    update chunk size of the partial data
  */
  while ((ch & 0xC0) == 0x80)
  {
    ch = fgetc(data->fp);
    (partialData->chunk)[partialData->chunkSize++] = ch;
  }

  /* if its the EOF or not a multi byte sequence */
  if (ch == EOF || !(ch & 0x80))
  {
    data->previousCh = ch;
    (partialData->chunk)[partialData->chunkSize++] = ch;
    return;
  }

  int seq_len = 1;
  int c;
  int fn = ch;

  /* find out the number of bytes to read */

  while (ch & (0x80 >> seq_len))
  {
    c = fgetc(data->fp);
    (partialData->chunk)[partialData->chunkSize++] = c;
    /*
      shift to add 6 zeros on the right of the final char
      and use the 6 most representative bits of the read char
    */
    fn = (fn << 6) | ((partialData->chunk)[partialData->chunkSize++] & 0x3F);
    seq_len++;
  }
  /* add the initial bits after the sequence length identifier bits */
  ch = fn & ((1 << ((7 - seq_len) + 6 * (seq_len - 1))) - 1);

  /* update the previous character of the file to process as the character found */
  data->previousCh = handleSpecialChars(ch);
}
