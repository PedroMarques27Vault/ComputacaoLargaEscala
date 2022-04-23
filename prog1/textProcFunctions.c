#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "sharedRegion.h"

int isAlpha(int ch)
{
  return ((65 <= ch && ch <= 90) || (97 <= ch && ch <= 122));
}

int isVowel(int ch)
{
  return ((ch == 65) || (ch == 69) || (ch == 73) || (ch == 79) ||
          (ch == 85) || (ch == 97) || (ch == 101) || (ch == 105) ||
          (ch == 111) || (ch == 117));
}

// checks is the char is alpha and not a vowel
int isConsonant(int ch)
{
  return (isAlpha(ch) && !isVowel(ch));
}

int isWhiteSpace(int ch)
{
  return ((ch == 9) || (ch == 10) || (ch == 13) || (ch == 32));
}

int isSeparation(int ch)
{
  return ((ch == 34) || (ch == 40) || (ch == 41) || (ch == 45) ||
          (ch == 91) || (ch == 93) || (ch == 171) || (ch == 187) ||
          (ch == 8211) || (ch == 8220) || (ch == 8221));
}

int isPunctuation(int ch)
{
  return ((ch == 33) || (ch == 44) || (ch == 46) || (ch == 58) ||
          (ch == 59) || (ch == 63) || (ch == 8212) || (ch == 8230));
}

int isMergeChar(int ch)
{
  return ((ch == 39) || (ch == 8216) || (ch == 8217));
}

int isUnderscore(int ch)
{
  return (ch == 95);
}

int isNumeric(int ch)
{
  return (48 <= ch && ch <= 57);
}

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

void extractAChar(FILE *fp, int charUTF8Bytes[2])
{
  int ch = fgetc(fp);

  // if its the EOF or not a multi byte sequence
  if (ch == EOF || !(ch & 0x80))
  {
    charUTF8Bytes[0] = ch;
    charUTF8Bytes[1] = 1;
    return;
  }

  int seq_len = 1;
  int c;
  int fn = ch;

  // find out the number of bytes to read
  for (; ch & (0x80 >> seq_len); seq_len++)
  {
    if ((c = fgetc(fp)) == EOF)
    {
      charUTF8Bytes[0] = handleSpecialChars(ch);
      charUTF8Bytes[1] = seq_len;
      return;
    }
    // shift to add 6 zeros on the right of the final char
    // and use the 6 most representative bits of the read char
    fn = (fn << 6) | (c & 0x3F);
  }
  // add the initial bits after the sequence length identifier bits
  ch = fn & ((1 << ((7 - seq_len) + 6 * (seq_len - 1))) - 1);

  charUTF8Bytes[0] = handleSpecialChars(ch);
  charUTF8Bytes[1] = seq_len;
}

void extractACharFromBuffer(unsigned char *buffer, int index, int charUTF8Bytes[2])
{
  int ch = buffer[index++];

  // if its the EOF or not a multi byte sequence
  if (ch == EOF || !(ch & 0x80))
  {
    charUTF8Bytes[0] = ch;
    charUTF8Bytes[1] = 1;
    return;
  }

  int seq_len = 1;
  int fn = ch;

  // find out the number of bytes to read
  while (ch & (0x80 >> seq_len))
  {
    // shift to add 6 zeros on the right of the final char
    // and use the 6 most representative bits of the read char
    fn = (fn << 6) | (buffer[index++] & 0x3F);
    seq_len++;
  }
  // add the initial bits after the sequence length identifier bits
  ch = fn & ((1 << ((7 - seq_len) + 6 * (seq_len - 1))) - 1);

  charUTF8Bytes[0] = handleSpecialChars(ch);
  charUTF8Bytes[1] = seq_len;
}

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
    extractACharFromBuffer(partialData->chunk, numChars, charUTF8Bytes);
    ch = charUTF8Bytes[0];

    numChars += charUTF8Bytes[1];

    // if its not processing any word
    // and the char is alfa, numeric or a underscore
    if (!inWord &&
        (isAlpha(ch) || isNumeric(ch) || isUnderscore(ch)))
    {
      // check if the start of the word is a vowel
      if (isVowel(ch))
        nWordsBV++;
      // processing a new word
      inWord = true;
      // update previous read char
      previousCh = ch;
      // increase one word
      nWords++;
    }
    // if its processing a word
    else if (inWord)
    {
      // if its a char that merges words
      // update the previous read char value
      // and continue reading next chars
      if (isMergeChar(ch) || isAlpha(ch) ||
          isNumeric(ch) || isUnderscore(ch))
        previousCh = ch;
      // else if its a white space, separation, punctuation char
      // or the end of the file
      // it means the current word has ended
      else if (isWhiteSpace(ch) ||
               isSeparation(ch) ||
               isPunctuation(ch) ||
               ch == EOF)
      {
        // checking if the last previous read char value is a consonant
        if (isConsonant(previousCh))
          nWordsEC++;

        // end processing the word
        inWord = false;
      }
    }
  }

  partialData->nWords = nWords;
  partialData->nWordsBV = nWordsBV;
  partialData->nWordsEC = nWordsEC;
}



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
