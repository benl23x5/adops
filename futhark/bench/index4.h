#pragma once
#include <stddef.h>

/** Four dimensional array shape and index. */
typedef struct {
  size_t  img;    // image
  size_t  cha;    // channel
  size_t  row;    // row
  size_t  col;    // column
} index4_t;

static inline size_t
extent
  (index4_t sh)
{
  size_t sRow   = sh.col;
  size_t sCha   = sh.row * sRow;
  size_t sImg   = sh.cha * sCha;
  size_t lix    = sImg * sh.img + sCha * sh.cha + sRow * sh.row + sh.col;
  return lix;
}
