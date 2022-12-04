// Futhark Benchmark Wrapper.
// Given a futhark program compiled as a library, run the given entry point
// with inputs specified in the companion csv.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>
#include <sys/time.h>

#include "index4.h"
#include "conv2d_dKrn.h"


index4_t sizes_conv2d_dKrn_arrA[] =
 { {   1,   2,  32,   32}
 , {   2,   4,  32,   32}
 , {   4,  16,  32,   32}
 , {  16,  32,  32,   32}
 , {  32,  64,  32,   32}
 , {   0,   0,   0,    0} };

index4_t sizes_conv2d_dKrn_arrK[] =
 { {   4,   2,   3,    3}
 , {  16,   4,   3,    3}
 , {  64,  16,   3,    3}
 , { 128,  32,   3,    3}
 , { 256,  64,   3,    3}
 , {   0,   0,   0,    0} };

index4_t sizes_conv2d_dKrn_arrO[] =
 { {   1,   4,  32,   32}
 , {   2,  16,  32,   32}
 , {   4,  64,  32,   32}
 , {  16, 128,  32,   32}
 , {  32, 256,  32,   32}
 , {   0,   0,   0,    0} };


int main(int argc, char** argv) {
  struct futhark_context_config *cfg =
    futhark_context_config_new();

  struct futhark_context *ctx =
    futhark_context_new(cfg);

  for (int i = 0; sizes_conv2d_dKrn_arrA[i].img != 0; i++) {
    index4_t                     arrA_size,  arrK_size,  arrO_size;
    float                       *arrA_data, *arrK_data, *arrO_data;
    struct futhark_f32_4d  *dK, *arrA,      *arrK,      *arrO;
    struct timeval         t_start, t_end;

    arrA_size   = sizes_conv2d_dKrn_arrA[i];
    arrK_size   = sizes_conv2d_dKrn_arrK[i];
    arrO_size   = sizes_conv2d_dKrn_arrO[i];

    arrA_data   = (float*)(malloc(sizeof(float) * extent(arrA_size)));
    arrK_data   = (float*)(malloc(sizeof(float) * extent(arrK_size)));
    arrO_data   = (float*)(malloc(sizeof(float) * extent(arrO_size)));

    arrA = futhark_new_f32_4d(
      ctx,
      arrA_data,
      arrA_size.img,
      arrA_size.cha,
      arrA_size.row,
      arrA_size.col);

    arrK = futhark_new_f32_4d(
      ctx,
      arrK_data,
      arrK_size.img,
      arrK_size.cha,
      arrK_size.row,
      arrK_size.col);

    arrO = futhark_new_f32_4d(
      ctx,
      arrO_data,
      arrO_size.img,
      arrO_size.cha,
      arrO_size.row,
      arrO_size.col);

    printf("starting futhark bench #%d...\n", i);
    gettimeofday(&t_start, NULL);
    futhark_entry_conv2d_dKrn(
      ctx, &dK,
      (const struct futhark_f32_4d*)arrA,
      (const struct futhark_f32_4d*)arrK,
      (const struct futhark_f32_4d*)arrO);
    gettimeofday(&t_end, NULL);

    float t_diff =
      ((t_end.  tv_sec * 1000000 + t_end.  tv_usec) -
       (t_start.tv_sec * 1000000 + t_start.tv_usec)) / 1000000.0;

    printf("%6zu,%6zu,%6zu,%6zu,%6zu,%6zu,%6zu,%6zu,%10.0f\n"
          , arrA_size.img, arrA_size.cha, arrA_size.row, arrA_size.col
          , arrK_size.img, arrK_size.cha, arrK_size.row, arrK_size.col
          , t_diff);

    futhark_free_f32_4d(ctx, arrA);
    futhark_free_f32_4d(ctx, arrK);
    futhark_free_f32_4d(ctx, arrO);

    free(arrA_data);
    free(arrK_data);
    free(arrO_data);
  }

  futhark_context_free(ctx);
  futhark_context_config_free(cfg);
}