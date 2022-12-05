// Futhark Benchmark Wrapper.
// Given a futhark program compiled as a library, run the given entry point.
// make && ./conv2d_dKrn_bench "ad"   > conv2d_dKrn_ad.csv
// make && ./conv2d_dKrn_bench "impl" > conv2d_dKrn_impl.csv
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>
#include <sys/time.h>

#include "../index4.h"

#ifdef OPENCL
#include "conv2d_dKrn_cl.h"
#else
#include "conv2d_dKrn.h"
#endif

#define millisecs_elapsed(s,e) \
  (((e.tv_sec - s.tv_sec) * 1000.0) + ((e.tv_usec - s.tv_usec) / 1000.0))

index4_t sizes_conv2d_dKrn_arrA[] =
 { {   4,   4,  128,  512}
 , {   4,   4,  128,  512}
 , {   4,   4,  128,  512}
 , {   4,   4,  128,  512}
 , {   4,   4,  128,  512}
 , {   4,   4,  128,  512}
 , {   4,   4,  128,  512}
 , {   4,   4,  128,  512}
 , {   4,   4,  128,  512}
 , {   0,   0,    0,    0} };

index4_t sizes_conv2d_dKrn_arrK[] =
 { {   1,   4,    3,    3}
 , {   4,   4,    3,    3}
 , {  16,   4,    3,    3}
 , {  32,   4,    3,    3}
 , {  64,   4,    3,    3}
 , { 128,   4,    3,    3}
 , { 256,   4,    3,    3}
 , { 512,   4,    3,    3}
 , {1024,   4,    3,    3}
 , {   0,   0,    0,    0} };

index4_t sizes_conv2d_dKrn_arrO[] =
 { {   4,   1,  128,  512}
 , {   4,   4,  128,  512}
 , {   4,  16,  128,  512}
 , {   4,  32,  128,  512}
 , {   4,  64,  128,  512}
 , {   4, 128,  128,  512}
 , {   4, 256,  128,  512}
 , {   4, 512,  128,  512}
 , {   4,1024,  128,  512}
 , {   0,   0,    0,    0} };


int main(int argc, char** argv) {
  for (int i = 0; sizes_conv2d_dKrn_arrA[i].img != 0; i++) {
    assert(argc == 2);
    char* entry = argv[1];

    struct futhark_context_config *cfg = futhark_context_config_new();
#ifdef OPENCL
  if (getenv("OPENCL_DEVICE") != NULL)
    futhark_context_config_set_device(cfg, getenv("OPENCL_DEVICE"));
#endif
    struct futhark_context        *ctx = futhark_context_new(cfg);

    index4_t                arrA_size,  arrK_size,  arrO_size;
    float                  *arrA_data, *arrK_data, *arrO_data;
    struct futhark_f32_4d  *arrA,      *arrK,      *arrO;
    struct futhark_f32_1d  *dK;

    struct timeval t_start, t_end;
    int err;

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
    assert(arrA != NULL);

    arrK = futhark_new_f32_4d(
      ctx,
      arrK_data,
      arrK_size.img,
      arrK_size.cha,
      arrK_size.row,
      arrK_size.col);
    assert(arrA != NULL);

    arrO = futhark_new_f32_4d(
      ctx,
      arrO_data,
      arrO_size.img,
      arrO_size.cha,
      arrO_size.row,
      arrO_size.col);
    assert(arrA != NULL);

    gettimeofday(&t_start, NULL);
    if (!strcmp(entry, "ad")) {
      err = futhark_entry_conv2d_dKrn_flat(
        ctx, &dK,
        (const struct futhark_f32_4d*)arrA,
        (const struct futhark_f32_4d*)arrK,
        (const struct futhark_f32_4d*)arrO);
    } else if (!strcmp(entry, "impl")) {
      err = futhark_entry_conv2d_dKrn_impl_flat(
        ctx, &dK,
        arrK_size.row, arrK_size.col,
        (const struct futhark_f32_4d*)arrA,
        (const struct futhark_f32_4d*)arrO);
    }

    if (err) {
      printf("%s\n", futhark_context_get_error(ctx));
    } else {
      gettimeofday(&t_end, NULL);
      printf("%6zu,%6zu,%6zu,%6zu,%6zu,%6zu,%6zu,%6zu,%10.0f\n"
            , arrA_size.img, arrA_size.cha, arrA_size.row, arrA_size.col
            , arrK_size.img, arrK_size.cha, arrK_size.row, arrK_size.col
            , millisecs_elapsed(t_start, t_end));
    }

    futhark_free_f32_4d(ctx, arrA);
    futhark_free_f32_4d(ctx, arrK);
    futhark_free_f32_4d(ctx, arrO);

    free(arrA_data);
    free(arrK_data);
    free(arrO_data);

    futhark_context_free(ctx);
    futhark_context_config_free(cfg);

    assert(err == FUTHARK_SUCCESS);
  }

}