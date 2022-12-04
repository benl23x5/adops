
open import "/prelude/math"

def softmax [n] (elts: [n]f32): [n]f32
 = let mx  = f32.maximum elts
   let den = f32.sum (map (\x -> f32.exp (x - mx)) elts)
   in  map (\x -> f32.exp (x - mx) / den) elts

def regression
    [nImgs][nChas][nRows][nCols]
    (arr: [nImgs][nChas][nRows][nCols]f32)
 :  [nImgs][nRows][nCols]f32
 = let aKs = tabulate nChas (\iCha -> f32.i64 iCha)
   in  tabulate_3d nImgs nRows nCols
        (\iImg iRow iCol ->
          let aNeg = tabulate nChas (\iCha -> 0 - arr[iImg, iCha, iRow, iCol])
          in  f32.sum (map2 (\xK xV -> xK * xV) aKs (softmax aNeg)))


def main (arrA: [1][8][16][32]f32): [1][16][32]f32
 = regression arrA


-- $ futhark dev --type-check --inline-aggressively --ad --fuse-soacs --inline-aggressively regression.fut > dump/regression.txt
-- ------------------------------------------------------------------------------------------------
-- entry("main",
--       {arrA: [][][][]f32},
--       {[][][]f32})
--   entry_main (arrA_7738 : [1i64][8i64][16i64][32i64]f32)
--   : {[1i64][16i64][32i64]f32} = {
--   let {iota_res_8960 : [8i64]i64} =
--     iota64(8i64, 0i64, 1i64)
--   let {defunc_1_map_res_9089 : [8i64]f32} =
--     map(8i64,
--         {iota_res_8960},
--         \ {x_8965 : i64}
--           : {f32} ->
--           let {i64_res_8966 : f32} = sitofp i64 x_8965 to f32
--           in {i64_res_8966})
--   let {iota_res_8977 : [16i64]i64} =
--     iota64(16i64, 0i64, 1i64)
--   let {iota_res_8981 : [32i64]i64} =
--     iota64(32i64, 0i64, 1i64)
--   let {defunc_2_map_res_9090 : [16i64][32i64]f32} =
--     map(16i64,
--         {iota_res_8977},
--         \ {x_8989 : i64}
--           : {[32i64]f32} ->
--           let {x_8990 : bool} = sle64(0i64, x_8989)
--           let {y_8991 : bool} = slt64(x_8989, 16i64)
--           let {bounds_check_8992 : bool} = logand(x_8990, y_8991)
--           let {defunc_1_map_res_9088 : [32i64]f32} =
--             map(32i64,
--                 {iota_res_8981},
--                 \ {x_8994 : i64}
--                   : {f32} ->
--                   let {x_8995 : bool} = sle64(0i64, x_8994)
--                   let {y_8996 : bool} = slt64(x_8994, 32i64)
--                   let {bounds_check_8997 : bool} = logand(x_8995, y_8996)
--                   let {y_8999 : bool} = logand(bounds_check_8992, bounds_check_8997)
--                   let {defunc_2_reduce_res_9084 : f32,
--                        defunc_1_map_res_9085 : [8i64]f32} =
--                     redomap(8i64,
--                             {iota_res_8960},
--                             {\ {x_9010 : f32, x_9011 : f32}
--                               : {f32} ->
--                               let {defunc_1_op_res_9012 : f32} = fmax32(x_9010, x_9011)
--                               in {defunc_1_op_res_9012},
--                             {-f32.inf}},
--                             \ {x_9066 : i64}
--                               : {f32,
--                                  f32} ->
--                               let {x_9067 : bool} = sle64(0i64, x_9066)
--                               let {y_9068 : bool} = slt64(x_9066, 8i64)
--                               let {bounds_check_9069 : bool} = logand(x_9067, y_9068)
--                               let {index_ok_9070 : bool} = logand(y_8999, bounds_check_9069)
--                               let {index_certs_9071 : unit} =
--                                 assert(index_ok_9070, {"Index [", 0i64 : i64, ", ", x_9066 : i64, ", ", x_8989 : i64, ", ", x_8994 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 8i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "regression.fut:16:51-77")
--                               let {arg_9072 : f32} =
--                                 #{index_certs_9071}
--                                 arrA_7738[0i64, x_9066, x_8989, x_8994]
--                               let {defunc_0_f_res_9073 : f32} = fsub32(0.0f32, arg_9072)
--                               in {defunc_0_f_res_9073, defunc_0_f_res_9073})
--                   let {defunc_2_reduce_res_9086 : f32} =
--                     redomap(8i64,
--                             {defunc_1_map_res_9085},
--                             {\ {x_9019 : f32, x_9020 : f32}
--                               : {f32} ->
--                               let {defunc_1_op_res_9021 : f32} = fadd32(x_9019, x_9020)
--                               in {defunc_1_op_res_9021},
--                             {0.0f32}},
--                             \ {x_9061 : f32}
--                               : {f32} ->
--                               let {exp_arg_9062 : f32} = fsub32(x_9061, defunc_2_reduce_res_9084)
--                               let {exp_res_9063 : f32} =
--                                 apply exp32(exp_arg_9062)
--                                 : {f32}
--                               in {exp_res_9063})
--                   let {defunc_2_reduce_res_9087 : f32} =
--                     redomap(8i64,
--                             {defunc_1_map_res_9085, defunc_1_map_res_9089},
--                             {\ {x_9033 : f32, x_9034 : f32}
--                               : {f32} ->
--                               let {defunc_1_op_res_9035 : f32} = fadd32(x_9033, x_9034)
--                               in {defunc_1_op_res_9035},
--                             {0.0f32}},
--                             \ {x_9052 : f32, x_9053 : f32}
--                               : {f32} ->
--                               let {exp_arg_9054 : f32} = fsub32(x_9052, defunc_2_reduce_res_9084)
--                               let {exp_res_9055 : f32} =
--                                 apply exp32(exp_arg_9054)
--                                 : {f32}
--                               let {defunc_0_f_res_9056 : f32} = fdiv32(exp_res_9055, defunc_2_reduce_res_9086)
--                               let {defunc_1_f_res_9058 : f32} = fmul32(x_9053, defunc_0_f_res_9056)
--                               in {defunc_1_f_res_9058})
--                   in {defunc_2_reduce_res_9087})
--           in {defunc_1_map_res_9088})
--   let {defunc_3_map_res_8983 : [1i64][16i64][32i64]f32} =
--     replicate([1i64], defunc_2_map_res_9090)
--   in {defunc_3_map_res_8983}
-- }
