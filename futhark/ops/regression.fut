
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


def main
    [nImgs][nChas][nRows][nCols]
    (arrA: [nImgs][nChas][nRows][nCols]f32)
 :  [nImgs][nRows][nCols]f32
 = regression arrA


-- $ futhark dev --type-check --inline-aggressively --ad --fuse-soacs --inline-aggressively regression.fut > dump/regression.txt
-- ------------------------------------------------------------------------------------------------
-- entry("main",
--       {arrA: [][][][]f32},
--       {[][][]f32})
--   entry_main (nImgs_7746 : i64,
--               nChas_7747 : i64,
--               nRows_7748 : i64,
--               nCols_7749 : i64,
--               arrA_7750 : [nImgs_7746][nChas_7747][nRows_7748][nCols_7749]f32)
--   : {[nImgs_7746][nRows_7748][nCols_7749]f32} = {
--   let {iota_res_8976 : [nChas_7747]i64} =
--     iota64(nChas_7747, 0i64, 1i64)
--   let {defunc_1_map_res_9110 : [nChas_7747]f32} =
--     map(nChas_7747,
--         {iota_res_8976},
--         \ {x_8981 : i64}
--           : {f32} ->
--           let {i64_res_8982 : f32} = sitofp i64 x_8981 to f32
--           in {i64_res_8982})
--   let {iota_res_8989 : [nImgs_7746]i64} =
--     iota64(nImgs_7746, 0i64, 1i64)
--   let {iota_res_8993 : [nRows_7748]i64} =
--     iota64(nRows_7748, 0i64, 1i64)
--   let {iota_res_8997 : [nCols_7749]i64} =
--     iota64(nCols_7749, 0i64, 1i64)
--   let {defunc_3_map_res_9111 : [nImgs_7746][nRows_7748][nCols_7749]f32} =
--     map(nImgs_7746,
--         {iota_res_8989},
--         \ {x_9000 : i64}
--           : {[nRows_7748][nCols_7749]f32} ->
--           let {x_9001 : bool} = sle64(0i64, x_9000)
--           let {y_9002 : bool} = slt64(x_9000, nImgs_7746)
--           let {bounds_check_9003 : bool} = logand(x_9001, y_9002)
--           let {defunc_2_map_res_9109 : [nRows_7748][nCols_7749]f32} =
--             map(nRows_7748,
--                 {iota_res_8993},
--                 \ {x_9005 : i64}
--                   : {[nCols_7749]f32} ->
--                   let {x_9006 : bool} = sle64(0i64, x_9005)
--                   let {y_9007 : bool} = slt64(x_9005, nRows_7748)
--                   let {bounds_check_9008 : bool} = logand(x_9006, y_9007)
--                   let {defunc_1_map_res_9108 : [nCols_7749]f32} =
--                     map(nCols_7749,
--                         {iota_res_8997},
--                         \ {x_9010 : i64}
--                           : {f32} ->
--                           let {x_9011 : bool} = sle64(0i64, x_9010)
--                           let {y_9012 : bool} = slt64(x_9010, nCols_7749)
--                           let {bounds_check_9013 : bool} = logand(x_9011, y_9012)
--                           let {y_9014 : bool} = logand(bounds_check_9003, bounds_check_9013)
--                           let {y_9015 : bool} = logand(bounds_check_9008, y_9014)
--                           let {defunc_2_reduce_res_9104 : f32,
--                                defunc_1_map_res_9105 : [nChas_7747]f32} =
--                             redomap(nChas_7747,
--                                     {iota_res_8976},
--                                     {\ {x_9026 : f32, x_9027 : f32}
--                                       : {f32} ->
--                                       let {defunc_1_op_res_9028 : f32} = fmax32(x_9026, x_9027)
--                                       in {defunc_1_op_res_9028},
--                                     {-f32.inf}},
--                                     \ {x_9080 : i64}
--                                       : {f32,
--                                          f32} ->
--                                       let {x_9081 : bool} = sle64(0i64, x_9080)
--                                       let {y_9082 : bool} = slt64(x_9080, nChas_7747)
--                                       let {bounds_check_9083 : bool} = logand(x_9081, y_9082)
--                                       let {index_ok_9084 : bool} = logand(y_9015, bounds_check_9083)
--                                       let {index_certs_9085 : unit} =
--                                         assert(index_ok_9084, {"Index [", x_9000 : i64, ", ", x_9080 : i64, ", ", x_9005 : i64, ", ", x_9010 : i64, "] out of bounds for array of shape [", nImgs_7746 : i64, "][", nChas_7747 : i64, "][", nRows_7748 : i64, "][", nCols_7749 : i64, "]."}, "regression.fut:17:51-77")
--                                       let {arg_9086 : f32} =
--                                         #{index_certs_9085}
--                                         arrA_7750[x_9000, x_9080, x_9005, x_9010]
--                                       let {defunc_0_f_res_9087 : f32} = fsub32(0.0f32, arg_9086)
--                                       in {defunc_0_f_res_9087, defunc_0_f_res_9087})
--                           let {defunc_2_reduce_res_9106 : f32} =
--                             redomap(nChas_7747,
--                                     {defunc_1_map_res_9105},
--                                     {\ {x_9035 : f32, x_9036 : f32}
--                                       : {f32} ->
--                                       let {defunc_1_op_res_9037 : f32} = fadd32(x_9035, x_9036)
--                                       in {defunc_1_op_res_9037},
--                                     {0.0f32}},
--                                     \ {x_9075 : f32}
--                                       : {f32} ->
--                                       let {exp_arg_9076 : f32} = fsub32(x_9075, defunc_2_reduce_res_9104)
--                                       let {exp_res_9077 : f32} =
--                                         apply exp32(exp_arg_9076)
--                                         : {f32}
--                                       in {exp_res_9077})
--                           let {defunc_2_reduce_res_9107 : f32} =
--                             redomap(nChas_7747,
--                                     {defunc_1_map_res_9105, defunc_1_map_res_9110},
--                                     {\ {x_9049 : f32, x_9050 : f32}
--                                       : {f32} ->
--                                       let {defunc_1_op_res_9051 : f32} = fadd32(x_9049, x_9050)
--                                       in {defunc_1_op_res_9051},
--                                     {0.0f32}},
--                                     \ {x_9066 : f32, x_9067 : f32}
--                                       : {f32} ->
--                                       let {exp_arg_9068 : f32} = fsub32(x_9066, defunc_2_reduce_res_9104)
--                                       let {exp_res_9069 : f32} =
--                                         apply exp32(exp_arg_9068)
--                                         : {f32}
--                                       let {defunc_0_f_res_9070 : f32} = fdiv32(exp_res_9069, defunc_2_reduce_res_9106)
--                                       let {defunc_1_f_res_9072 : f32} = fmul32(x_9067, defunc_0_f_res_9070)
--                                       in {defunc_1_f_res_9072})
--                           in {defunc_2_reduce_res_9107})
--                   in {defunc_1_map_res_9108})
--           in {defunc_2_map_res_9109})
--   in {defunc_3_map_res_9111}
-- }
--