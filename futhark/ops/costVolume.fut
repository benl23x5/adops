open import "/prelude/math"



def costVolume
    [nImgs][nChas][nRows][nCols]
    (iStart: i64) (nCount: i64)
    (arrL: [nImgs][nChas][nRows][nCols]f32)
    (arrR: [nImgs][nChas][nRows][nCols]f32)
 :  [nImgs][nCount][nRows][nCols]f32
 =  tabulate_4d nImgs nCount nRows nCols
     (\iImg iDisp iRow iCol ->
        let arrVecL = tabulate nChas (\iCha -> arrL[iImg, iCha, iRow, iCol])
        let iSrc    = iCol - iStart - iDisp
        -- TODO: index with zero for out of range
        let arrVecR = tabulate nChas (\iCha -> arrR[iImg, iCha, iRow, iSrc])
        in f32.sum (map2 (\xL xR -> f32.abs (xL - xR)) arrVecL arrVecR))


def costVolume_dArrR
    [nImgs][nChas][nRows][nCols]
    (iStart: i64) (nCount: i64)
    (arrL: [nImgs][nChas][nRows][nCols]f32)
    (arrR: [nImgs][nChas][nRows][nCols]f32)
    (arrO: [nImgs][nCount][nRows][nCols]f32)
  : [nImgs][nChas][nRows][nCols]f32
  = vjp (costVolume iStart nCount arrL) arrR arrO


def main
    (arrA: [1] [8][16][32]f32)
    (arrB: [1] [8][16][32]f32)
    (arrO: [1][12][16][32]f32)
 : [1][8][16][32]f32
 = costVolume_dArrR 0 12 arrA arrB arrO




-- $ futhark dev --type-check --inline-aggressively --ad --fuse-soacs --inline-aggressively costVolume.fut > dump/costVolume.txt
-- ------------------------------------------------------------------------------------------------
-- entry("main",
--       {arrA: [][][][]f32,
--        arrB: [][][][]f32,
--        arrO: [][][][]f32},
--       {[][][][]f32})
--   entry_main (arrA_10742 : [1i64][8i64][16i64][32i64]f32,
--               arrB_10743 : [1i64][8i64][16i64][32i64]f32,
--               arrO_10744 : [1i64][12i64][16i64][32i64]f32)
--   : {[1i64][8i64][16i64][32i64]f32} = {
--   let {iota_res_12723 : [12i64]i64}        = iota64(12i64, 0i64, 1i64)
--   let {iota_res_12724 : [16i64]i64}        = iota64(16i64, 0i64, 1i64)
--   let {iota_res_12725 : [32i64]i64}        = iota64(32i64, 0i64, 1i64)
--   let {iota_res_12726 : [8i64]i64}         = iota64(8i64, 0i64, 1i64)
--   let {x_12791 : [12i64][16i64][32i64]f32} =
--     arrO_10744[0i64, 0i64 :+ 12i64 * 1i64, 0i64 :+ 16i64 * 1i64, 0i64 :+ 32i64 * 1i64]
--
--   let {zeroes__12865 : [1i64][8i64][16i64][32i64]f32} =
--     #[sequential]
--     replicate([1i64][8i64][16i64][32i64], 0.0f32)
--
--   let {withhacc_res_13170 : [1i64][8i64][16i64][32i64]f32} =
--     with_acc({([1i64][8i64][16i64][32i64], {zeroes__12865},
--               (\ {idx_12869 : i64, idx_12870 : i64, idx_12871 : i64, idx_12872 : i64, x_12866 : f32, y_12867 : f32}
--                 : {f32} ->
--                 let {binlam_res_12868 : f32} = fadd32(x_12866, y_12867)
--                 in  {binlam_res_12868},
--               {0.0f32}))},
--     \ {acc_cert_p_12881 : unit, acc_p_12882 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})}
--       : {acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} ->
--
--       let {map_adjs_13317 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} =
--         map(12i64, {iota_res_12723, x_12791, acc_p_12882},
--             \ {x_12794 : i64, map_adj_p_12793 : [16i64][32i64]f32, free_adj_p_12887 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})}
--               : {acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} ->
--
--               let {map_adjs_13316 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} =
--                 map(16i64, {iota_res_12724, map_adj_p_12793, free_adj_p_12887},
--                     \ {x_12889 : i64, map_adj_p_12888 : [32i64]f32, free_adj_p_12934 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})}
--                       : {acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} ->
--                       let {x_12890 : bool}            = sle64(0i64, x_12889)
--                       let {y_12891 : bool}            = slt64(x_12889, 16i64)
--                       let {bounds_check_12892 : bool} = logand(x_12890, y_12891)
--
--                       let {map_adjs_13315 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} =
--                         map(32i64, {iota_res_12725, map_adj_p_12888, free_adj_p_12934},
--                             \ {x_12936 : i64, map_adj_p_12935 : f32, free_adj_p_12977 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})}
--                               : {acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} ->
--                               let {x_12937 : bool}            = sle64(0i64, x_12936)
--                               let {y_12938 : bool}            = slt64(x_12936, 32i64)
--                               let {bounds_check_12939 : bool} = logand(x_12937, y_12938)
--                               let {y_12940 : bool}            = logand(bounds_check_12892, bounds_check_12939)
--                               let {iSrc_12949 : i64}          = sub64(x_12936, x_12794)
--                               let {y_12951 : bool}            = slt64(iSrc_12949, 32i64)
--                               let {x_12950 : bool}            = sle64(0i64, iSrc_12949)
--                               let {bounds_check_12952 : bool} = logand(x_12950, y_12951)
--                               let {y_12953 : bool}            = logand(bounds_check_12892, bounds_check_12952)
--                               let {map_adjs_13314 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} =
--                                 map(8i64,
--                                     {iota_res_12726, free_adj_p_12977},
--                                     \ {x_13277 : i64, free_adj_p_13278 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})}
--                                       : {acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} ->
--                                       let {x_13280 : bool}            = sle64(0i64, x_13277)
--                                       let {y_13281 : bool}            = slt64(x_13277, 8i64)
--                                       let {bounds_check_13282 : bool} = logand(x_13280, y_13281)
--                                       let {index_ok_13283 : bool}     = logand(y_12940, bounds_check_13282)
--                                       let {index_certs_13284 : unit} =
--                                         assert(index_ok_13283, {"Index [", 0i64 : i64, ", ", x_13277 : i64, ", ", x_12889 : i64, ", ", x_12936 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 8i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "costVolume.fut:57:48-75")
--                                       let {defunc_0_f_res_13285 : f32} =
--                                         #{index_certs_13284}
--                                         arrA_10742[0i64, x_13277, x_12889, x_12936]
--                                       let {x_13288 : bool}             = sle64(0i64, x_13277)
--                                       let {y_13289 : bool}             = slt64(x_13277, 8i64)
--                                       let {bounds_check_13290 : bool}  = logand(x_13288, y_13289)
--                                       let {index_ok_13291 : bool}      = logand(y_12953, bounds_check_13290)
--                                       let {index_certs_13292 : unit}   = assert(index_ok_13291, {"Index [", 0i64 : i64, ", ", x_13277 : i64, ", ", x_12889 : i64, ", ", iSrc_12949 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 8i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "costVolume.fut:59:48-75")
--                                       let {defunc_0_f_res_13293 : f32} = #{index_certs_13292} arrB_10743[0i64, x_13277, x_12889, iSrc_12949]
--                                       let {abs_arg_13295 : f32}        = fsub32(defunc_0_f_res_13285, defunc_0_f_res_13293)
--                                       let {binop_y_13296 : f32}        = fsignum32 abs_arg_13295
--                                       let {contrib_13297 : f32}        = fmul32(map_adj_p_12935, binop_y_13296)
--                                       let {binop_y_adj_13298 : f32}    = fmul32(-1.0f32, contrib_13297)
--                                       let {free_adj_p_13300 : acc(acc_cert_p_12881, [1i64][8i64][16i64][32i64], {f32})} =
--                                         update_acc(free_adj_p_13278, {0i64, x_13277, x_12889, iSrc_12949}, {binop_y_adj_13298})
--                                       in {free_adj_p_13300})
--                               in {map_adjs_13314})
--                       in {map_adjs_13315})
--               in {map_adjs_13316})
--       in {map_adjs_13317})
--   in {withhacc_res_13170}
-- }
