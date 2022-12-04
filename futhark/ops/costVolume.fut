open import "/prelude/math"


def tabulate_4d
    (nN: i64)(nC: i64)(nH: i64)(nW: i64)
    (f: i64 -> i64 -> i64 -> i64 -> f32)
 :  [nN][nC][nH][nW]f32
 = tabulate nN (\iN ->
    tabulate nC (\iC ->
     tabulate nH (\iH ->
      tabulate nW (\iW -> f iN iC iH iW))))


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
        let arrVecR = tabulate nChas (\iCha ->
                if iSrc >= 0 then arrR[iImg, iCha, iRow, iSrc]
                else 0)
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
--   entry_main (arrA_10624 : [1i64][8i64][16i64][32i64]f32,
--               arrB_10625 : [1i64][8i64][16i64][32i64]f32,
--               arrO_10626 : [1i64][12i64][16i64][32i64]f32)
--   : {[1i64][8i64][16i64][32i64]f32} = {
--   let {iota_res_12632 : [12i64]i64} =
--     iota64(12i64, 0i64, 1i64)
--   let {iota_res_12633 : [16i64]i64} =
--     iota64(16i64, 0i64, 1i64)
--   let {iota_res_12634 : [32i64]i64} =
--     iota64(32i64, 0i64, 1i64)
--   let {iota_res_12635 : [8i64]i64} =
--     iota64(8i64, 0i64, 1i64)
--   let {x_12638 : [1i64][8i64][16i64][32i64]f32} =
--     copy(arrB_10625)
--   let {defunc_4_map_res_adj_12685 : [1i64][12i64][16i64][32i64]f32} =
--     copy(arrO_10626)
--   let {x_12701 : [12i64][16i64][32i64]f32} =
--     defunc_4_map_res_adj_12685[0i64, 0i64 :+ 12i64 * 1i64, 0i64 :+ 16i64 * 1i64, 0i64 :+ 32i64 * 1i64]
--   let {zeroes__12776 : [1i64][8i64][16i64][32i64]f32} =
--     #[sequential]
--     replicate([1i64][8i64][16i64][32i64], 0.0f32)
--   let {withhacc_res_13125 : [1i64][8i64][16i64][32i64]f32} =
--     with_acc({([1i64][8i64][16i64][32i64], {zeroes__12776},
--               (\ {idx_12780 : i64, idx_12781 : i64, idx_12782 : i64, idx_12783 : i64, x_12777 : f32, y_12778 : f32}
--                 : {f32} ->
--                 let {binlam_res_12779 : f32} = fadd32(x_12777, y_12778)
--                 in {binlam_res_12779},
--               {0.0f32}))},
--     \ {acc_cert_p_12792 : unit, acc_p_12793 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})}
--       : {acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} ->
--       let {map_adjs_13353 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} =
--         map(12i64,
--             {iota_res_12632, x_12701, acc_p_12793},
--             \ {x_12704 : i64, map_adj_p_12703 : [16i64][32i64]f32, free_adj_p_12798 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})}
--               : {acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} ->
--               let {map_adjs_13352 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} =
--                 map(16i64,
--                     {iota_res_12633, map_adj_p_12703, free_adj_p_12798},
--                     \ {x_12800 : i64, map_adj_p_12799 : [32i64]f32, free_adj_p_12846 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})}
--                       : {acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} ->
--                       let {x_12801 : bool} = sle64(0i64, x_12800)
--                       let {y_12802 : bool} = slt64(x_12800, 16i64)
--                       let {bounds_check_12803 : bool} = logand(x_12801, y_12802)
--                       let {map_adjs_13351 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} =
--                         map(32i64,
--                             {iota_res_12634, map_adj_p_12799, free_adj_p_12846},
--                             \ {x_12848 : i64, map_adj_p_12847 : f32, free_adj_p_12890 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})}
--                               : {acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} ->
--                               let {x_12849 : bool} = sle64(0i64, x_12848)
--                               let {y_12850 : bool} = slt64(x_12848, 32i64)
--                               let {bounds_check_12851 : bool} = logand(x_12849, y_12850)
--                               let {y_12852 : bool} = logand(bounds_check_12803, bounds_check_12851)
--                               let {iSrc_12861 : i64} = sub64(x_12848, x_12704)
--                               let {y_12863 : bool} = slt64(iSrc_12861, 32i64)
--                               let {cond_12862 : bool} = sle64(0i64, iSrc_12861)
--                               let {bounds_check_12864 : bool} = logand(cond_12862, y_12863)
--                               let {y_12865 : bool} = logand(bounds_check_12803, bounds_check_12864)
--                               let {map_adjs_13350 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} =
--                                 map(8i64,
--                                     {iota_res_12635, free_adj_p_12890},
--                                     \ {x_13263 : i64, free_adj_p_13264 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})}
--                                       : {acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} ->
--                                       let {x_13266 : bool} = sle64(0i64, x_13263)
--                                       let {y_13267 : bool} = slt64(x_13263, 8i64)
--                                       let {bounds_check_13268 : bool} = logand(x_13266, y_13267)
--                                       let {index_ok_13269 : bool} = logand(y_12852, bounds_check_13268)
--                                       let {index_certs_13270 : unit} =
--                                         assert(index_ok_13269, {"Index [", 0i64 : i64, ", ", x_13263 : i64, ", ", x_12800 : i64, ", ", x_12848 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 8i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "costVolume.fut:22:48-75")
--                                       let {defunc_0_f_res_13271 : f32} =
--                                         #{index_certs_13270}
--                                         arrA_10624[0i64, x_13263, x_12800, x_12848]
--                                       let {defunc_0_f_res_13274 : f32} =
--                                         if  cond_12862
--                                         then {
--                                           let {x_13343 : bool} = sle64(0i64, x_13263)
--                                           let {y_13344 : bool} = slt64(x_13263, 8i64)
--                                           let {bounds_check_13345 : bool} = logand(x_13343, y_13344)
--                                           let {index_ok_13346 : bool} = logand(y_12865, bounds_check_13345)
--                                           let {index_certs_13347 : unit} =
--                                             assert(index_ok_13346, {"Index [", 0i64 : i64, ", ", x_13263 : i64, ", ", x_12800 : i64, ", ", iSrc_12861 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 8i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "costVolume.fut:25:35-62")
--                                           let {defunc_0_f_res_t_res_13348 : f32} =
--                                             #{index_certs_13347}
--                                             x_12638[0i64, x_13263, x_12800, iSrc_12861]
--                                           in {defunc_0_f_res_t_res_13348}
--                                         } else {0.0f32}
--                                         : {f32}
--                                       let {abs_arg_13282 : f32} = fsub32(defunc_0_f_res_13271, defunc_0_f_res_13274)
--                                       let {binop_y_13283 : f32} = fsignum32 abs_arg_13282
--                                       let {contrib_13284 : f32} = fmul32(map_adj_p_12847, binop_y_13283)
--                                       let {binop_y_adj_13285 : f32} = fmul32(-1.0f32, contrib_13284)
--                                       let {branch_adj_13287 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} =
--                                         if  cond_12862
--                                         then {
--                                           let {free_adj_p_13349 : acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})} =
--                                             update_acc(free_adj_p_13264, {0i64, x_13263, x_12800, iSrc_12861}, {binop_y_adj_13285})
--                                           in {free_adj_p_13349}
--                                         } else {free_adj_p_13264}
--                                         : {acc(acc_cert_p_12792, [1i64][8i64][16i64][32i64], {f32})}
--                                       in {branch_adj_13287})
--                               in {map_adjs_13350})
--                       in {map_adjs_13351})
--               in {map_adjs_13352})
--       in {map_adjs_13353})
--   in {withhacc_res_13125}
-- }
--