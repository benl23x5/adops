
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

def regression_dInp
    [nImgs][nChas][nRows][nCols]
    (arrA: [nImgs][nChas][nRows][nCols]f32)
    (arrO: [nImgs][nRows][nCols]f32)
 :  [nImgs][nChas][nRows][nCols]f32
 = vjp regression arrA arrO

def main
    (arrA: [1][8][16][32]f32)
    (arrO: [1][16][32]f32)
 :  [1][8][16][32]f32
 = regression_dInp arrA arrO


-- $ futhark dev --type-check --inline-aggressively --ad --fuse-soacs --inline-aggressively regression_dInp.fut > dump/regression_dInp.txt
-- ------------------------------------------------------------------------------------------------
-- entry("main",
--       {arrA: [][][][]f32,
--        arrO: [][][]f32},
--       {[][][][]f32})
--   entry_main (arrA_8989 : [1i64][8i64][16i64][32i64]f32,
--               arrO_8990 : [1i64][16i64][32i64]f32)
--   : {[1i64][8i64][16i64][32i64]f32} = {
--   let {iota_res_10476 : [8i64]i64} =
--     iota64(8i64, 0i64, 1i64)
--   let {defunc_1_map_res_11083 : [8i64]f32} =
--     map(8i64,
--         {iota_res_10476},
--         \ {x_10478 : i64}
--           : {f32} ->
--           let {i64_res_10479 : f32} = sitofp i64 x_10478 to f32
--           in {i64_res_10479})
--   let {iota_res_10481 : [16i64]i64} =
--     iota64(16i64, 0i64, 1i64)
--   let {iota_res_10482 : [32i64]i64} =
--     iota64(32i64, 0i64, 1i64)
--   let {x_10491 : [1i64][8i64][16i64][32i64]f32} =
--     copy(arrA_8989)
--   let {defunc_3_map_res_adj_10549 : [1i64][16i64][32i64]f32} =
--     copy(arrO_8990)
--   let {x_10562 : [16i64][32i64]f32} =
--     defunc_3_map_res_adj_10549[0i64, 0i64 :+ 16i64 * 1i64, 0i64 :+ 32i64 * 1i64]
--   let {zeroes__10631 : [1i64][8i64][16i64][32i64]f32} =
--     #[sequential]
--     replicate([1i64][8i64][16i64][32i64], 0.0f32)
--   let {red_iota_10711 : [8i64]i64} =
--     iota64(8i64, 0i64, 1i64)
--   let {withhacc_res_10882 : [1i64][8i64][16i64][32i64]f32} =
--     with_acc({([1i64][8i64][16i64][32i64], {zeroes__10631},
--               (\ {idx_10635 : i64, idx_10636 : i64, idx_10637 : i64, idx_10638 : i64, x_10632 : f32, y_10633 : f32}
--                 : {f32} ->
--                 let {binlam_res_10634 : f32} = fadd32(x_10632, y_10633)
--                 in {binlam_res_10634},
--               {0.0f32}))},
--     \ {acc_cert_p_10645 : unit, acc_p_10646 : acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})}
--       : {acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})} ->
--       let {map_adjs_11082 : acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})} =
--         map(16i64,
--             {iota_res_10481, x_10562, acc_p_10646},
--             \ {x_10565 : i64, map_adj_p_10564 : [32i64]f32, free_adj_p_10650 : acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})}
--               : {acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})} ->
--               let {x_10566 : bool} = sle64(0i64, x_10565)
--               let {y_10567 : bool} = slt64(x_10565, 16i64)
--               let {bounds_check_10568 : bool} = logand(x_10566, y_10567)
--               let {map_adjs_11081 : acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})} =
--                 map(32i64,
--                     {iota_res_10482, map_adj_p_10564, free_adj_p_10650},
--                     \ {x_10652 : i64, map_adj_p_10651 : f32, free_adj_p_10698 : acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})}
--                       : {acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})} ->
--                       let {x_10653 : bool} = sle64(0i64, x_10652)
--                       let {y_10654 : bool} = slt64(x_10652, 32i64)
--                       let {bounds_check_10655 : bool} = logand(x_10653, y_10654)
--                       let {y_10656 : bool} = logand(bounds_check_10568, bounds_check_10655)
--                       let {defunc_2_reduce_res_11071 : f32,
--                            defunc_2_reduce_res_ind_11072 : i64,
--                            defunc_1_map_res_11073 : [8i64]f32} =
--                         redomap(8i64,
--                                 {iota_res_10476, red_iota_10711},
--                                 {commutative \ {acc_v_10699 : f32, acc_i_10700 : i64, v_10701 : f32, i_10702 : i64}
--                                   : {f32,
--                                      i64} ->
--                                   let {cond_10703 : bool} = eq_f32(acc_v_10699, v_10701)
--                                   let {idx_res_10709 : f32,
--                                        idx_res_10710 : i64} =
--                                     if  cond_10703
--                                     then {
--                                       let {x_10704 : i64} = smin64(acc_i_10700, i_10702)
--                                       in {acc_v_10699, x_10704}
--                                     } else {
--                                       let {y_10705 : f32} = fmax32(acc_v_10699, v_10701)
--                                       let {cond_10706 : bool} = eq_f32(acc_v_10699, y_10705)
--                                       let {x_10707 : f32} =
--                                         if  cond_10706
--                                         then {acc_v_10699} else {v_10701}
--                                         : {f32}
--                                       let {x_10708 : i64} =
--                                         if  cond_10706
--                                         then {acc_i_10700} else {i_10702}
--                                         : {i64}
--                                       in {x_10707, x_10708}
--                                     }
--                                     : {f32,
--                                        i64}
--                                   in {idx_res_10709, idx_res_10710},
--                                 {-f32.inf, -1i64}},
--                                 \ {x_11023 : i64, x_11024 : i64}
--                                   : {f32,
--                                      i64,
--                                      f32} ->
--                                   let {x_11025 : bool} = sle64(0i64, x_11023)
--                                   let {y_11026 : bool} = slt64(x_11023, 8i64)
--                                   let {bounds_check_11027 : bool} = logand(x_11025, y_11026)
--                                   let {index_ok_11028 : bool} = logand(y_10656, bounds_check_11027)
--                                   let {index_certs_11029 : unit} =
--                                     assert(index_ok_11028, {"Index [", 0i64 : i64, ", ", x_11023 : i64, ", ", x_10565 : i64, ", ", x_10652 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 8i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "regression_dInp.fut:16:51-77")
--                                   let {arg_11030 : f32} =
--                                     #{index_certs_11029}
--                                     x_10491[0i64, x_11023, x_10565, x_10652]
--                                   let {defunc_0_f_res_11031 : f32} = fsub32(0.0f32, arg_11030)
--                                   in {defunc_0_f_res_11031, x_11024, defunc_0_f_res_11031})
--                       let {defunc_2_reduce_res_11074 : f32} =
--                         redomap(8i64,
--                                 {defunc_1_map_res_11073},
--                                 {\ {x_10676 : f32, x_10677 : f32}
--                                   : {f32} ->
--                                   let {defunc_1_op_res_10678 : f32} = fadd32(x_10676, x_10677)
--                                   in {defunc_1_op_res_10678},
--                                 {0.0f32}},
--                                 \ {x_11018 : f32}
--                                   : {f32} ->
--                                   let {exp_arg_11019 : f32} = fsub32(x_11018, defunc_2_reduce_res_11071)
--                                   let {exp_res_11020 : f32} =
--                                     apply exp32(exp_arg_11019)
--                                     : {f32}
--                                   in {exp_res_11020})
--                       let {binop_y_10739 : f32} = fmul32(defunc_2_reduce_res_11074, defunc_2_reduce_res_11074)
--                       let {binop_y_10737 : f32} = fdiv32(1.0f32, defunc_2_reduce_res_11074)
--                       let {defunc_2_reduce_res_contrib_sum_11075 : f32,
--                            defunc_2_reduce_res_contrib_sum_11076 : f32,
--                            map_adjs_11077 : [8i64]f32} =
--                         redomap(8i64,
--                                 {defunc_1_map_res_11083, defunc_1_map_res_11073},
--                                 {commutative \ {x_10755 : f32, y_10756 : f32}
--                                   : {f32} ->
--                                   let {binlam_res_10757 : f32} = fadd32(x_10755, y_10756)
--                                   in {binlam_res_10757},
--                                 {0.0f32},
--                                 commutative \ {x_10750 : f32, y_10751 : f32}
--                                   : {f32} ->
--                                   let {binlam_res_10752 : f32} = fadd32(x_10750, y_10751)
--                                   in {binlam_res_10752},
--                                 {0.0f32}},
--                                 \ {x_10999 : f32, x_11000 : f32}
--                                   : {f32,
--                                      f32,
--                                      f32} ->
--                                   let {binop_y_adj_11001 : f32} = fmul32(map_adj_p_10651, x_10999)
--                                   let {exp_arg_11003 : f32} = fsub32(x_11000, defunc_2_reduce_res_11071)
--                                   let {exp_res_11004 : f32} =
--                                     apply exp32(exp_arg_11003)
--                                     : {f32}
--                                   let {binop_x_adj_11005 : f32} = fmul32(binop_y_10737, binop_y_adj_11001)
--                                   let {binop_y_11006 : f32} = fdiv32(exp_res_11004, binop_y_10739)
--                                   let {binop_y_11007 : f32} = fsub32(0.0f32, binop_y_11006)
--                                   let {binop_y_adj_11008 : f32} = fmul32(binop_y_adj_11001, binop_y_11007)
--                                   let {binop_y_11009 : f32} =
--                                     apply exp32(exp_arg_11003)
--                                     : {f32}
--                                   let {contrib_11010 : f32} = fmul32(binop_x_adj_11005, binop_y_11009)
--                                   let {binop_y_adj_11011 : f32} = fmul32(-1.0f32, contrib_11010)
--                                   in {binop_y_adj_11008, binop_y_adj_11011, contrib_11010})
--                       let {defunc_2_reduce_res_contrib_sum_11078 : f32,
--                            defunc_1_map_res_adj_11079 : [8i64]f32} =
--                         redomap(8i64,
--                                 {defunc_1_map_res_11073, map_adjs_11077},
--                                 {commutative \ {x_10771 : f32, y_10772 : f32}
--                                   : {f32} ->
--                                   let {binlam_res_10773 : f32} = fadd32(x_10771, y_10772)
--                                   in {binlam_res_10773},
--                                 {0.0f32}},
--                                 \ {x_10951 : f32, x_10952 : f32}
--                                   : {f32,
--                                      f32} ->
--                                   let {exp_arg_10953 : f32} = fsub32(x_10951, defunc_2_reduce_res_11071)
--                                   let {binop_y_10954 : f32} =
--                                     apply exp32(exp_arg_10953)
--                                     : {f32}
--                                   let {contrib_10955 : f32} = fmul32(binop_y_10954, defunc_2_reduce_res_contrib_sum_11075)
--                                   let {binop_y_adj_10956 : f32} = fmul32(-1.0f32, contrib_10955)
--                                   let {binlam_res_10959 : f32} = fadd32(x_10952, contrib_10955)
--                                   in {binop_y_adj_10956, binlam_res_10959})
--                       let {defunc_2_reduce_res_adj_10776 : f32} = fadd32(defunc_2_reduce_res_contrib_sum_11076, defunc_2_reduce_res_contrib_sum_11078)
--                       let {defunc_1_map_res_adj_i_10782 : f32} =
--                         defunc_1_map_res_adj_11079[defunc_2_reduce_res_ind_11072]
--                       let {updated_adj_i_10783 : f32} = fadd32(defunc_2_reduce_res_adj_10776, defunc_1_map_res_adj_i_10782)
--                       let {defunc_1_map_res_adj_10784 : [8i64]f32} =
--                         defunc_1_map_res_adj_11079 with? [defunc_2_reduce_res_ind_11072] = updated_adj_i_10783
--                       let {map_adjs_11080 : acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})} =
--                         map(8i64,
--                             {iota_res_10476, defunc_1_map_res_adj_10784, free_adj_p_10698},
--                             \ {x_10786 : i64, map_adj_p_10785 : f32, free_adj_p_10797 : acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})}
--                               : {acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})} ->
--                               let {binop_y_adj_10799 : f32} = fmul32(-1.0f32, map_adj_p_10785)
--                               let {free_adj_p_10800 : acc(acc_cert_p_10645, [1i64][8i64][16i64][32i64], {f32})} =
--                                 update_acc(free_adj_p_10797, {0i64, x_10786, x_10565, x_10652}, {binop_y_adj_10799})
--                               in {free_adj_p_10800})
--                       in {map_adjs_11080})
--               in {map_adjs_11081})
--       in {map_adjs_11082})
--   in {withhacc_res_10882}
-- }
--