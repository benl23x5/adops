
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
    [nImgs][nChas][nRows][nCols]
    (arrA: [nImgs][nChas][nRows][nCols]f32)
    (arrO: [nImgs][nRows][nCols]f32)
 :  [nImgs][nChas][nRows][nCols]f32
 = regression_dInp arrA arrO


-- $ futhark dev --type-check --inline-aggressively --ad --fuse-soacs --inline-aggressively regression_dInp.fut > dump/regression_dInp.txt
-- ------------------------------------------------------------------------------------------------
-- entry("main",
--       {arrA: [][][][]f32,
--        arrO: [][][]f32},
--       {[][][][]f32})
--   entry_main (nImgs_8997 : i64,
--               nChas_8998 : i64,
--               nRows_8999 : i64,
--               nCols_9000 : i64,
--               arrA_9001 : [nImgs_8997][nChas_8998][nRows_8999][nCols_9000]f32,
--               arrO_9002 : [nImgs_8997][nRows_8999][nCols_9000]f32)
--   : {[nImgs_8997][nChas_8998][nRows_8999][nCols_9000]f32} = {
--   let {iota_res_10492 : [nChas_8998]i64} =
--     iota64(nChas_8998, 0i64, 1i64)
--   let {defunc_1_map_res_11403 : [nChas_8998]f32} =
--     map(nChas_8998,
--         {iota_res_10492},
--         \ {x_10494 : i64}
--           : {f32} ->
--           let {i64_res_10495 : f32} = sitofp i64 x_10494 to f32
--           in {i64_res_10495})
--   let {iota_res_10496 : [nImgs_8997]i64} =
--     iota64(nImgs_8997, 0i64, 1i64)
--   let {iota_res_10497 : [nRows_8999]i64} =
--     iota64(nRows_8999, 0i64, 1i64)
--   let {iota_res_10498 : [nCols_9000]i64} =
--     iota64(nCols_9000, 0i64, 1i64)
--   let {x_10507 : [nImgs_8997][nChas_8998][nRows_8999][nCols_9000]f32} =
--     copy(arrA_9001)
--   let {defunc_3_map_res_adj_10563 : [nImgs_8997][nRows_8999][nCols_9000]f32} =
--     copy(arrO_9002)
--   let {zeroes__10647 : [nImgs_8997][nChas_8998][nRows_8999][nCols_9000]f32} =
--     #[sequential]
--     replicate([nImgs_8997][nChas_8998][nRows_8999][nCols_9000], 0.0f32)
--   let {red_iota_10796 : [nChas_8998]i64} =
--     iota64(nChas_8998, 0i64, 1i64)
--   let {withhacc_res_11126 : [nImgs_8997][nChas_8998][nRows_8999][nCols_9000]f32} =
--     with_acc({([nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {zeroes__10647},
--               (\ {idx_10651 : i64, idx_10652 : i64, idx_10653 : i64, idx_10654 : i64, x_10648 : f32, y_10649 : f32}
--                 : {f32} ->
--                 let {binlam_res_10650 : f32} = fadd32(x_10648, y_10649)
--                 in {binlam_res_10650},
--               {0.0f32}))},
--     \ {acc_cert_p_10663 : unit, acc_p_10664 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})}
--       : {acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} ->
--       let {map_adjs_11402 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} =
--         map(nImgs_8997,
--             {iota_res_10496, defunc_3_map_res_adj_10563, acc_p_10664},
--             \ {x_10565 : i64, map_adj_p_10564 : [nRows_8999][nCols_9000]f32, free_adj_p_10669 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})}
--               : {acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} ->
--               let {x_10566 : bool} = sle64(0i64, x_10565)
--               let {y_10567 : bool} = slt64(x_10565, nImgs_8997)
--               let {bounds_check_10568 : bool} = logand(x_10566, y_10567)
--               let {map_adjs_11401 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} =
--                 map(nRows_8999,
--                     {iota_res_10497, map_adj_p_10564, free_adj_p_10669},
--                     \ {x_10671 : i64, map_adj_p_10670 : [nCols_9000]f32, free_adj_p_10728 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})}
--                       : {acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} ->
--                       let {x_10672 : bool} = sle64(0i64, x_10671)
--                       let {y_10673 : bool} = slt64(x_10671, nRows_8999)
--                       let {bounds_check_10674 : bool} = logand(x_10672, y_10673)
--                       let {map_adjs_11400 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} =
--                         map(nCols_9000,
--                             {iota_res_10498, map_adj_p_10670, free_adj_p_10728},
--                             \ {x_10730 : i64, map_adj_p_10729 : f32, free_adj_p_10783 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})}
--                               : {acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} ->
--                               let {x_10731 : bool} = sle64(0i64, x_10730)
--                               let {y_10732 : bool} = slt64(x_10730, nCols_9000)
--                               let {bounds_check_10733 : bool} = logand(x_10731, y_10732)
--                               let {y_10734 : bool} = logand(bounds_check_10568, bounds_check_10733)
--                               let {y_10735 : bool} = logand(bounds_check_10674, y_10734)
--                               let {defunc_2_reduce_res_11390 : f32,
--                                    defunc_2_reduce_res_ind_11391 : i64,
--                                    defunc_1_map_res_11392 : [nChas_8998]f32} =
--                                 redomap(nChas_8998,
--                                         {iota_res_10492, red_iota_10796},
--                                         {commutative \ {acc_v_10784 : f32, acc_i_10785 : i64, v_10786 : f32, i_10787 : i64}
--                                           : {f32,
--                                              i64} ->
--                                           let {cond_10788 : bool} = eq_f32(acc_v_10784, v_10786)
--                                           let {idx_res_10794 : f32,
--                                                idx_res_10795 : i64} =
--                                             if  cond_10788
--                                             then {
--                                               let {x_10789 : i64} = smin64(acc_i_10785, i_10787)
--                                               in {acc_v_10784, x_10789}
--                                             } else {
--                                               let {y_10790 : f32} = fmax32(acc_v_10784, v_10786)
--                                               let {cond_10791 : bool} = eq_f32(acc_v_10784, y_10790)
--                                               let {x_10792 : f32} =
--                                                 if  cond_10791
--                                                 then {acc_v_10784} else {v_10786}
--                                                 : {f32}
--                                               let {x_10793 : i64} =
--                                                 if  cond_10791
--                                                 then {acc_i_10785} else {i_10787}
--                                                 : {i64}
--                                               in {x_10792, x_10793}
--                                             }
--                                             : {f32,
--                                                i64}
--                                           in {idx_res_10794, idx_res_10795},
--                                         {-f32.inf, -1i64}},
--                                         \ {x_11328 : i64, x_11329 : i64}
--                                           : {f32,
--                                              i64,
--                                              f32} ->
--                                           let {x_11330 : bool} = sle64(0i64, x_11328)
--                                           let {y_11331 : bool} = slt64(x_11328, nChas_8998)
--                                           let {bounds_check_11332 : bool} = logand(x_11330, y_11331)
--                                           let {index_ok_11333 : bool} = logand(y_10735, bounds_check_11332)
--                                           let {index_certs_11334 : unit} =
--                                             assert(index_ok_11333, {"Index [", x_10565 : i64, ", ", x_11328 : i64, ", ", x_10671 : i64, ", ", x_10730 : i64, "] out of bounds for array of shape [", nImgs_8997 : i64, "][", nChas_8998 : i64, "][", nRows_8999 : i64, "][", nCols_9000 : i64, "]."}, "regression_dInp.fut:17:51-77")
--                                           let {arg_11335 : f32} =
--                                             #{index_certs_11334}
--                                             x_10507[x_10565, x_11328, x_10671, x_10730]
--                                           let {defunc_0_f_res_11336 : f32} = fsub32(0.0f32, arg_11335)
--                                           in {defunc_0_f_res_11336, x_11329, defunc_0_f_res_11336})
--                               let {defunc_2_reduce_res_11393 : f32} =
--                                 redomap(nChas_8998,
--                                         {defunc_1_map_res_11392},
--                                         {\ {x_10755 : f32, x_10756 : f32}
--                                           : {f32} ->
--                                           let {defunc_1_op_res_10757 : f32} = fadd32(x_10755, x_10756)
--                                           in {defunc_1_op_res_10757},
--                                         {0.0f32}},
--                                         \ {x_11323 : f32}
--                                           : {f32} ->
--                                           let {exp_arg_11324 : f32} = fsub32(x_11323, defunc_2_reduce_res_11390)
--                                           let {exp_res_11325 : f32} =
--                                             apply exp32(exp_arg_11324)
--                                             : {f32}
--                                           in {exp_res_11325})
--                               let {binop_y_10824 : f32} = fmul32(defunc_2_reduce_res_11393, defunc_2_reduce_res_11393)
--                               let {binop_y_10822 : f32} = fdiv32(1.0f32, defunc_2_reduce_res_11393)
--                               let {defunc_2_reduce_res_contrib_sum_11394 : f32,
--                                    defunc_2_reduce_res_contrib_sum_11395 : f32,
--                                    map_adjs_11396 : [nChas_8998]f32} =
--                                 redomap(nChas_8998,
--                                         {defunc_1_map_res_11403, defunc_1_map_res_11392},
--                                         {commutative \ {x_10840 : f32, y_10841 : f32}
--                                           : {f32} ->
--                                           let {binlam_res_10842 : f32} = fadd32(x_10840, y_10841)
--                                           in {binlam_res_10842},
--                                         {0.0f32},
--                                         commutative \ {x_10835 : f32, y_10836 : f32}
--                                           : {f32} ->
--                                           let {binlam_res_10837 : f32} = fadd32(x_10835, y_10836)
--                                           in {binlam_res_10837},
--                                         {0.0f32}},
--                                         \ {x_11304 : f32, x_11305 : f32}
--                                           : {f32,
--                                              f32,
--                                              f32} ->
--                                           let {binop_y_adj_11306 : f32} = fmul32(map_adj_p_10729, x_11304)
--                                           let {exp_arg_11308 : f32} = fsub32(x_11305, defunc_2_reduce_res_11390)
--                                           let {exp_res_11309 : f32} =
--                                             apply exp32(exp_arg_11308)
--                                             : {f32}
--                                           let {binop_x_adj_11310 : f32} = fmul32(binop_y_10822, binop_y_adj_11306)
--                                           let {binop_y_11311 : f32} = fdiv32(exp_res_11309, binop_y_10824)
--                                           let {binop_y_11312 : f32} = fsub32(0.0f32, binop_y_11311)
--                                           let {binop_y_adj_11313 : f32} = fmul32(binop_y_adj_11306, binop_y_11312)
--                                           let {binop_y_11314 : f32} =
--                                             apply exp32(exp_arg_11308)
--                                             : {f32}
--                                           let {contrib_11315 : f32} = fmul32(binop_x_adj_11310, binop_y_11314)
--                                           let {binop_y_adj_11316 : f32} = fmul32(-1.0f32, contrib_11315)
--                                           in {binop_y_adj_11313, binop_y_adj_11316, contrib_11315})
--                               let {defunc_2_reduce_res_contrib_sum_11397 : f32,
--                                    defunc_1_map_res_adj_11398 : [nChas_8998]f32} =
--                                 redomap(nChas_8998,
--                                         {defunc_1_map_res_11392, map_adjs_11396},
--                                         {commutative \ {x_10856 : f32, y_10857 : f32}
--                                           : {f32} ->
--                                           let {binlam_res_10858 : f32} = fadd32(x_10856, y_10857)
--                                           in {binlam_res_10858},
--                                         {0.0f32}},
--                                         \ {x_11256 : f32, x_11257 : f32}
--                                           : {f32,
--                                              f32} ->
--                                           let {exp_arg_11258 : f32} = fsub32(x_11256, defunc_2_reduce_res_11390)
--                                           let {binop_y_11259 : f32} =
--                                             apply exp32(exp_arg_11258)
--                                             : {f32}
--                                           let {contrib_11260 : f32} = fmul32(binop_y_11259, defunc_2_reduce_res_contrib_sum_11394)
--                                           let {binop_y_adj_11261 : f32} = fmul32(-1.0f32, contrib_11260)
--                                           let {binlam_res_11264 : f32} = fadd32(x_11257, contrib_11260)
--                                           in {binop_y_adj_11261, binlam_res_11264})
--                               let {defunc_2_reduce_res_adj_10861 : f32} = fadd32(defunc_2_reduce_res_contrib_sum_11395, defunc_2_reduce_res_contrib_sum_11397)
--                               let {defunc_1_map_res_adj_i_10867 : f32} =
--                                 defunc_1_map_res_adj_11398[defunc_2_reduce_res_ind_11391]
--                               let {updated_adj_i_10868 : f32} = fadd32(defunc_2_reduce_res_adj_10861, defunc_1_map_res_adj_i_10867)
--                               let {defunc_1_map_res_adj_10869 : [nChas_8998]f32} =
--                                 defunc_1_map_res_adj_11398 with? [defunc_2_reduce_res_ind_11391] = updated_adj_i_10868
--                               let {map_adjs_11399 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} =
--                                 map(nChas_8998,
--                                     {iota_res_10492, defunc_1_map_res_adj_10869, free_adj_p_10783},
--                                     \ {x_10871 : i64, map_adj_p_10870 : f32, free_adj_p_10887 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})}
--                                       : {acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} ->
--                                       let {binop_y_adj_10889 : f32} = fmul32(-1.0f32, map_adj_p_10870)
--                                       let {free_adj_p_10890 : acc(acc_cert_p_10663, [nImgs_8997][nChas_8998][nRows_8999][nCols_9000], {f32})} =
--                                         update_acc(free_adj_p_10887, {x_10565, x_10871, x_10671, x_10730}, {binop_y_adj_10889})
--                                       in {free_adj_p_10890})
--                               in {map_adjs_11399})
--                       in {map_adjs_11400})
--               in {map_adjs_11401})
--       in {map_adjs_11402})
--   in {withhacc_res_11126}
-- }
--