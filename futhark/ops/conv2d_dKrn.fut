
def tabulate_4d
    (nN: i64)(nC: i64)(nH: i64)(nW: i64)
    (f: i64 -> i64 -> i64 -> i64 -> f32)
 :  [nN][nC][nH][nW]f32
 = tabulate nN (\iN ->
    tabulate nC (\iC ->
     tabulate nH (\iH ->
      tabulate nW (\iW -> f iN iC iH iW))))


def slice4
    [nAi][nAc][nAh][nAw]
    (nKi: i64) (nKc: i64) (nKh: i64) (nKw: i64)
    (arrA:  [nAi][nAc][nAh][nAw]f32)
    (start: (i64, i64, i64, i64))
 :  ([nKi][nKc][nKh][nKw]f32)
 = arrA[
    start.0 : start.0 + nKi,
    start.1 : start.1 + nKc,
    start.2 : start.2 + nKh,
    start.3 : start.3 + nKw] :> [nKi][nKc][nKh][nKw]f32


def mmap4
    [nN][nC][nH][nW]
    (f: f32 -> f32 -> f32)
    (a: [nN][nC][nH][nW]f32)
    (b: [nN][nC][nH][nW]f32)
 :  [nN][nC][nH][nW]f32
 = map2 (\x1 y1 ->
   map2 (\x2 y2 ->
   map2 (\x3 y3 ->
   map2 (\x4 y4 -> f x4 y4) x3 y3) x2 y2) x1 y1) a b


def dot4
    [nN][nC][nH][nW]
    (arrA: [nN][nC][nH][nW]f32)
    (arrB: [nN][nC][nH][nW]f32)
 : f32
 = reduce (\x y -> x + y) 0 (flatten_4d (mmap4 (\x y -> x * y) arrA arrB))


def conv2d
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
 : ([nAi][nBc][nBh][nBw]f32)
 = tabulate_4d nAi nBc nBh nBw (\iImg iCout iBh iBw ->
        let arrAt = slice4 1 nAc nKh nKw arrA (iImg,  0, iBh, iBw)
        let arrKt = slice4 1 nAc nKh nKw arrK (iCout, 0,   0,   0)
        in  dot4 arrAt arrKt)


def conv2d_dKrn
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
    (arrO: [nAi][nBc][nBh][nBw]f32)
 : ([nBc][nAc][nKh][nKw]f32)
 = vjp (conv2d arrA) arrK arrO


def main
    (arrA: [1][4][16][32]f32)
    (arrK: [4][4][3][3]f32)
    (arrO: [1][4][16][32]f32)
 : [4][4][3][3]f32
 = conv2d_dKrn arrA arrK arrO



-- $ futhark dev --type-check --inline-aggressively --ad conv2d_dKrn.fut > dump/conv2d_dKrn.txt
-- ------------------------------------------------------------------------------------------------
-- entry("main",
--       {arrA: [][][][]f32,
--        arrK: [][][][]f32,
--        arrO: [][][][]f32},
--       {[][][][]f32})
--   entry_main (arrA_11006 : [1i64][4i64][16i64][32i64]f32,
--               arrK_11007 : [4i64][4i64][3i64][3i64]f32,
--               arrO_11008 : [1i64][4i64][16i64][32i64]f32)
--   : {[4i64][4i64][3i64][3i64]f32} = {
--   let {iota_res_15126 : [4i64]i64} =
--     iota64(4i64, 0i64, 1i64)
--   let {iota_res_15130 : [16i64]i64} =
--     iota64(16i64, 0i64, 1i64)
--   let {iota_res_15134 : [32i64]i64} =
--     iota64(32i64, 0i64, 1i64)
--   let {x_15171 : [4i64][4i64][3i64][3i64]f32} =
--     copy(arrK_11007)
--   let {defunc_4_map_res_adj_15248 : [1i64][4i64][16i64][32i64]f32} =
--     copy(arrO_11008)
--   let {defunc_3_map_res_15182 : [4i64][16i64][32i64]f32} =
--     map(4i64,
--         {iota_res_15126},
--         \ {x_15183 : i64}
--           : {[16i64][32i64]f32} ->
--           let {j_15184 : i64} = add64(1i64, x_15183)
--           let {zero_leq_i_p_m_t_s_15185 : bool} = sle64(0i64, x_15183)
--           let {i_p_m_t_s_leq_w_15186 : bool} = slt64(x_15183, 4i64)
--           let {i_lte_j_15187 : bool} = sle64(x_15183, j_15184)
--           let {y_15188 : bool} = logand(zero_leq_i_p_m_t_s_15185, i_p_m_t_s_leq_w_15186)
--           let {y_15189 : bool} = logand(zero_leq_i_p_m_t_s_15185, y_15188)
--           let {y_15190 : bool} = logand(i_lte_j_15187, y_15189)
--           let {forwards_ok_15191 : bool} = logand(zero_leq_i_p_m_t_s_15185, y_15190)
--           let {index_certs_15195 : unit} =
--             assert(forwards_ok_15191, {"Index [", x_15183 : i64, ":", j_15184 : i64, ", ", 0i64 : i64, ":", 4i64 : i64, ", ", 0i64 : i64, ":", 3i64 : i64, ", ", 0i64 : i64, ":", 3i64 : i64, "] out of bounds for array of shape [", 4i64 : i64, "][", 4i64 : i64, "][", 3i64 : i64, "][", 3i64 : i64, "]."}, "conv2d_dKrn.fut:18:4-22:28")
--           let {as_transformed_row_15196 : [4i64][3i64][3i64]f32} =
--             #{index_certs_15195}
--             x_15171[x_15183, 0i64 :+ 4i64 * 1i64, 0i64 :+ 3i64 * 1i64, 0i64 :+ 3i64 * 1i64]
--           let {defunc_2_map_res_15197 : [16i64][32i64]f32} =
--             map(16i64,
--                 {iota_res_15130},
--                 \ {x_15198 : i64}
--                   : {[32i64]f32} ->
--                   let {j_15199 : i64} = add64(3i64, x_15198)
--                   let {i_p_m_t_s_15200 : i64} = add64(2i64, x_15198)
--                   let {zero_leq_i_p_m_t_s_15201 : bool} = sle64(0i64, i_p_m_t_s_15200)
--                   let {i_p_m_t_s_leq_w_15202 : bool} = slt64(i_p_m_t_s_15200, 16i64)
--                   let {zero_lte_i_15203 : bool} = sle64(0i64, x_15198)
--                   let {i_lte_j_15204 : bool} = sle64(x_15198, j_15199)
--                   let {y_15205 : bool} = logand(i_p_m_t_s_leq_w_15202, zero_lte_i_15203)
--                   let {y_15206 : bool} = logand(zero_leq_i_p_m_t_s_15201, y_15205)
--                   let {y_15207 : bool} = logand(i_lte_j_15204, y_15206)
--                   let {forwards_ok_15208 : bool} = logand(zero_lte_i_15203, y_15207)
--                   let {defunc_1_map_res_15210 : [32i64]f32} =
--                     map(32i64,
--                         {iota_res_15134},
--                         \ {x_15211 : i64}
--                           : {f32} ->
--                           let {j_15212 : i64} = add64(3i64, x_15211)
--                           let {i_p_m_t_s_15213 : i64} = add64(2i64, x_15211)
--                           let {zero_leq_i_p_m_t_s_15214 : bool} = sle64(0i64, i_p_m_t_s_15213)
--                           let {i_p_m_t_s_leq_w_15215 : bool} = slt64(i_p_m_t_s_15213, 32i64)
--                           let {zero_lte_i_15216 : bool} = sle64(0i64, x_15211)
--                           let {i_lte_j_15217 : bool} = sle64(x_15211, j_15212)
--                           let {y_15218 : bool} = logand(i_p_m_t_s_leq_w_15215, zero_lte_i_15216)
--                           let {y_15219 : bool} = logand(zero_leq_i_p_m_t_s_15214, y_15218)
--                           let {y_15220 : bool} = logand(i_lte_j_15217, y_15219)
--                           let {forwards_ok_15221 : bool} = logand(zero_lte_i_15216, y_15220)
--                           let {y_15224 : bool} = logand(forwards_ok_15208, forwards_ok_15221)
--                           let {index_certs_15226 : unit} =
--                             assert(y_15224, {"Index [", 0i64 : i64, ":", 1i64 : i64, ", ", 0i64 : i64, ":", 4i64 : i64, ", ", x_15198 : i64, ":", j_15199 : i64, ", ", x_15211 : i64, ":", j_15212 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 4i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "conv2d_dKrn.fut:18:4-22:28")
--                           let {x_15227 : [4i64][3i64][3i64]f32} =
--                             #{index_certs_15226}
--                             arrA_11006[0i64, 0i64 :+ 4i64 * 1i64, x_15198 :+ 3i64 * 1i64, x_15211 :+ 3i64 * 1i64]
--                           let {defunc_7_map_res_15228 : [4i64][3i64][3i64]f32} =
--                             map(4i64,
--                                 {x_15227, as_transformed_row_15196},
--                                 \ {x_15229 : [3i64][3i64]f32, zip_copy_transformed_row_15230 : [3i64][3i64]f32}
--                                   : {[3i64][3i64]f32} ->
--                                   let {defunc_4_map_res_15231 : [3i64][3i64]f32} =
--                                     map(3i64,
--                                         {x_15229, zip_copy_transformed_row_15230},
--                                         \ {x_15232 : [3i64]f32, x_15233 : [3i64]f32}
--                                           : {[3i64]f32} ->
--                                           let {defunc_1_map_res_15234 : [3i64]f32} =
--                                             map(3i64,
--                                                 {x_15232, x_15233},
--                                                 \ {x_15235 : f32, x_15236 : f32}
--                                                   : {f32} ->
--                                                   let {defunc_1_f_res_15237 : f32} = fmul32(x_15235, x_15236)
--                                                   in {defunc_1_f_res_15237})
--                                           in {defunc_1_map_res_15234})
--                                   in {defunc_4_map_res_15231})
--                           let {defunc_10_map_res_15238 : [1i64][4i64][3i64][3i64]f32} =
--                             replicate([1i64], defunc_7_map_res_15228)
--                           let {flatten_res_15239 : [36i64]f32} =
--                             reshape([36i64], defunc_10_map_res_15238)
--                           let {defunc_2_reduce_res_15240 : f32} =
--                             redomap(36i64,
--                                     {flatten_res_15239},
--                                     {\ {x_15241 : f32, x_15242 : f32}
--                                       : {f32} ->
--                                       let {defunc_1_op_res_15243 : f32} = fadd32(x_15241, x_15242)
--                                       in {defunc_1_op_res_15243},
--                                     {0.0f32}},
--                                     \ {x_15244 : f32}
--                                       : {f32} ->
--                                       {x_15244})
--                           in {defunc_2_reduce_res_15240})
--                   in {defunc_1_map_res_15210})
--           in {defunc_2_map_res_15197})
--   let {defunc_4_map_res_15172 : [1i64][4i64][16i64][32i64]f32} =
--     replicate([1i64], defunc_3_map_res_15182)
--   let {x_15264 : [4i64][16i64][32i64]f32} =
--     defunc_4_map_res_adj_15248[0i64, 0i64 :+ 4i64 * 1i64, 0i64 :+ 16i64 * 1i64, 0i64 :+ 32i64 * 1i64]
--   let {zeroes__15344 : [4i64][4i64][3i64][3i64]f32} =
--     #[sequential]
--     replicate([4i64][4i64][3i64][3i64], 0.0f32)
--   let {tab_iota_15506 : [4i64]i64} =
--     iota64(4i64, 0i64, 1i64)
--   let {tab_iota_15510 : [3i64]i64} =
--     iota64(3i64, 0i64, 1i64)
--   let {tab_iota_15514 : [3i64]i64} =
--     iota64(3i64, 0i64, 1i64)
--   let {tab_iota_15631 : [4i64]i64} =
--     iota64(4i64, 0i64, 1i64)
--   let {tab_iota_15635 : [3i64]i64} =
--     iota64(3i64, 0i64, 1i64)
--   let {tab_iota_15639 : [3i64]i64} =
--     iota64(3i64, 0i64, 1i64)
--   let {withhacc_res_15675 : [4i64][4i64][3i64][3i64]f32} =
--     with_acc({([4i64][4i64][3i64][3i64], {zeroes__15344},
--               (\ {idx_15348 : i64, idx_15349 : i64, idx_15350 : i64, idx_15351 : i64, x_15345 : f32, y_15346 : f32}
--                 : {f32} ->
--                 let {binlam_res_15347 : f32} = fadd32(x_15345, y_15346)
--                 in {binlam_res_15347},
--               {0.0f32}))},
--     \ {acc_cert_p_15358 : unit, acc_p_15359 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})}
--       : {acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} ->
--       let {map_adjs_15671 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} =
--         map(4i64,
--             {iota_res_15126, x_15264, acc_p_15359},
--             \ {x_15267 : i64, map_adj_p_15266 : [16i64][32i64]f32, free_adj_p_15363 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})}
--               : {acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} ->
--               let {zeroes__15409 : [4i64][3i64][3i64]f32} =
--                 #[sequential]
--                 replicate([4i64][3i64][3i64], 0.0f32)
--               let {withhacc_res_15627 : [4i64][3i64][3i64]f32} =
--                 with_acc({([4i64][3i64][3i64], {zeroes__15409},
--                           (\ {idx_15413 : i64, idx_15414 : i64, idx_15415 : i64, x_15410 : f32, y_15411 : f32}
--                             : {f32} ->
--                             let {binlam_res_15412 : f32} = fadd32(x_15410, y_15411)
--                             in {binlam_res_15412},
--                           {0.0f32}))},
--                 \ {acc_cert_p_15416 : unit, acc_p_15417 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})}
--                   : {acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} ->
--                   let {map_adjs_15620 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} =
--                     map(16i64,
--                         {iota_res_15130, map_adj_p_15266, acc_p_15417},
--                         \ {x_15365 : i64, map_adj_p_15364 : [32i64]f32, free_adj_p_15420 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})}
--                           : {acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} ->
--                           let {j_15366 : i64} = add64(3i64, x_15365)
--                           let {i_p_m_t_s_15367 : i64} = add64(2i64, x_15365)
--                           let {zero_leq_i_p_m_t_s_15368 : bool} = sle64(0i64, i_p_m_t_s_15367)
--                           let {i_p_m_t_s_leq_w_15369 : bool} = slt64(i_p_m_t_s_15367, 16i64)
--                           let {zero_lte_i_15370 : bool} = sle64(0i64, x_15365)
--                           let {i_lte_j_15371 : bool} = sle64(x_15365, j_15366)
--                           let {y_15372 : bool} = logand(i_p_m_t_s_leq_w_15369, zero_lte_i_15370)
--                           let {y_15373 : bool} = logand(zero_leq_i_p_m_t_s_15368, y_15372)
--                           let {y_15374 : bool} = logand(i_lte_j_15371, y_15373)
--                           let {forwards_ok_15375 : bool} = logand(zero_lte_i_15370, y_15374)
--                           let {map_adjs_15569 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} =
--                             map(32i64,
--                                 {iota_res_15134, map_adj_p_15364, free_adj_p_15420},
--                                 \ {x_15422 : i64, map_adj_p_15421 : f32, free_adj_p_15457 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})}
--                                   : {acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} ->
--                                   let {j_15423 : i64} = add64(3i64, x_15422)
--                                   let {i_p_m_t_s_15424 : i64} = add64(2i64, x_15422)
--                                   let {zero_leq_i_p_m_t_s_15425 : bool} = sle64(0i64, i_p_m_t_s_15424)
--                                   let {i_p_m_t_s_leq_w_15426 : bool} = slt64(i_p_m_t_s_15424, 32i64)
--                                   let {zero_lte_i_15427 : bool} = sle64(0i64, x_15422)
--                                   let {i_lte_j_15428 : bool} = sle64(x_15422, j_15423)
--                                   let {y_15429 : bool} = logand(i_p_m_t_s_leq_w_15426, zero_lte_i_15427)
--                                   let {y_15430 : bool} = logand(zero_leq_i_p_m_t_s_15425, y_15429)
--                                   let {y_15431 : bool} = logand(i_lte_j_15428, y_15430)
--                                   let {forwards_ok_15432 : bool} = logand(zero_lte_i_15427, y_15431)
--                                   let {y_15433 : bool} = logand(forwards_ok_15375, forwards_ok_15432)
--                                   let {index_certs_15434 : unit} =
--                                     assert(y_15433, {"Index [", 0i64 : i64, ":", 1i64 : i64, ", ", 0i64 : i64, ":", 4i64 : i64, ", ", x_15365 : i64, ":", j_15366 : i64, ", ", x_15422 : i64, ":", j_15423 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 4i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "conv2d_dKrn.fut:18:4-22:28")
--                                   let {x_15435 : [4i64][3i64][3i64]f32} =
--                                     #{index_certs_15434}
--                                     arrA_11006[0i64, 0i64 :+ 4i64 * 1i64, x_15365 :+ 3i64 * 1i64, x_15422 :+ 3i64 * 1i64]
--                                   let {map_adjs_15505 : [4i64][3i64][3i64]f32} =
--                                     map(4i64,
--                                         {x_15435},
--                                         \ {x_15478 : [3i64][3i64]f32}
--                                           : {[3i64][3i64]f32} ->
--                                           let {map_adjs_15503 : [3i64][3i64]f32} =
--                                             map(3i64,
--                                                 {x_15478},
--                                                 \ {x_15488 : [3i64]f32}
--                                                   : {[3i64]f32} ->
--                                                   let {map_adjs_15501 : [3i64]f32} =
--                                                     map(3i64,
--                                                         {x_15488},
--                                                         \ {x_15495 : f32}
--                                                           : {f32} ->
--                                                           let {binop_y_adj_15499 : f32} = fmul32(map_adj_p_15421, x_15495)
--                                                           in {binop_y_adj_15499})
--                                                   in {map_adjs_15501})
--                                           in {map_adjs_15503})
--                                   let {tab_15521 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} =
--                                     map(4i64,
--                                         {tab_iota_15506, map_adjs_15505, free_adj_p_15457},
--                                         \ {i_15507 : i64, map_adjs_p_15508 : [3i64][3i64]f32, free_adj_p_p_15509 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})}
--                                           : {acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} ->
--                                           let {tab_15520 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} =
--                                             map(3i64,
--                                                 {tab_iota_15510, map_adjs_p_15508, free_adj_p_p_15509},
--                                                 \ {i_15511 : i64, map_adjs_p_p_15512 : [3i64]f32, free_adj_p_p_p_15513 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})}
--                                                   : {acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} ->
--                                                   let {tab_15519 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} =
--                                                     map(3i64,
--                                                         {tab_iota_15514, map_adjs_p_p_15512, free_adj_p_p_p_15513},
--                                                         \ {i_15515 : i64, map_adjs_p_p_p_15516 : f32, free_adj_p_p_p_p_15517 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})}
--                                                           : {acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} ->
--                                                           let {acc_15518 : acc(acc_cert_p_15416, [4i64][3i64][3i64], {f32})} =
--                                                             update_acc(free_adj_p_p_p_p_15517, {i_15507, i_15511, i_15515}, {map_adjs_p_p_p_15516})
--                                                           in {acc_15518})
--                                                   in {tab_15519})
--                                           in {tab_15520})
--                                   in {tab_15521})
--                           in {map_adjs_15569})
--                   in {map_adjs_15620})
--               let {tab_15646 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} =
--                 map(4i64,
--                     {tab_iota_15631, withhacc_res_15627, free_adj_p_15363},
--                     \ {i_15632 : i64, withhacc_res_p_15633 : [3i64][3i64]f32, free_adj_p_p_15634 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})}
--                       : {acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} ->
--                       let {tab_15645 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} =
--                         map(3i64,
--                             {tab_iota_15635, withhacc_res_p_15633, free_adj_p_p_15634},
--                             \ {i_15636 : i64, withhacc_res_p_p_15637 : [3i64]f32, free_adj_p_p_p_15638 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})}
--                               : {acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} ->
--                               let {tab_15644 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} =
--                                 map(3i64,
--                                     {tab_iota_15639, withhacc_res_p_p_15637, free_adj_p_p_p_15638},
--                                     \ {i_15640 : i64, withhacc_res_p_p_p_15641 : f32, free_adj_p_p_p_p_15642 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})}
--                                       : {acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} ->
--                                       let {free_adj_p_p_p_p_15643 : acc(acc_cert_p_15358, [4i64][4i64][3i64][3i64], {f32})} =
--                                         update_acc(free_adj_p_p_p_p_15642, {x_15267, i_15632, i_15636, i_15640}, {withhacc_res_p_p_p_15641})
--                                       in {free_adj_p_p_p_p_15643})
--                               in {tab_15644})
--                       in {tab_15645})
--               in {tab_15646})
--       in {map_adjs_15671})
--   let {defunc_10_vjp2_res_15169 : [1i64][4i64][16i64][32i64]f32} = defunc_4_map_res_15172
--   let {defunc_10_vjp2_res_15170 : [4i64][4i64][3i64][3i64]f32} = withhacc_res_15675
--   in {defunc_10_vjp2_res_15170}
-- }
--
--
