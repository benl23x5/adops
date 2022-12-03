

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


def main
    (arrA: [1][4][16][32]f32)
    (arrK: [4][4][3][3]f32)
 : [1][4][16][32]f32
 = conv2d arrA arrK


-- $ futhark dev --gpu conv2d.fut > dump/conv2d.txt
-- ------------------------------------------------------------------------------------------------
--
--   entry("main",
--       {arrA: [][][][]f32,
--        arrK: [][][][]f32},
--       {[][][][]f32})
--
--   entry_main (arrA_9328 : [1i64][4i64][16i64][32i64]f32,
--               arrK_9329 : [4i64][4i64][3i64][3i64]f32)
--   : {[1i64][4i64][16i64][32i64]f32} = {
--   let {iota_res_13013 : [4i64]i64}  = iota64(4i64,  0i64, 1i64)
--   let {iota_res_13017 : [16i64]i64} = iota64(16i64, 0i64, 1i64)
--   let {iota_res_13021 : [32i64]i64} = iota64(32i64, 0i64, 1i64)
--
--   let {defunc_3_map_res_13058 : [4i64][16i64][32i64]f32} =
--     map(4i64, {iota_res_13013},
--         \ {x_13059 : i64} : {[16i64][32i64]f32} ->
--           let {j_13060 : i64}                   = add64(1i64, x_13059)
--           let {zero_leq_i_p_m_t_s_13061 : bool} = sle64(0i64, x_13059)
--           let {i_p_m_t_s_leq_w_13062 : bool}    = slt64(x_13059, 4i64)
--           let {i_lte_j_13063 : bool}            = sle64(x_13059, j_13060)
--           let {y_13064 : bool}                  = logand(zero_leq_i_p_m_t_s_13061, i_p_m_t_s_leq_w_13062)
--           let {y_13065 : bool}                  = logand(zero_leq_i_p_m_t_s_13061, y_13064)
--           let {y_13066 : bool}                  = logand(i_lte_j_13063, y_13065)
--           let {forwards_ok_13067 : bool}        = logand(zero_leq_i_p_m_t_s_13061, y_13066)
--           let {index_certs_13071 : unit} =
--             assert(forwards_ok_13067, {"Index [", x_13059 : i64, ":", j_13060 : i64, ", ", 0i64 : i64, ":", 4i64 : i64, ", ", 0i64 : i64, ":", 3i64 : i64, ", ", 0i64 : i64, ":", 3i64 : i64, "] out of bounds for array of shape [", 4i64 : i64, "][", 4i64 : i64, "][", 3i64 : i64, "][", 3i64 : i64, "]."}, "conv2d.fut:19:4-23:28")
--           let {as_transformed_row_13072 : [4i64][3i64][3i64]f32} =
--             #{index_certs_13071}
--             arrK_9329[x_13059, 0i64 :+ 4i64 * 1i64, 0i64 :+ 3i64 * 1i64, 0i64 :+ 3i64 * 1i64]
--
--           let {defunc_2_map_res_13073 : [16i64][32i64]f32} =
--             map(16i64, {iota_res_13017},
--                 \ {x_13074 : i64} : {[32i64]f32} ->
--                   let {j_13075 : i64}                   = add64(3i64, x_13074)
--                   let {i_p_m_t_s_13076 : i64}           = add64(2i64, x_13074)
--                   let {zero_leq_i_p_m_t_s_13077 : bool} = sle64(0i64, i_p_m_t_s_13076)
--                   let {i_p_m_t_s_leq_w_13078 : bool}    = slt64(i_p_m_t_s_13076, 16i64)
--                   let {zero_lte_i_13079 : bool}         = sle64(0i64, x_13074)
--                   let {i_lte_j_13080 : bool}            = sle64(x_13074, j_13075)
--                   let {y_13081 : bool}                  = logand(i_p_m_t_s_leq_w_13078, zero_lte_i_13079)
--                   let {y_13082 : bool}                  = logand(zero_leq_i_p_m_t_s_13077, y_13081)
--                   let {y_13083 : bool}                  = logand(i_lte_j_13080, y_13082)
--                   let {forwards_ok_13084 : bool}        = logand(zero_lte_i_13079, y_13083)
--
--                   let {defunc_1_map_res_13086 : [32i64]f32} =
--                     map(32i64, {iota_res_13021},
--                         \ {x_13087 : i64}: {f32} ->
--                           let {j_13088 : i64}                   = add64(3i64, x_13087)
--                           let {i_p_m_t_s_13089 : i64}           = add64(2i64, x_13087)
--                           let {zero_leq_i_p_m_t_s_13090 : bool} = sle64(0i64, i_p_m_t_s_13089)
--                           let {i_p_m_t_s_leq_w_13091 : bool}    = slt64(i_p_m_t_s_13089, 32i64)
--                           let {zero_lte_i_13092 : bool}         = sle64(0i64, x_13087)
--                           let {i_lte_j_13093 : bool}            = sle64(x_13087, j_13088)
--                           let {y_13094 : bool}                  = logand(i_p_m_t_s_leq_w_13091, zero_lte_i_13092)
--                           let {y_13095 : bool}                  = logand(zero_leq_i_p_m_t_s_13090, y_13094)
--                           let {y_13096 : bool}                  = logand(i_lte_j_13093, y_13095)
--                           let {forwards_ok_13097 : bool}        = logand(zero_lte_i_13092, y_13096)
--                           let {y_13100 : bool}                  = logand(forwards_ok_13084, forwards_ok_13097)
--                           let {index_certs_13102 : unit} =
--                             assert(y_13100, {"Index [", 0i64 : i64, ":", 1i64 : i64, ", ", 0i64 : i64, ":", 4i64 : i64, ", ", x_13074 : i64, ":", j_13075 : i64, ", ", x_13087 : i64, ":", j_13088 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 4i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "conv2d.fut:19:4-23:28")
--                           let {x_13103 : [4i64][3i64][3i64]f32} =
--                             #{index_certs_13102}
--                             arrA_9328[0i64, 0i64 :+ 4i64 * 1i64, x_13074 :+ 3i64 * 1i64, x_13087 :+ 3i64 * 1i64]
--
--                           let {defunc_7_map_res_13104 : [4i64][3i64][3i64]f32} =
--                             map(4i64, {x_13103, as_transformed_row_13072},
--                                 \ {x_13105 : [3i64][3i64]f32, zip_copy_transformed_row_13106 : [3i64][3i64]f32} : {[3i64][3i64]f32} ->
--                                   let {defunc_4_map_res_13107 : [3i64][3i64]f32} =
--                                     map(3i64, {x_13105, zip_copy_transformed_row_13106},
--                                         \ {x_13108 : [3i64]f32, x_13109 : [3i64]f32} : {[3i64]f32} ->
--                                           let {defunc_1_map_res_13110 : [3i64]f32} =
--                                             map(3i64, {x_13108, x_13109},
--                                                 \ {x_13111 : f32, x_13112 : f32} : {f32} ->
--                                                   let {defunc_1_f_res_13113 : f32} = fmul32(x_13111, x_13112)
--                                                   in  {defunc_1_f_res_13113})
--                                           in {defunc_1_map_res_13110})
--                                   in {defunc_4_map_res_13107})
--
--                           let {defunc_10_map_res_13114 : [1i64][4i64][3i64][3i64]f32} =
--                             replicate([1i64], defunc_7_map_res_13104)
--
--                           let {flatten_res_13115 : [36i64]f32} =
--                             reshape([36i64], defunc_10_map_res_13114)
--
--                           let {defunc_2_reduce_res_13116 : f32} =
--                             redomap(36i64, {flatten_res_13115},
--                                     {\ {x_13117 : f32, x_13118 : f32} : {f32} ->
--                                       let {defunc_1_op_res_13119 : f32} = fadd32(x_13117, x_13118)
--                                       in  {defunc_1_op_res_13119},
--                                     {0.0f32}},
--                                     \ {x_13120 : f32} : {f32} ->
--                                       {x_13120})
--                           in {defunc_2_reduce_res_13116})
--                   in {defunc_1_map_res_13086})
--           in {defunc_2_map_res_13073})
--
--   let {defunc_4_map_res_13048 : [1i64][4i64][16i64][32i64]f32} =
--     replicate([1i64], defunc_3_map_res_13058)
--   in {defunc_4_map_res_13048}
-- }
--