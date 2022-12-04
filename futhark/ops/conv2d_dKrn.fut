
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


def indexz4
    [nN][nC][nH][nW]
    (iN: i64) (iC: i64) (iH: i64) (iW: i64)
    (a: [nN][nC][nH][nW]f32)
  :  f32
  = if   iN < nN && iC < nC && iH < nH && iW < nW
    then a[iN, iC, iH, iW]
    else 0


def slicez4
    [nAi][nAc][nAh][nAw]
    (nKi: i64) (nKc: i64) (nKh: i64) (nKw: i64)
    (arrA:  [nAi][nAc][nAh][nAw]f32)
    (start: (i64, i64, i64, i64))
 :  ([nKi][nKc][nKh][nKw]f32)
 = tabulate nKi (\iN ->
    tabulate nKc (\iC ->
     tabulate nKh (\iH ->
      tabulate nKw (\iW ->
        let oN = start.0 + iN
        let oC = start.1 + iC
        let oH = start.2 + iH
        let oW = start.3 + iW
        in  indexz4 oN oC oH oW arrA))))


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
 = reduce (\x y -> x + y) 0
    (flatten_4d (mmap4 (\x y -> x * y) arrA arrB))


def sum(xs: []f32): f32
 = reduce (+) 0 xs


def conv2d
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
 : ([nAi][nBc][nBh][nBw]f32)
 = tabulate_4d nAi nBc nBh nBw (\iImg iCout iBh iBw ->
        let arrAt = slicez4 1 nAc nKh nKw arrA (iImg,  0, iBh, iBw)
        let arrKt = slicez4 1 nAc nKh nKw arrK (iCout, 0,   0,   0)
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


def conv2d_dKrn_impl
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrO: [nAi][nBc][nBh][nBw]f32)
 : ([nBc][nAc][nKh][nKw]f32)
 = tabulate_4d nBc nAc nKh nKw (\iCout iCinp iKh iKw ->
    sum (tabulate nAi (\iImg ->
      let arrOt = slicez4 1 1 nBh nBw arrO (iImg, iCout, 0,   0)
      let arrAt = slicez4 1 1 nBh nBw arrA (iImg, iCinp, iKh, iKw)
      in  dot4 arrOt arrAt)))

-- ------------------------------------------------------------------------------------------------
-- There is a 'futhark bench' that lets us run benchmarking stanzas:
-- https://futhark-lang.org/examples/benchmarking.html
-- But I can't see a way to "quasiquote" test inputs. It seems to want
-- only literals. Instead compile this file as a library and use the C
-- wrapper to grind it.

-- Wrap the derivatives in a flatten because futhark doesn't allow me to use
-- conv2d_dKrn_impl as an entry point. It says:
-- > Entry point functions must not be size-polymorphic in their return type.
-- However it was ok with conv2d_dkrn as an entry point. I assume the ad
-- transform did the magic.

def conv2d_dKrn_flat
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
    (arrO: [nAi][nBc][nBh][nBw]f32)
 : []f32
 = flatten_4d (conv2d_dKrn arrA arrK arrO)

def conv2d_dKrn_impl_flat
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
    (arrO: [nAi][nBc][nBh][nBw]f32)
 : []f32
 = let dK: [nBc][nAc][nKh][nKw]f32 = conv2d_dKrn_impl arrA arrO
   in  flatten_4d dK


def fill4
    [n][c][h][w]
    (value: f32)
  : [n][c][h][w]f32
  = unflatten_4d n c h w (replicate (n * c * h * w) (value : f32))

def nImg_1  : i64 = 1
def nCinp_1 : i64 = 2
def nAh_1   : i64 = 32
def nAw_1   : i64 = 32
def nCout_1 : i64 = 4
def nKh_1   : i64 = 3
def nKw_1   : i64 = 3

def test_1_ad =
  let aA : [nImg_1] [nCinp_1][nAh_1][nAw_1]f32 = fill4 0.5
  let aO : [nImg_1] [nCout_1][nAh_1][nAw_1]f32 = fill4   1
  let aK : [nCout_1][nCinp_1][nKh_1][nKw_1]f32 = fill4 0.1
  let dK = conv2d_dKrn aA aK aO
  in  dK

def test_1_impl =
  let aA : [nImg_1] [nCinp_1][nAh_1][nAw_1]f32 = fill4 0.5
  let aO : [nImg_1] [nCout_1][nAh_1][nAw_1]f32 = fill4   1
  let dK : [nCout_1][nCinp_1][nKh_1][nKw_1]f32 = conv2d_dKrn_impl aA aO
  in  dK

def test1 =
  test_1_ad == test_1_impl

def main
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
    (arrO: [nAi][nBc][nBh][nBw]f32)
 : [nBc][nAc][nKh][nKw]f32
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






-- $ futhark dev --type-check --inline-aggressively --ad --gpu conv2d_dKrn.fut > dump/conv2d_dKrn.txt
-- ------------------------------------------------------------------------------------------------
-- entry("main",
--       {arrA: [][][][]f32,
--        arrK: [][][][]f32,
--        arrO: [][][][]f32},
--       {[][][][]f32})
--   entry_main (nAi_11024 : i64,
--               nAc_11025 : i64,
--               nAh_11026 : i64,
--               nAw_11027 : i64,
--               nBc_11028 : i64,
--               nKh_11029 : i64,
--               nKw_11030 : i64,
--               nBh_11031 : i64,
--               nBw_11032 : i64,
--               arrA_11033 : [nAi_11024][nAc_11025][nAh_11026][nAw_11027]f32,
--               arrK_11034 : [nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32,
--               arrO_11035 : [nAi_11024][nBc_11028][nBh_11031][nBw_11032]f32)
--   : {[nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32} = {
--   let {empty_slice_15171 : bool} = eq_i64(nAc_11025, 0i64)
--   let {m_15172 : i64} = sub64(nAc_11025, 1i64)
--   let {zero_leq_i_p_m_t_s_15173 : bool} = sle64(0i64, m_15172)
--   let {i_p_m_t_s_leq_w_15174 : bool} = slt64(m_15172, nAc_11025)
--   let {i_lte_j_15175 : bool} = sle64(0i64, nAc_11025)
--   let {y_15176 : bool} = logand(zero_leq_i_p_m_t_s_15173, i_p_m_t_s_leq_w_15174)
--   let {y_15177 : bool} = logand(i_lte_j_15175, y_15176)
--   let {ok_or_empty_15178 : bool} = logor(empty_slice_15171, y_15177)
--   let {empty_slice_15179 : bool} = eq_i64(nKh_11029, 0i64)
--   let {m_15180 : i64} = sub64(nKh_11029, 1i64)
--   let {empty_slice_15181 : bool} = eq_i64(nKw_11030, 0i64)
--   let {m_15182 : i64} = sub64(nKw_11030, 1i64)
--   let {zeroes__15401 : [nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32} =
--     #[sequential]
--     replicate([nBc_11028][nAc_11025][nKh_11029][nKw_11030], 0.0f32)
--   let {y_18135 : i64} = mul_nw64(nBh_11031, nBw_11032)
--   let {y_18136 : i64} = mul_nw64(nBc_11028, y_18135)
--   let {nest_size_18137 : i64} = mul_nw64(nAi_11024, y_18136)
--   let {segmap_group_size_18138 : i64} =
--     get_size(segmap_group_size_17130, group_size)
--   let {segmap_usable_groups_18139 : i64} = sdiv_up64(nest_size_18137, segmap_group_size_18138)
--   let {index_certs_r_r_r_r_18140 : [nAi_11024][nBc_11028][nBh_11031][nBw_11032]unit} =
--     segmap(thread; ; groups=segmap_usable_groups_18139; groupsize=segmap_group_size_18138)
--     (gtid_18141 < nAi_11024, gtid_18142 < nBc_11028, gtid_18143 < nBh_11031, gtid_18144 < nBw_11032) (~phys_tid_18145) : {unit} {
--       let {index_primexp_18345 : i64} = add64(1i64, gtid_18141)
--       let {index_primexp_18356 : bool} = sle64(gtid_18141, index_primexp_18345)
--       let {index_primexp_18342 : i64} = add64(nKh_11029, gtid_18143)
--       let {binop_x_18347 : bool} = sle64(gtid_18143, index_primexp_18342)
--       let {cmpop_y_18348 : i64} = add64(m_15180, gtid_18143)
--       let {binop_x_18349 : bool} = sle64(0i64, cmpop_y_18348)
--       let {binop_y_18351 : bool} = slt64(cmpop_y_18348, nAh_11026)
--       let {binop_y_18352 : bool} = logand(binop_x_18349, binop_y_18351)
--       let {binop_y_18353 : bool} = logand(binop_x_18347, binop_y_18352)
--       let {index_primexp_18354 : bool} = logor(empty_slice_15179, binop_y_18353)
--       let {j_18153 : i64} = add64(nKw_11030, gtid_18144)
--       let {i_p_m_t_s_18154 : i64} = add64(m_15182, gtid_18144)
--       let {zero_leq_i_p_m_t_s_18155 : bool} = sle64(0i64, i_p_m_t_s_18154)
--       let {i_p_m_t_s_leq_w_18156 : bool} = slt64(i_p_m_t_s_18154, nAw_11027)
--       let {i_lte_j_18158 : bool} = sle64(gtid_18144, j_18153)
--       let {y_18160 : bool} = logand(zero_leq_i_p_m_t_s_18155, i_p_m_t_s_leq_w_18156)
--       let {y_18161 : bool} = logand(i_lte_j_18158, y_18160)
--       let {ok_or_empty_18163 : bool} = logor(empty_slice_15181, y_18161)
--       let {y_18164 : bool} = logand(ok_or_empty_18163, index_primexp_18356)
--       let {y_18165 : bool} = logand(y_18164, index_primexp_18354)
--       let {index_ok_18166 : bool} = logand(ok_or_empty_15178, y_18165)
--       let {index_certs_18167 : unit} =
--         assert(index_ok_18166, {"Index [", gtid_18141 : i64, ":", index_primexp_18345 : i64, ", ", 0i64 : i64, ":", nAc_11025 : i64, ", ", gtid_18143 : i64, ":", index_primexp_18342 : i64, ", ", gtid_18144 : i64, ":", j_18153 : i64, "] out of bounds for array of shape [", nAi_11024 : i64, "][", nAc_11025 : i64, "][", nAh_11026 : i64, "][", nAw_11027 : i64, "]."}, "conv2d_dKrn.fut:18:4-22:28")
--       return {returns index_certs_18167}
--     }
--   let {y_18169 : i64} = mul_nw64(nKh_11029, nKw_11030)
--   let {y_18170 : i64} = mul_nw64(nAc_11025, y_18169)
--   let {y_18171 : i64} = mul_nw64(nBw_11032, y_18170)
--   let {y_18172 : i64} = mul_nw64(nBh_11031, y_18171)
--   let {y_18173 : i64} = mul_nw64(nBc_11028, y_18172)
--   let {nest_size_18174 : i64} = mul_nw64(nAi_11024, y_18173)
--   let {segmap_group_size_18175 : i64} =
--     get_size(segmap_group_size_17100, group_size)
--   let {segmap_usable_groups_18176 : i64} = sdiv_up64(nest_size_18174, segmap_group_size_18175)
--   let {x_r_r_r_r_18177 : [nAi_11024][nBc_11028][nBh_11031][nBw_11032][nAc_11025][nKh_11029][nKw_11030]f32} =
--     segmap(thread; ; groups=segmap_usable_groups_18176; groupsize=segmap_group_size_18175)
--     (gtid_18181 < nAi_11024, gtid_18182 < nBc_11028, gtid_18183 < nBh_11031, gtid_18184 < nBw_11032, gtid_slice_18178 < nAc_11025, gtid_slice_18179 < nKh_11029, gtid_slice_18180 < nKw_11030) (~phys_tid_18185) : {f32} {
--       let {index_certs_18192 : unit} =
--         index_certs_r_r_r_r_18140[gtid_18181, gtid_18182, gtid_18183, gtid_18184]
--       let {slice_18193 : i64} = add_nw64(gtid_slice_18179, gtid_18183)
--       let {slice_18194 : i64} = add_nw64(gtid_slice_18180, gtid_18184)
--       let {v_18195 : f32} =
--         #{index_certs_18192}
--         arrA_11033[gtid_18181, gtid_slice_18178, slice_18193, slice_18194]
--       return {returns v_18195}
--     }
--   let {segmap_group_size_18230 : i64} =
--     get_size(segmap_group_size_16936, group_size)
--   let {num_groups_18231 : i64} =
--     calc_num_groups(nest_size_18174, segmap_num_groups_16938, segmap_group_size_18230)
--   let {y_18283 : i64} = mul_nw64(nBc_11028, y_18170)
--   let {nest_size_18284 : i64} = mul_nw64(nAi_11024, y_18283)
--   let {segmap_group_size_18285 : i64} =
--     get_size(segmap_group_size_16699, group_size)
--   let {num_groups_18286 : i64} =
--     calc_num_groups(nest_size_18284, segmap_num_groups_16701, segmap_group_size_18285)
--   let {binop_x_18369 : i64} = mul_nw64(nBc_11028, nBh_11031)
--   let {binop_x_18370 : i64} = mul_nw64(nBw_11032, binop_x_18369)
--   let {binop_x_18371 : i64} = mul_nw64(nAc_11025, binop_x_18370)
--   let {binop_x_18372 : i64} = mul_nw64(nKh_11029, binop_x_18371)
--   let {binop_y_18373 : i64} = mul_nw64(nKw_11030, binop_x_18372)
--   let {binop_x_18388 : i64} = mul_nw64(nAc_11025, y_18135)
--   let {binop_x_18389 : i64} = mul_nw64(nKh_11029, binop_x_18388)
--   let {binop_y_18390 : i64} = mul_nw64(nKw_11030, binop_x_18389)
--   let {binop_x_18428 : i64} = mul_nw64(nAc_11025, nBw_11032)
--   let {binop_x_18429 : i64} = mul_nw64(nKh_11029, binop_x_18428)
--   let {binop_y_18430 : i64} = mul_nw64(nKw_11030, binop_x_18429)
--   let {binop_x_18514 : i64} = mul_nw64(nAc_11025, nKh_11029)
--   let {binop_y_18515 : i64} = mul_nw64(nKw_11030, binop_x_18514)
--   let {binop_x_19764 : i64} = mul_nw64(nAc_11025, nBc_11028)
--   let {binop_x_19765 : i64} = mul_nw64(nKh_11029, binop_x_19764)
--   let {binop_y_19766 : i64} = mul_nw64(nKw_11030, binop_x_19765)
--   let {withhacc_res_16353 : [nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32} =
--     with_acc({([nBc_11028][nAc_11025][nKh_11029][nKw_11030], {zeroes__15401},
--               (\ {idx_15405 : i64, idx_15406 : i64, idx_15407 : i64, idx_15408 : i64, x_15402 : f32, y_15403 : f32}
--                 : {f32} ->
--                 let {binlam_res_15404 : f32} = fadd32(x_15402, y_15403)
--                 in {binlam_res_15404},
--               {0.0f32}))},
--     \ {acc_cert_p_15417 : unit, acc_p_15418 : acc(acc_cert_p_15417, [nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})}
--       : {acc(acc_cert_p_15417, [nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})} ->
--       let {zeroes__r_r_18075 : [nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32} =
--         replicate([nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030], 0.0f32)
--       let {withhacc_res_r_r_18082 : [nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32} =
--         with_acc({([nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030], {zeroes__r_r_18075},
--                   (\ {idx_18083 : i64, idx_18084 : i64, idx_18085 : i64, idx_18086 : i64, idx_18087 : i64, x_18088 : f32, y_18089 : f32}
--                     : {f32} ->
--                     let {binlam_res_18090 : f32} = fadd32(x_18088, y_18089)
--                     in {binlam_res_18090},
--                   {0.0f32}))},
--         \ {acc_cert_p_18091 : unit, acc_p_18092 : acc(acc_cert_p_18091, [nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})}
--           : {acc(acc_cert_p_18091, [nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})} ->
--           let {hist_dest_18357 : [nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32} =
--             replicate([nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030], 0.0f32)
--           let {hist_dest_upd_19745 : [nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32} =
--             seghist(thread; virtualise; groups=num_groups_18231; groupsize=segmap_group_size_18230)
--             (gtid_18361 < nest_size_18174) (~phys_tid_18241)
--             ([nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030], 1i64,
--             {hist_dest_18357},
--             {0.0f32},
--             ,
--             \ {x_18358 : f32, y_18359 : f32}
--               : {f32} ->
--               let {binlam_res_18360 : f32} = fadd32(x_18358, y_18359)
--               in {binlam_res_18360})
--             : {i64, i64, i64, i64, i64, f32} {
--               let {binop_x_18379 : i64} = squot64(gtid_18361, binop_y_18373)
--               let {binop_y_18385 : i64} = mul_nw64(binop_y_18373, binop_x_18379)
--               let {binop_x_18386 : i64} = sub_nw64(gtid_18361, binop_y_18385)
--               let {binop_x_18421 : i64} = squot64(binop_x_18386, binop_y_18390)
--               let {binop_y_18426 : i64} = mul_nw64(binop_y_18390, binop_x_18421)
--               let {binop_x_18427 : i64} = sub_nw64(binop_x_18386, binop_y_18426)
--               let {binop_x_18508 : i64} = squot64(binop_x_18427, binop_y_18430)
--               let {binop_y_18512 : i64} = mul_nw64(binop_y_18430, binop_x_18508)
--               let {binop_x_18513 : i64} = sub_nw64(binop_x_18427, binop_y_18512)
--               let {binop_x_18684 : i64} = squot64(binop_x_18513, binop_y_18515)
--               let {binop_y_18687 : i64} = mul_nw64(binop_y_18515, binop_x_18684)
--               let {binop_x_18688 : i64} = sub_nw64(binop_x_18513, binop_y_18687)
--               let {binop_x_19037 : i64} = squot64(binop_x_18688, y_18169)
--               let {binop_y_19039 : i64} = mul_nw64(y_18169, binop_x_19037)
--               let {binop_x_19040 : i64} = sub_nw64(binop_x_18688, binop_y_19039)
--               let {binop_x_19743 : i64} = squot64(binop_x_19040, nKw_11030)
--               let {binop_y_19744 : i64} = mul_nw64(nKw_11030, binop_x_19743)
--               let {gtid_18240 : i64} = sub_nw64(binop_x_19040, binop_y_19744)
--               let {map_adj_p_18244 : f32} =
--                 arrO_11035[binop_x_18379, binop_x_18421, binop_x_18508, binop_x_18684]
--               let {x_18247 : f32} =
--                 x_r_r_r_r_18177[binop_x_18379, binop_x_18421, binop_x_18508, binop_x_18684, binop_x_19037, binop_x_19743, gtid_18240]
--               let {binop_y_adj_18250 : f32} = fmul32(map_adj_p_18244, x_18247)
--               return {returns binop_x_18379, returns binop_x_18421, returns binop_x_19037, returns binop_x_19743, returns gtid_18240, returns binop_y_adj_18250}
--             }
--           let {withacc_inter_18233 : acc(acc_cert_p_18091, [nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})} =
--             segmap(thread; virtualise; groups=num_groups_18231; groupsize=segmap_group_size_18230)
--             (gtid_19747 < nAi_11024, gtid_19748 < nBc_11028, gtid_19749 < nAc_11025, gtid_19750 < nKh_11029, gtid_19751 < nKw_11030) (~phys_tid_19746) : {acc(acc_cert_p_18091, [nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})} {
--               let {hist_dest_upd_elem_19752 : f32} =
--                 hist_dest_upd_19745[gtid_19747, gtid_19748, gtid_19749, gtid_19750, gtid_19751]
--               let {acc_p_upd_19753 : acc(acc_cert_p_18091, [nAi_11024][nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})} =
--                 update_acc(acc_p_18092, {gtid_19747, gtid_19748, gtid_19749, gtid_19750, gtid_19751}, {hist_dest_upd_elem_19752})
--               return {returns acc_p_upd_19753}
--             }
--           in {withacc_inter_18233})
--       let {hist_dest_19754 : [nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32} =
--         replicate([nBc_11028][nAc_11025][nKh_11029][nKw_11030], 0.0f32)
--       let {hist_dest_upd_19971 : [nBc_11028][nAc_11025][nKh_11029][nKw_11030]f32} =
--         seghist(thread; virtualise; groups=num_groups_18286; groupsize=segmap_group_size_18285)
--         (gtid_19758 < nest_size_18284) (~phys_tid_18294)
--         ([nBc_11028][nAc_11025][nKh_11029][nKw_11030], 1i64,
--         {hist_dest_19754},
--         {0.0f32},
--         ,
--         \ {x_19755 : f32, y_19756 : f32}
--           : {f32} ->
--           let {binlam_res_19757 : f32} = fadd32(x_19755, y_19756)
--           in {binlam_res_19757})
--         : {i64, i64, i64, i64, f32} {
--           let {binop_x_19770 : i64} = squot64(gtid_19758, binop_y_19766)
--           let {binop_y_19774 : i64} = mul_nw64(binop_y_19766, binop_x_19770)
--           let {binop_x_19775 : i64} = sub_nw64(gtid_19758, binop_y_19774)
--           let {binop_x_19798 : i64} = squot64(binop_x_19775, binop_y_18515)
--           let {binop_y_19801 : i64} = mul_nw64(binop_y_18515, binop_x_19798)
--           let {binop_x_19802 : i64} = sub_nw64(binop_x_19775, binop_y_19801)
--           let {binop_x_19855 : i64} = squot64(binop_x_19802, y_18169)
--           let {binop_y_19857 : i64} = mul_nw64(y_18169, binop_x_19855)
--           let {binop_x_19858 : i64} = sub_nw64(binop_x_19802, binop_y_19857)
--           let {binop_x_19969 : i64} = squot64(binop_x_19858, nKw_11030)
--           let {binop_y_19970 : i64} = mul_nw64(nKw_11030, binop_x_19969)
--           let {gtid_18293 : i64} = sub_nw64(binop_x_19858, binop_y_19970)
--           let {withhacc_res_p_p_p_18299 : f32} =
--             withhacc_res_r_r_18082[binop_x_19770, binop_x_19798, binop_x_19855, binop_x_19969, gtid_18293]
--           return {returns binop_x_19798, returns binop_x_19855, returns binop_x_19969, returns gtid_18293, returns withhacc_res_p_p_p_18299}
--         }
--       let {map_adjs_18288 : acc(acc_cert_p_15417, [nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})} =
--         segmap(thread; virtualise; groups=num_groups_18286; groupsize=segmap_group_size_18285)
--         (gtid_19973 < nBc_11028, gtid_19974 < nAc_11025, gtid_19975 < nKh_11029, gtid_19976 < nKw_11030) (~phys_tid_19972) : {acc(acc_cert_p_15417, [nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})} {
--           let {hist_dest_upd_elem_19977 : f32} =
--             hist_dest_upd_19971[gtid_19973, gtid_19974, gtid_19975, gtid_19976]
--           let {acc_p_upd_19978 : acc(acc_cert_p_15417, [nBc_11028][nAc_11025][nKh_11029][nKw_11030], {f32})} =
--             update_acc(acc_p_15418, {gtid_19973, gtid_19974, gtid_19975, gtid_19976}, {hist_dest_upd_elem_19977})
--           return {returns acc_p_upd_19978}
--         }
--       in {map_adjs_18288})
--   in {withhacc_res_16353}
-- }
--