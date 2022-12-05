
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
        let arrAt = slice4 1 nAc nKh nKw arrA (iImg,  0, iBh, iBw)
        let arrKt = slice4 1 nAc nKh nKw arrK (iCout, 0,   0,   0)
        in  dot4 arrAt arrKt)


def indexz4
    [nN][nC][nH][nW]
    (iN: i64) (iC: i64) (iH: i64) (iW: i64)
    (a: [nN][nC][nH][nW]f32)
  :  f32
  = if   iN < nN && iC < nC && iH < nH && iW < nW
    then a[iN, iC, iH, iW]
    else 0


def conv2d_dInp
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
    (arrO: [nAi][nBc][nBh][nBw]f32)
 : ([nAi][nAc][nAh][nAw]f32)
 = vjp (\(a : [nAi][nAc][nAh][nAw]f32) : [nAi][nBc][nBh][nBw]f32 -> conv2d a arrK)
       arrA arrO


def conv2d_dInp_impl
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrK: [nBc][nAc][nKh][nKw]f32)
    (arrO: [nAi][nBc][nBh][nBw]f32)
 : ([nAi][nAc][nAh][nAw]f32)
  = let padh = nKh / 2
    let padw = nKw / 2
    in  tabulate_4d nAi nAc nAh nAw (\iImg iCinp iAh iAw ->
        sum (flatten_3d (tabulate_3d nBc nBh nBw (\iCout iBh iBw ->
          let iKh  = iAh - iBh + padh
          let iKw  = iAw - iBw + padw
          let xOut = arrO[iImg,  iCout, iBh, iBw]
          let xKrn = indexz4 iCout iCinp iKh iKw arrK
          in  xOut * xKrn))))

-- ------------------------------------------------------------------------------------------------

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
  let dI = conv2d_dInp aA aK aO
  in  dI

def test_1_impl =
  let aK : [nCout_1][nCinp_1][nKh_1][nKw_1]f32 = fill4 0.1
  let aO : [nImg_1] [nCout_1][nAh_1][nAw_1]f32 = fill4   1
  let dI : [nImg_1] [nCinp_1][nAh_1][nAw_1]f32 = conv2d_dInp_impl aK aO
  in  dI

def test1 =
  test_1_ad == test_1_impl

def main
    [nAi][nAc][nAh][nAw]
    [nBc][nKh][nKw]
    [nBh][nBw]
    (arrA: [nAi][nAc][nAh][nAw]f32)
    (arrK: [nBc][nAc][nKh][nKw]f32)
    (arrO: [nAi][nBc][nBh][nBw]f32)
 : [nAi][nAc][nAh][nAw]f32
 = conv2d_dInp arrA arrK arrO


-- # Trying dev -ad by itself fails in the AD implementation.
--   Inspection of the case requires that 'apply' needs to be appled to a primtive operator
--   not an arbitrary function like 'defunc_0_f_6753'.
--   Forcing inlining before running AD makes this case work.
--   We're not sure if this is always possible, or that code explosion can always be avoided.
--
-- $ futhark dev --type-check --ad conv2d_dInp.fut > dump/conv2d_dInp.txt
-- ------------------------------------------------------------------------------------------------
-- Internal compiler error (unhandled IO exception).
-- Please report this at https://github.com/diku-dk/futhark/issues
-- diffStm unhandled:
-- let {lam_res_9655 : [d_9626][nBc_9635][f_9639][f_9640]f32} =
--   apply defunc_0_f_6753(d_9626, nAc_9634, d_9628, d_9629, nBc_9635, nKh_9636, nKw_9637, f_9638, f_9639, f_9640, x_9654)
--   : {[d_9626][nBc_9635][f_9639][f_9640]f32}
-- CallStack (from HasCallStack):
--   error, called at src/Futhark/AD/Rev.hs:312:17 in fthrk-0.23.0-f26eba86:Futhark.AD.Rev
--


-- $ futhark dev --type-check --inline-aggressively --ad conv2d_dInp.fut > dump/conv2d_dInp.txt
-- ------------------------------------------------------------------------------------------------
-- entry("main",
--       {arrA: [][][][]f32,
--        arrK: [][][][]f32,
--        arrO: [][][][]f32},
--       {[][][][]f32})
--   entry_main (arrA_9742 : [1i64][4i64][16i64][32i64]f32,
--               arrK_9743 : [4i64][4i64][3i64][3i64]f32,
--               arrO_9744 : [1i64][4i64][16i64][32i64]f32)
--   : {[1i64][4i64][16i64][32i64]f32} = {
--   let {iota_res_13974 : [4i64]i64} =
--     iota64(4i64, 0i64, 1i64)
--   let {iota_res_13975 : [16i64]i64} =
--     iota64(16i64, 0i64, 1i64)
--   let {iota_res_13976 : [32i64]i64} =
--     iota64(32i64, 0i64, 1i64)
--   let {x_13979 : [1i64][4i64][16i64][32i64]f32} =
--     copy(arrA_9742)
--   let {defunc_4_map_res_adj_14037 : [1i64][4i64][16i64][32i64]f32} =
--     copy(arrO_9744)
--   let {defunc_3_map_res_13980 : [4i64][16i64][32i64]f32} =
--     map(4i64,
--         {iota_res_13974},
--         \ {x_13981 : i64}
--           : {[16i64][32i64]f32} ->
--           let {j_13982 : i64} = add64(1i64, x_13981)
--           let {zero_leq_i_p_m_t_s_13983 : bool} = sle64(0i64, x_13981)
--           let {i_p_m_t_s_leq_w_13984 : bool} = slt64(x_13981, 4i64)
--           let {i_lte_j_13985 : bool} = sle64(x_13981, j_13982)
--           let {y_13986 : bool} = logand(zero_leq_i_p_m_t_s_13983, i_p_m_t_s_leq_w_13984)
--           let {y_13987 : bool} = logand(zero_leq_i_p_m_t_s_13983, y_13986)
--           let {y_13988 : bool} = logand(i_lte_j_13985, y_13987)
--           let {forwards_ok_13989 : bool} = logand(zero_leq_i_p_m_t_s_13983, y_13988)
--           let {index_certs_13990 : unit} =
--             assert(forwards_ok_13989, {"Index [", x_13981 : i64, ":", j_13982 : i64, ", ", 0i64 : i64, ":", 4i64 : i64, ", ", 0i64 : i64, ":", 3i64 : i64, ", ", 0i64 : i64, ":", 3i64 : i64, "] out of bounds for array of shape [", 4i64 : i64, "][", 4i64 : i64, "][", 3i64 : i64, "][", 3i64 : i64, "]."}, "conv2d_dInp.fut:18:4-22:28")
--           let {as_transformed_row_13991 : [4i64][3i64][3i64]f32} =
--             #{index_certs_13990}
--             arrK_9743[x_13981, 0i64 :+ 4i64 * 1i64, 0i64 :+ 3i64 * 1i64, 0i64 :+ 3i64 * 1i64]
--           let {defunc_2_map_res_13992 : [16i64][32i64]f32} =
--             map(16i64,
--                 {iota_res_13975},
--                 \ {x_13993 : i64}
--                   : {[32i64]f32} ->
--                   let {j_13994 : i64} = add64(3i64, x_13993)
--                   let {i_p_m_t_s_13995 : i64} = add64(2i64, x_13993)
--                   let {zero_leq_i_p_m_t_s_13996 : bool} = sle64(0i64, i_p_m_t_s_13995)
--                   let {i_p_m_t_s_leq_w_13997 : bool} = slt64(i_p_m_t_s_13995, 16i64)
--                   let {zero_lte_i_13998 : bool} = sle64(0i64, x_13993)
--                   let {i_lte_j_13999 : bool} = sle64(x_13993, j_13994)
--                   let {y_14000 : bool} = logand(i_p_m_t_s_leq_w_13997, zero_lte_i_13998)
--                   let {y_14001 : bool} = logand(zero_leq_i_p_m_t_s_13996, y_14000)
--                   let {y_14002 : bool} = logand(i_lte_j_13999, y_14001)
--                   let {forwards_ok_14003 : bool} = logand(zero_lte_i_13998, y_14002)
--                   let {defunc_1_map_res_14004 : [32i64]f32} =
--                     map(32i64,
--                         {iota_res_13976},
--                         \ {x_14005 : i64}
--                           : {f32} ->
--                           let {j_14006 : i64} = add64(3i64, x_14005)
--                           let {i_p_m_t_s_14007 : i64} = add64(2i64, x_14005)
--                           let {zero_leq_i_p_m_t_s_14008 : bool} = sle64(0i64, i_p_m_t_s_14007)
--                           let {i_p_m_t_s_leq_w_14009 : bool} = slt64(i_p_m_t_s_14007, 32i64)
--                           let {zero_lte_i_14010 : bool} = sle64(0i64, x_14005)
--                           let {i_lte_j_14011 : bool} = sle64(x_14005, j_14006)
--                           let {y_14012 : bool} = logand(i_p_m_t_s_leq_w_14009, zero_lte_i_14010)
--                           let {y_14013 : bool} = logand(zero_leq_i_p_m_t_s_14008, y_14012)
--                           let {y_14014 : bool} = logand(i_lte_j_14011, y_14013)
--                           let {forwards_ok_14015 : bool} = logand(zero_lte_i_14010, y_14014)
--                           let {y_14016 : bool} = logand(forwards_ok_14003, forwards_ok_14015)
--                           let {index_certs_14017 : unit} =
--                             assert(y_14016, {"Index [", 0i64 : i64, ":", 1i64 : i64, ", ", 0i64 : i64, ":", 4i64 : i64, ", ", x_13993 : i64, ":", j_13994 : i64, ", ", x_14005 : i64, ":", j_14006 : i64, "] out of bounds for array of shape [", 1i64 : i64, "][", 4i64 : i64, "][", 16i64 : i64, "][", 32i64 : i64, "]."}, "conv2d_dInp.fut:18:4-22:28")
--                           let {x_14018 : [4i64][3i64][3i64]f32} =
--                             #{index_certs_14017}
--                             x_13979[0i64, 0i64 :+ 4i64 * 1i64, x_13993 :+ 3i64 * 1i64, x_14005 :+ 3i64 * 1i64]
--                           let {defunc_7_map_res_14019 : [4i64][3i64][3i64]f32} =
--                             map(4i64,
--                                 {x_14018, as_transformed_row_13991},
--                                 \ {x_14020 : [3i64][3i64]f32, zip_copy_transformed_row_14021 : [3i64][3i64]f32}
--                                   : {[3i64][3i64]f32} ->
--                                   let {defunc_4_map_res_14022 : [3i64][3i64]f32} =
--                                     map(3i64,
--                                         {x_14020, zip_copy_transformed_row_14021},
--                                         \ {x_14023 : [3i64]f32, x_14024 : [3i64]f32}
--                                           : {[3i64]f32} ->
--                                           let {defunc_1_map_res_14025 : [3i64]f32} =
--                                             map(3i64,
--                                                 {x_14023, x_14024},
--                                                 \ {x_14026 : f32, x_14027 : f32}
--                                                   : {f32} ->
--                                                   let {defunc_1_f_res_14028 : f32} = fmul32(x_14026, x_14027)
--                                                   in {defunc_1_f_res_14028})
--                                           in {defunc_1_map_res_14025})
--                                   in {defunc_4_map_res_14022})
--                           let {defunc_10_map_res_14029 : [1i64][4i64][3i64][3i64]f32} =
--                             replicate([1i64], defunc_7_map_res_14019)
--                           let {flatten_res_14030 : [36i64]f32} =
--                             reshape([36i64], defunc_10_map_res_14029)
--                           let {defunc_2_reduce_res_14031 : f32} =
--                             redomap(36i64,
--                                     {flatten_res_14030},
--                                     {\ {x_14032 : f32, x_14033 : f32}
--                                       : {f32} ->
--                                       let {defunc_1_op_res_14034 : f32} = fadd32(x_14032, x_14033)
--                                       in {defunc_1_op_res_14034},
--                                     {0.0f32}},
--                                     \ {x_14035 : f32}
--                                       : {f32} ->
--                                       {x_14035})
--                           in {defunc_2_reduce_res_14031})
--                   in {defunc_1_map_res_14004})
--           in {defunc_2_map_res_13992})
--   let {defunc_4_map_res_14036 : [1i64][4i64][16i64][32i64]f32} =
--     replicate([1i64], defunc_3_map_res_13980)
--   let {x_14053 : [4i64][16i64][32i64]f32} =
--     defunc_4_map_res_adj_14037[0i64, 0i64 :+ 4i64 * 1i64, 0i64 :+ 16i64 * 1i64, 0i64 :+ 32i64 * 1i64]
--   let {zeroes__14133 : [1i64][4i64][16i64][32i64]f32} =
--     #[sequential]
--     replicate([1i64][4i64][16i64][32i64], 0.0f32)
--   let {tab_iota_14311 : [4i64]i64} =
--     iota64(4i64, 0i64, 1i64)
--   let {tab_iota_14315 : [3i64]i64} =
--     iota64(3i64, 0i64, 1i64)
--   let {tab_iota_14319 : [3i64]i64} =
--     iota64(3i64, 0i64, 1i64)
--   let {withhacc_res_14464 : [1i64][4i64][16i64][32i64]f32} =
--     with_acc({([1i64][4i64][16i64][32i64], {zeroes__14133},
--               (\ {idx_14137 : i64, idx_14138 : i64, idx_14139 : i64, idx_14140 : i64, x_14134 : f32, y_14135 : f32}
--                 : {f32} ->
--                 let {binlam_res_14136 : f32} = fadd32(x_14134, y_14135)
--                 in {binlam_res_14136},
--               {0.0f32}))},
--     \ {acc_cert_p_14147 : unit, acc_p_14148 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})}
--       : {acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} ->
--       let {map_adjs_14460 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} =
--         map(4i64,
--             {iota_res_13974, x_14053, acc_p_14148},
--             \ {x_14056 : i64, map_adj_p_14055 : [16i64][32i64]f32, free_adj_p_14152 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})}
--               : {acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} ->
--               let {j_14057 : i64} = add64(1i64, x_14056)
--               let {zero_leq_i_p_m_t_s_14058 : bool} = sle64(0i64, x_14056)
--               let {i_p_m_t_s_leq_w_14059 : bool} = slt64(x_14056, 4i64)
--               let {i_lte_j_14060 : bool} = sle64(x_14056, j_14057)
--               let {y_14061 : bool} = logand(zero_leq_i_p_m_t_s_14058, i_p_m_t_s_leq_w_14059)
--               let {y_14062 : bool} = logand(zero_leq_i_p_m_t_s_14058, y_14061)
--               let {y_14063 : bool} = logand(i_lte_j_14060, y_14062)
--               let {forwards_ok_14064 : bool} = logand(zero_leq_i_p_m_t_s_14058, y_14063)
--               let {index_certs_14065 : unit} =
--                 assert(forwards_ok_14064, {"Index [", x_14056 : i64, ":", j_14057 : i64, ", ", 0i64 : i64, ":", 4i64 : i64, ", ", 0i64 : i64, ":", 3i64 : i64, ", ", 0i64 : i64, ":", 3i64 : i64, "] out of bounds for array of shape [", 4i64 : i64, "][", 4i64 : i64, "][", 3i64 : i64, "][", 3i64 : i64, "]."}, "conv2d_dInp.fut:18:4-22:28")
--               let {as_transformed_row_14066 : [4i64][3i64][3i64]f32} =
--                 #{index_certs_14065}
--                 arrK_9743[x_14056, 0i64 :+ 4i64 * 1i64, 0i64 :+ 3i64 * 1i64, 0i64 :+ 3i64 * 1i64]
--               let {map_adjs_14408 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} =
--                 map(16i64,
--                     {iota_res_13975, map_adj_p_14055, free_adj_p_14152},
--                     \ {x_14154 : i64, map_adj_p_14153 : [32i64]f32, free_adj_p_14208 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})}
--                       : {acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} ->
--                       let {map_adjs_14357 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} =
--                         map(32i64,
--                             {iota_res_13976, map_adj_p_14153, free_adj_p_14208},
--                             \ {x_14211 : i64, map_adj_p_14210 : f32, free_adj_p_14245 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})}
--                               : {acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} ->
--                               let {map_adjs_14293 : [4i64][3i64][3i64]f32} =
--                                 map(4i64,
--                                     {as_transformed_row_14066},
--                                     \ {zip_copy_transformed_row_14268 : [3i64][3i64]f32}
--                                       : {[3i64][3i64]f32} ->
--                                       let {map_adjs_14291 : [3i64][3i64]f32} =
--                                         map(3i64,
--                                             {zip_copy_transformed_row_14268},
--                                             \ {x_14278 : [3i64]f32}
--                                               : {[3i64]f32} ->
--                                               let {map_adjs_14289 : [3i64]f32} =
--                                                 map(3i64,
--                                                     {x_14278},
--                                                     \ {x_14285 : f32}
--                                                       : {f32} ->
--                                                       let {binop_x_adj_14287 : f32} = fmul32(map_adj_p_14210, x_14285)
--                                                       in {binop_x_adj_14287})
--                                               in {map_adjs_14289})
--                                       in {map_adjs_14291})
--                               let {tab_14328 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} =
--                                 map(4i64,
--                                     {tab_iota_14311, map_adjs_14293, free_adj_p_14245},
--                                     \ {i_14312 : i64, map_adjs_p_14313 : [3i64][3i64]f32, free_adj_p_p_14314 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})}
--                                       : {acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} ->
--                                       let {tab_14327 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} =
--                                         map(3i64,
--                                             {tab_iota_14315, map_adjs_p_14313, free_adj_p_p_14314},
--                                             \ {i_14316 : i64, map_adjs_p_p_14317 : [3i64]f32, free_adj_p_p_p_14318 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})}
--                                               : {acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} ->
--                                               let {index_14323 : i64} = add_nw64(x_14154, i_14316)
--                                               let {tab_14326 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} =
--                                                 map(3i64,
--                                                     {tab_iota_14319, map_adjs_p_p_14317, free_adj_p_p_p_14318},
--                                                     \ {i_14320 : i64, map_adjs_p_p_p_14321 : f32, free_adj_p_p_p_p_14322 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})}
--                                                       : {acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} ->
--                                                       let {index_14324 : i64} = add_nw64(x_14211, i_14320)
--                                                       let {free_adj_p_p_p_p_14325 : acc(acc_cert_p_14147, [1i64][4i64][16i64][32i64], {f32})} =
--                                                         update_acc(free_adj_p_p_p_p_14322, {0i64, i_14312, index_14323, index_14324}, {map_adjs_p_p_p_14321})
--                                                       in {free_adj_p_p_p_p_14325})
--                                               in {tab_14326})
--                                       in {tab_14327})
--                               in {tab_14328})
--                       in {map_adjs_14357})
--               in {map_adjs_14408})
--       in {map_adjs_14460})
--   let {defunc_10_vjp2_res_13977 : [1i64][4i64][16i64][32i64]f32} = defunc_4_map_res_14036
--   let {defunc_10_vjp2_res_13978 : [1i64][4i64][16i64][32i64]f32} = withhacc_res_14464
--   in {defunc_10_vjp2_res_13978}
-- }
--