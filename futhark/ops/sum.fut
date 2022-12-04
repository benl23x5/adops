
def sum(xs: []f32): f32
 = reduce (+) 0 xs

def sum_dInp(xs: []f32): []f32
 = vjp sum xs 1

def main(arr: [32]f32): [32]f32
 = sum_dInp arr

-- $ futhark dev --type-check --inline-aggressively --ad sum.fut > dump/sum.txt
-- -----------------------------------------------------------------------------
-- entry("main",
--       {arr: []f32},
--       {[]f32})
--   entry_main (arr_6135 : [32i64]f32)
--   : {[32i64]f32} = {
--   let {x_6197 : [32i64]f32} =
--     copy(arr_6135)
--   let {defunc_2_reduce_res_adj_6203 : f32} = 1.0f32
--   let {defunc_2_reduce_res_6198 : f32} =
--     redomap(32i64,
--             {x_6197},
--             {\ {x_6199 : f32, x_6200 : f32}
--               : {f32} ->
--               let {defunc_1_op_res_6201 : f32} = fadd32(x_6199, x_6200)
--               in {defunc_1_op_res_6201},
--             {0.0f32}},
--             \ {x_6202 : f32}
--               : {f32} ->
--               {x_6202})
--   let {defunc_2_reduce_res_adj_rep_6204 : [32i64]f32} =
--     replicate([32i64], defunc_2_reduce_res_adj_6203)
--   let {defunc_3_vjp2_res_6195 : f32} = defunc_2_reduce_res_6198
--   let {defunc_3_vjp2_res_6196 : [32i64]f32} = defunc_2_reduce_res_adj_rep_6204
--   in {defunc_3_vjp2_res_6196}
-- }
