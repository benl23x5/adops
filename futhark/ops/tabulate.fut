
def main (n: i64) : [n]i64
 = tabulate n (\i -> i * 123)


-- $ futhark dev --gpu tabulate.fut > dump/tabulate.txt
-- ------------------------------------------------------------------------------------------------
--
-- let {segmap_group_size_6092 : i64} =
--   get_size(segmap_group_size_6084, group_size)
-- let {segmap_usable_groups_6093 : i64} =
--   sdiv_up64(100i64, segmap_group_size_6092)
-- let {defunc_1_map_res_6094 : [100i64]i64} =
--   segmap(thread; ; groups=segmap_usable_groups_6093; groupsize=segmap_group_size_6092)
--   (gtid_6095 < 100i64) (~phys_tid_6096) : {i64} {
--     let {defunc_0_f_res_6098 : i64} = mul64(123i64, gtid_6095)
--     return {returns defunc_0_f_res_6098}
--   }
--
-- entry("main",
--       {},
--       {[]i64})
--   entry_main ()
--   : {[100i64]i64} = {
--   {defunc_1_map_res_6094}
-- }


-- % futhark dev --type-check --inline-aggressively --ad --fuse-soacs --inline-aggressively tabulate.fut > dump/tabulate.txt
-- entry("main",
--       {n: i64},
--       {[]i64})
--   entry_main (n_6052 : i64)
--   : {[n_6052]i64} = {
--   let {bounds_invalid_upwards_6096 : bool} = slt64(n_6052, 0i64)
--   let {valid_6097 : bool} = not bounds_invalid_upwards_6096
--   let {range_valid_c_6098 : unit} =
--     assert(valid_6097, {"Range ", 0i64 : i64, "..", 1i64 : i64, "..<", n_6052 : i64, " is invalid."}, "/prelude/array.fut:95:3-10")
--   let {iota_res_6099 : [n_6052]i64} =
--     #{range_valid_c_6098}
--     iota64(n_6052, 0i64, 1i64)
--   let {defunc_1_map_res_6106 : [n_6052]i64} =
--     map(n_6052,
--         {iota_res_6099},
--         \ {x_6101 : i64}
--           : {i64} ->
--           let {defunc_0_f_res_6102 : i64} = mul64(123i64, x_6101)
--           in {defunc_0_f_res_6102})
--   let {dim_match_6103 : bool} = eq_i64(n_6052, n_6052)
--   let {empty_or_match_cert_6104 : unit} =
--     assert(dim_match_6103, {"Function return value does not match shape of declared return type."}, "tabulate.fut:2:1-3:29")
--   let {result_proper_shape_6105 : [n_6052]i64} =
--     #{empty_or_match_cert_6104}
--     coerce([n_6052], defunc_1_map_res_6106)
--   in {result_proper_shape_6105}
-- }
--