
def main : [100]i64
 = tabulate 100 (\i -> i * 123)


-- $ futhark dev --gpu tabulate.fut > dump/tabulate.txt
-- ------------------------------------------------------------------
-- types {
--
-- }
--
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
