
def main (n: i64) (c: i64) (h: i64) (w: i64): [n][c][h][w]i64
 = tabulate n (\i0 ->
   tabulate c (\i1 ->
   tabulate h (\i2 ->
   tabulate w (\i3 -> i0 * i1 * i2 * i3))))


-- $ futhark dev --gpu tabulate4.fut > dump/tabulate4.txt
-- ------------------------------------------------------------------------------------------------
--
-- let {segmap_group_size_7777 : i64} =
--   get_size(segmap_group_size_7587, group_size)
-- let {segmap_usable_groups_7778 : i64} =
--   sdiv_up64(100000000i64, segmap_group_size_7777)
-- let {defunc_2_map_res_7779 : [100i64][100i64][100i64][100i64]i64} =
--   segmap(thread; ; groups=segmap_usable_groups_7778; groupsize=segmap_group_size_7777)
--   (gtid_7780 < 100i64, gtid_7781 < 100i64, gtid_7782 < 100i64, gtid_7783 < 100i64) (~phys_tid_7784) : {i64} {
--     let {binop_y_7808 : i64}        = mul64(gtid_7780, gtid_7781)
--     let {index_primexp_7809 : i64}  = mul64(gtid_7782, binop_y_7808)
--     let {defunc_0_f_res_7787 : i64} = mul64(gtid_7783, index_primexp_7809)
--     return {returns defunc_0_f_res_7787}
--   }
--
-- entry("main",
--       {},
--       {[][][][]i64})
--   entry_main ()
--   : {[100i64][100i64][100i64][100i64]i64} = {
--   {defunc_2_map_res_7779}
-- }
--