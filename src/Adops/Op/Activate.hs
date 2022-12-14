
module Adops.Op.Activate where
import Adops.Array

-- Rectified linear unit.
relu :: (IsShape sh, Elem a, Ord a, Num a) => Array sh a -> Array sh a
relu = mapAll (\x -> if x < 0 then 0 else x)
{-# INLINE relu #-}