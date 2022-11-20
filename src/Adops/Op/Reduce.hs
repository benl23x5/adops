
module Adops.Op.Reduce where
import Adops.Array
import qualified Data.Vector.Unboxed as U

max
  :: (IsShape sh, U.Unbox a, Ord a)
  => Array sh a -> a
max (Array _ u) =
  U.maximum u
{-# INLINE max #-}