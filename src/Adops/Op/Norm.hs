
module Adops.Op.Norm where
import Adops.Array
import qualified Data.Vector.Unboxed as U

-- | Apply softmax to a rank-1 array of floats.
softmax :: Array Shape1 Float -> Array Shape1 Float
softmax (Array sh elts)
 = let  mx  = U.maximum elts
        den = U.sum $ U.map (\x -> exp(x - mx)) elts
        res = U.map (\x -> exp (x - mx) / den) elts
   in   Array sh res

