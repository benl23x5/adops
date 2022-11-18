
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


-- | Apply batch normalisation to a rank-3 array in CHW order, given
--   pre-computed 'scale' (gamma) and 'bias' (beta) parameters.
batchnorm
 :: Array1 Float -> Array1 Float
 -> Array3 Float -> Array3 Float
batchnorm aScale aBias aInput =
  build3 (shape aInput) $ \ix@(Index3 iImg iCha iElem) ->
    let input = index3 aInput ix
        scale = index1 aScale (Index1 iCha)
        bias  = index1 aBias  (Index1 iCha)
    in  input * scale + bias


-- | Compute the batch normalisation scale and bias from learned parameters.
batchnorm_params
 :: Array1 Float -> Array1 Float
 -> Array1 Float -> Array1 Float
 -> Float -> (Array1 Float, Array1 Float)
batchnorm_params aGamma aBeta aMean aVar eps =
  let Shape1 nCha = shape aGamma
      build f =
        build1 (Shape1 nCha) $ \iCha ->
          let gamma = index1 aGamma iCha
              beta  = index1 aBeta  iCha
              mean  = index1 aMean  iCha
              var   = index1 aVar   iCha
              sd    = sqrt (var + eps)
          in  f gamma beta mean sd
      scale gamma _ _ sd =
        gamma / sd
      bias gamma beta mean sd =
        beta - gamma * mean / sd
  in  (build scale, build bias)


-- | Normalize an array so that [0,1] represents the full range.
rescaleFull01 :: Array sh Float -> Array sh Float
rescaleFull01 arr
 = let  min   = minAll arr
        max   = maxAll arr
        delta = max - min
        scale x
         | delta == 0   = 0
         | otherwise    = (x - min) / delta
   in   mapAll scale arr
