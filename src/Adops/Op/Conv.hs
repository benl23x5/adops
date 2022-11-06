
module Adops.Op.Conv where
import Adops.Array

-- | Unpadded full convolution,
--   where the output size is the same as the input size.
conv2d
 :: (Elem a, Num a)
 => Array4 a -> Array4 a -> Array4 a

conv2d arrK arrA
 = let  Shape4 nImgs  nCinpA nAh nAw = shape arrA
        Shape4 nCoutK nCinpK nKh nKw = shape arrK
        nCinp   = same nCinpA nCinpK
        shB     = Shape4 nImgs nCoutK nAh nAw
        shK1    = Shape4 1 nCinp nKh nKw
   in   build4 shB $ \(Index4 iImg iCout iBh iBw) ->
        let arrAt = slicez4 arrA (Index4 iImg  0 iBh iBw) shK1
            arrKt = slicez4 arrK (Index4 iCout 0 0   0)   shK1
        in  dot arrAt arrKt


-- | Padded full convolution,
--   where the output size depends on the input size and kernel size.
conv2d_pad
 :: (Elem a, Num a)
 => (Int, Int) -> Array4 a -> Array4 a -> Array4 a

conv2d_pad (nPh, nPw) arrK arrA
 = let  Shape4 nImgs  nCinpA nAh nAw = shape arrA
        Shape4 nCoutK nCinpK nKh nKw = shape arrK
        nCinp   = same nCinpA nCinpK
        nBh     = nAh + 2 * nPh - nKh + 1
        nBw     = nAw + 2 * nPw - nKw + 1
        shB     = Shape4 nImgs nCoutK nBh nBw
        shK1    = Shape4 1 nCinp nKh nKw
   in   build4 shB $ \(Index4 iImg iCout iBh iBw) ->
        let iFh   = iBh - nPh
            iFw   = iBw - nPw
            arrAt = slicez4 arrA (Index4 iImg  0 iFh iFw) shK1
            arrKt = slicez4 arrK (Index4 iCout 0 0   0)   shK1
        in  dot arrAt arrKt

