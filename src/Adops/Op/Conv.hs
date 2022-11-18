
module Adops.Op.Conv where
import Adops.Array
import Debug.Trace

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

-- | Padded full convolution with unit stride over the volumetric dimensions
--   of a rank-5 array in NCDHW order.
--
conv3d_pad
  :: (Elem a, Num a)
  => (Int, Int, Int) -> Array5 a -> Array5 a -> Array5 a

conv3d_pad (nPd, nPh, nPw) arrK arrA
 = let  Shape5 nImgs  nCinpA nAd nAh nAw = shape arrA
        Shape5 nCoutK nCinpK nKd nKh nKw = shape arrK
        nCinp   = same nCinpA nCinpK
        nBd     = nAd + 2 * nPd - nKd + 1
        nBh     = nAh + 2 * nPh - nKh + 1
        nBw     = nAw + 2 * nPw - nKw + 1
        shB     = Shape5 nImgs nCoutK nBd nBh nBw
        shK1    = Shape5 1     nCinp  nKd nKh nKw
   in   build5 shB $ \(Index5 iImg iCout iBd iBh iBw) ->
        let iFd   = iBd - nPd
            iFh   = iBh - nPh
            iFw   = iBw - nPw
            arrAt = slicez5 arrA (Index5 iImg  0 iFd iFh iFw) shK1
            arrKt = slicez5 arrK (Index5 iCout 0 0   0   0)   shK1
        in  dot arrAt arrKt


-- | Padded channel-wise convolution with unit stride over the volumetric
--   dimensions of a rank-5 array in NCDHW order. This is the first part of a
--   separable convolution.
--
conv3d_chan
  :: (Elem a, Num a, Show a)
  => (Int, Int, Int) -> Array5 a -> Array1 a -> Array5 a -> Array5 a

conv3d_chan (nPd, nPh, nPw) arrK arrB arrA
 = let  Shape5 nImgs  nCinpA nAd nAh nAw = shape arrA
        Shape5 nCoutK      1 nKd nKh nKw = shape arrK
        nCinp   = nCinpA
        nBd     = nAd + 2 * nPd - nKd + 1
        nBh     = nAh + 2 * nPh - nKh + 1
        nBw     = nAw + 2 * nPw - nKw + 1
        shB     = Shape5 nImgs nCoutK nBd nBh nBw
        shK1    = Shape5     1      1 nKd nKh nKw
   in   build5 shB $ \(Index5 iImg iCout iBd iBh iBw) ->
        let iFd   = iBd - nPd
            iFh   = iBh - nPh
            iFw   = iBw - nPw
            arrAt = slicez5 arrA (Index5 iImg  iCout iFd iFh iFw) shK1
            arrKt = slice5  arrK (Index5 iCout 0       0   0   0) shK1
        in  dot arrAt arrKt + index1 arrB (Index1 iCout)
{-# INLINE conv3d_chan #-}

-- | Padded point-wise convolution with unit stride over the volumetric
--   dimensions of a rank-5 array in NCDHW order. This is the second part of a
--   separable convolution.
--
conv3d_point
  :: (Elem a, Num a)
  => (Int, Int, Int) -> Array5 a -> Array1 a -> Array5 a -> Array5 a

conv3d_point (nPd, nPh, nPw) arrK arrB arrA
 = let  Shape5 nImgs  nCinpA nAd nAh nAw = shape arrA
        Shape5 nCoutK nCinpK   1   1   1 = shape arrK
        nCinp   = same nCinpA nCinpK
        nBd     = nAd + 2 * nPd
        nBh     = nAh + 2 * nPh
        nBw     = nAw + 2 * nPw
        shB     = Shape5 nImgs nCoutK nBd nBh nBw
        shK1    = Shape5     1 nCinp    1   1   1
   in   build5 shB $ \(Index5 iImg iCout iBd iBh iBw) ->
        let iFd   = iBd - nPd
            iFh   = iBh - nPh
            iFw   = iBw - nPw
            arrAt = slicez5 arrA (Index5 iImg  0 iFd iFh iFw) shK1
            arrKt = slice5  arrK (Index5 iCout 0   0   0   0) shK1
        in  dot arrAt arrKt + index1 arrB (Index1 iCout)
{-# INLINE conv3d_point #-}


--------------------------------------------------------------------------------
-- | Derivative of unpadded full convolution with respect to the input image.
--
conv2d_dInp
 :: (Elem a, Num a)
 => Array4 a -> Array4 a -> Array4 a

conv2d_dInp arrK arrB
 = let  Shape4 nBimg nBchan nBh nBw = shape arrB
        Shape4 nKimg nKchan nKh nKw = shape arrK
        nCinp   = nKchan
        nImgs   = same nBchan nKimg
        shA     = Shape4 nImgs nCinp nBh nBw
        shK1    = Shape4 1     nCinp nKh nKw
   in   build4 shA $ \(Index4 iImg iCinp iAh iAw) ->
        let arrBt = slicez4 arrB (Index4 iImg  0 iAh iAw) shK1
            arrKt = slicez4 arrK (Index4 iCinp 0 0   0)   shK1
        in  dot arrBt arrKt

