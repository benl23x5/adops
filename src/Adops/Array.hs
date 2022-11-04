{-# OPTIONS -fno-warn-missing-methods #-}
module Adops.Array where
import Adops.Shape
import Adops.Elem
import qualified Data.Vector.Unboxed    as U


------------------------------------------------------------------------------
data Array sh a
 = Array sh (U.Vector a)
 deriving Show

type Array1 a = Array Shape1 a
type Array2 a = Array Shape2 a
type Array3 a = Array Shape3 a
type Array4 a = Array Shape4 a

instance IsShape sh => HasShape (Array sh a) where
 type Shape (Array sh a) = sh
 shape  (Array sh _) = sh
 extent (Array sh a) = size sh


------------------------------------------------------------------------------
instance (IsShape sh, U.Unbox a, Num a) => Num (Array sh a) where
 (+) (Array sh1 elems1) (Array sh2 elems2)
  | sh1 == sh2
  = Array sh1 (U.zipWith (+) elems1 elems2)
  | otherwise   = error "shape mismatch"

 (*) (Array sh1 elems1) (Array sh2 elems2)
  | sh1 == sh2
  = Array sh1 (U.zipWith (*) elems1 elems2)
  | otherwise   = error "shape mismatch"


------------------------------------------------------------------------------
fill    :: (IsShape sh, U.Unbox a)
        => sh -> a -> Array sh a
fill sh x
 = Array sh (U.replicate (size sh) x)

floats :: IsShape sh => sh -> Float -> Array sh Float
floats = fill



------------------------------------------------------------------------------
build1  :: Elem a
        => Shape1 -> (Index1 -> a) -> Array1 a
build1 sh make
 = Array sh $ U.generate (size sh) (\lix -> make $ fromLinear sh lix)

build1f :: Shape1 -> (Index1 -> Float) -> Array1 Float
build1f = build1


build4  :: Elem a
        => Shape4 -> (Index4 -> a) -> Array4 a
build4 sh make
 = Array sh $ U.generate (size sh) (\lix -> make $ fromLinear sh lix)

build4f :: Shape4 -> (Index4 -> Float) -> Array4 Float
build4f = build4


------------------------------------------------------------------------------
index4 :: Elem a
        => Array4 a -> Index4 -> a
index4 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise    = error "out of range"

indexz4 :: Elem a
        => Array4 a -> Index4 -> a
indexz4 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise    = zero


------------------------------------------------------------------------------
zipWith4
        :: (Elem a, Elem b, Elem c)
        => (a -> b -> c) -> Array4 a -> Array4 b -> Array4 c
zipWith4 f arrA arrB
 = build4 (shape arrA) $ \ix -> f (index4 arrA ix) (index4 arrB ix)


------------------------------------------------------------------------------
slicez4 :: Elem a
        => Array4 a -> Index4 -> Shape4 -> Array4 a
slicez4 arr ixBase shResult
 = build4 shResult $ \ixResult -> indexz4 arr (ixBase + ixResult)


------------------------------------------------------------------------------
sumAll  :: (Elem a, Num a)
        => Array sh a -> a
sumAll (Array _ elems)
 = U.sum elems


------------------------------------------------------------------------------
dot     :: (Elem a, Num a)
        => Array sh a -> Array sh a -> a
dot (Array sh1 elems1) (Array sh2 elems2)
 = U.sum $ U.zipWith (*) elems1 elems2


------------------------------------------------------------------------------
same    :: Eq a => a -> a -> a
same x1 x2
 | x1 == x2     = x1
 | otherwise    = error "not the same"

check   :: Bool -> a -> a
check True x = x
check False x = error "check failed"


------------------------------------------------------------------------------
-- Unpadded full convolution,
-- where the output size is the same as the input size.
conv2d  :: (Elem a, Num a)
        => Array4 a -> Array4 a -> Array4 a
conv2d arrA arrK
 = let  (Shape4 nImgs  nCinpA nAh nAw) = shape arrA
        (Shape4 nCoutK nCinpK nKh nKw) = shape arrK
        nCinp   = same nCinpA nCinpK
        shB     = Shape4 nImgs nCoutK nAh nAw
        shK1    = Shape4 1 nCinp nKh nKw
   in   build4 shB $ \(Index4 iImg iCout iBh iBw) ->
        let arrAt = slicez4 arrA (Index4 iImg  0 iBh iBw) shK1
            arrKt = slicez4 arrK (Index4 iCout 0 0   0)   shK1
        in  dot arrAt arrKt


-- Padded full convolution,
-- where the output size depends on the input size and kernel size.
conv2d_pad
        :: (Elem a, Num a)
        => (Int, Int) -> Array4 a -> Array4 a -> Array4 a

conv2d_pad (nPh, nPw) arrA arrK
 = let  (Shape4 nImgs  nCinpA nAh nAw) = shape arrA
        (Shape4 nCoutK nCinpK nKh nKw) = shape arrK
   in   check (nCinpA == nCinpK) $
        let nCinp   = nCinpA
            nBh     = nAh + 2 * nPh - nKh + 1
            nBw     = nAw + 2 * nPw - nKw + 1
            shB     = Shape4 nImgs nCoutK nBh nBw
            shK1    = Shape4 1 nCinp nKh nKw
        in  build4 shB $ \(Index4 iImg iCout iBh iBw) ->
            let iFh   = iBh - nPh
                iFw   = iBw - nPw
                arrAt = slicez4 arrA (Index4 iImg  0 iFh iFw) shK1
                arrKt = slicez4 arrK (Index4 iCout 0 0   0)   shK1
            in  dot arrAt arrKt


------------------------------------------------------------------------------
-- Disparity cost volume.
costVolumeLR
        :: (Elem a, Num a)
        => Int -> Int -> Array4 a -> Array4 a -> Array4 a

costVolumeLR iStart count arrL arrR
 = check (shape arrL == shape arrR)
 $ let  (Shape4 nImgs nChas nRows nCols) = shape arrL
        shOut = Shape4 nImgs count nRows nCols

   in   build4 shOut $ \(Index4 iImg iDisp iRow iCol) ->
        let arrVecL = build1 (Shape1 nChas) $ \(Index1 iCha) ->
                        index4  arrL (Index4 iImg iCha iRow iCol)

            iSrc = iCol - iStart - iDisp
            arrVecR = build1 (Shape1 nChas) $ \(Index1 iCha) ->
                        indexz4 arrR (Index4 iImg iCha iRow iSrc)

        in sumAll (abs (arrVecL - arrVecR))

