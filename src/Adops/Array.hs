{-# OPTIONS -fno-warn-missing-methods #-}
module Adops.Array
        ( module Adops.Array.Shape
        , module Adops.Array.Elem
        , Array(..)
        , Array1, Array2, Array3, Array4
        , IsShape(..)
        , fill, floats
        , build1, build4
        , index4, indexz4
        , slicez4
        , zipWith4
        , sumAll
        , dot
        , same, check)
where
import Adops.Array.Shape
import Adops.Array.Elem
import qualified Data.Vector.Unboxed    as U


------------------------------------------------------------------------------
-- | Multidimensional array.
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
-- | Construct an array of the given shape where all the elements have
--   the same value.
fill    :: (IsShape sh, U.Unbox a)
        => sh -> a -> Array sh a
fill sh x
 = Array sh $ U.replicate (size sh) x

-- | Alias for `fill` that constraints the element type to be `Float`.
floats :: IsShape sh => sh -> Float -> Array sh Float
floats = fill



------------------------------------------------------------------------------
-- | Build an array of the given rank-1 shape,
--   given an function to produce the element at each index.
build1  :: Elem a => Shape1 -> (Index1 -> a) -> Array1 a
build1 sh make
 = Array sh $ U.generate (size sh) (\lix -> make $ fromLinear sh lix)

-- | Alias for `build1` that constrains the element type to be `Float`.
build1f :: Shape1 -> (Index1 -> Float) -> Array1 Float
build1f = build1


-- | Build an array of the given rank-1 shape,
--   given an function to produce the element at each index.
build4  :: Elem a => Shape4 -> (Index4 -> a) -> Array4 a
build4 sh make
 = Array sh $ U.generate (size sh) (\lix -> make $ fromLinear sh lix)

-- | Alias for `build4` that constrains the element type to be `Float`.
build4f :: Shape4 -> (Index4 -> Float) -> Array4 Float
build4f = build4


------------------------------------------------------------------------------
-- | Retrieve the element at the given index,
--   throwing error on out of range indices.
index4  :: Elem a => Array4 a -> Index4 -> a
index4 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise    = error "out of range"

-- | Retrieve the element at the given index,
--   returning zero for out of range indices.
indexz4 :: Elem a => Array4 a -> Index4 -> a
indexz4 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise    = zero


------------------------------------------------------------------------------
-- | Slice a section out of a rank-4 array,
--   given a base offset and shape of the section.
--
--   If the slice extends out side the source array then the corresponding
--   elements are set to zero.
slicez4 :: Elem a
        => Array4 a -> Index4 -> Shape4 -> Array4 a
slicez4 arr ixBase shResult
 = build4 shResult $ \ixResult -> indexz4 arr (ixBase + ixResult)


------------------------------------------------------------------------------
-- | Combine corresponding elements of two rank-4 arrays element-wise.
--   The arrays must have the same shape else error.
zipWith4
 :: (Elem a, Elem b, Elem c)
 => (a -> b -> c) -> Array4 a -> Array4 b -> Array4 c
zipWith4 f arrA arrB
 | not $ shape arrA == shape arrB = error "array shape mismatch"
 | otherwise
 = build4 (shape arrA) $ \ix -> f (index4 arrA ix) (index4 arrB ix)


------------------------------------------------------------------------------
-- | Sum all the elements in an array.
sumAll :: (Elem a, Num a) => Array sh a -> a
sumAll (Array _ elems)
 = U.sum elems


------------------------------------------------------------------------------
-- | Compute the dot product of elements in two arrays.
--   The arrays must have the same shape else error.
dot :: (IsShape sh, Elem a, Num a) => Array sh a -> Array sh a -> a
dot (Array sh1 elems1) (Array sh2 elems2)
 | not $ sh1 == sh2 = error "array shape mismatch"
 | otherwise
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

