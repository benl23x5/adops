{-# OPTIONS -fno-warn-missing-methods #-}
module Adops.Array
        ( module Adops.Array.Shape
        , module Adops.Array.Elem
        , Array(..)
        , Array1, Array2, Array3, Array4, Array5
        , IsShape(..)
        , reshape
        , (+.), (-.), (*.), aabs
        , fill, floats, fromList
        , build1, build2, build3, build4, build5
        , packChas3, packChas4
        , index1
        , index2
        , index3, indexz3
        , index4, indexz4
        , index5, indexz5
        , slicez3
        , slice4, slicez4
        , slicez5
        , zipWith4
        , mapAll
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
type Array5 a = Array Shape5 a

instance IsShape sh => HasShape (Array sh a) where
 type Shape (Array sh a) = sh
 shape  (Array sh _) = sh
 extent (Array sh a) = size sh


-- | Apply a new shape to the elements of an array.
--   The new shape must represent the same number of elements as the old one.
reshape
 :: (IsShape sh1, IsShape sh2)
 => sh2 -> Array sh1 a -> Array sh2 a
reshape sh2 arr@(Array _ elts)
 | not $ size (shape arr) == size sh2
 = error "shape mismatch"

 | otherwise = Array sh2 elts


------------------------------------------------------------------------------
(+.) :: (IsShape sh, U.Unbox a, Num a)
     => Array sh a -> Array sh a -> Array sh a
(+.) (Array sh1 elems1) (Array sh2 elems2)
 | sh1 == sh2
 = Array sh1 (U.zipWith (+) elems1 elems2)
 | otherwise   = error "shape mismatch"


(-.) :: (IsShape sh, U.Unbox a, Num a)
     => Array sh a -> Array sh a -> Array sh a
(-.) (Array sh1 elems1) (Array sh2 elems2)
 | sh1 == sh2
 = Array sh1 (U.zipWith (-) elems1 elems2)
 | otherwise   = error "shape mismatch"


(*.) :: (IsShape sh, U.Unbox a, Num a)
     => Array sh a -> Array sh a -> Array sh a
(*.) (Array sh1 elems1) (Array sh2 elems2)
 | sh1 == sh2
 = Array sh1 (U.zipWith (*) elems1 elems2)
 | otherwise   = error "shape mismatch"

aabs :: (U.Unbox a, Num a) => Array sh a -> Array sh a
aabs (Array sh elems)
 = Array sh $ U.map abs elems


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

fromList :: (IsShape sh, U.Unbox a)
         => sh -> [a] -> Array sh a
fromList sh elts
 = Array sh (U.fromList elts)


------------------------------------------------------------------------------
-- | Build an array of the given rank-1 shape,
--   given an function to produce the element at each index.
build1  :: Elem a => Shape1 -> (Index1 -> a) -> Array1 a
build1 sh make
 = Array sh $ U.generate (size sh) (\lix -> make $ fromLinear sh lix)

-- | Alias for `build1` that constrains the element type to be `Float`.
build1f :: Shape1 -> (Index1 -> Float) -> Array1 Float
build1f = build1


-- | Build an array of the given rank-2 shape,
--   given an function to produce the element at each index.
build2 :: Elem a => Shape2 -> (Index2 -> a) -> Array2 a
build2 sh make
 = Array sh $ U.generate (size sh) (\lix -> make $ fromLinear sh lix)

-- | Alias for `build2` that constrains the element type to be `Float`.
build2f :: Shape2 -> (Index2 -> Float) -> Array2 Float
build2f = build2


-- | Build an array of the given rank-3 shape,
--   given an function to produce the element at each index.
build3 :: Elem a => Shape3 -> (Index3 -> a) -> Array3 a
build3 sh make
 = Array sh $ U.generate (size sh) (\lix -> make $ fromLinear sh lix)

-- | Alias for `build3` that constrains the element type to be `Float`.
build3f :: Shape3 -> (Index3 -> Float) -> Array3 Float
build3f = build3


-- | Build an array of the given rank-4 shape,
--   given an function to produce the element at each index.
build4  :: Elem a => Shape4 -> (Index4 -> a) -> Array4 a
build4 sh make
 = Array sh $ U.generate (size sh) (\lix -> make $ fromLinear sh lix)

-- | Alias for `build4` that constrains the element type to be `Float`.
build4f :: Shape4 -> (Index4 -> Float) -> Array4 Float
build4f = build4


-- | Build an array of the given rank-5 shape,
--   given an function to produce the element at each index.
build5  :: Elem a => Shape5 -> (Index5 -> a) -> Array5 a
build5 sh make
 = Array sh $ U.generate (size sh) (\lix -> make $ fromLinear sh lix)

-- | Alias for `build5` that constrains the element type to be `Float`.
build5f :: Shape5 -> (Index5 -> Float) -> Array5 Float
build5f = build5


------------------------------------------------------------------------------
packChas3 :: Shape3 -> [Array2 Float] -> Array3 Float
packChas3 sh xs
 = let Shape3 nImgs _ _ = sh
   in  check (nImgs == length xs)
        $ build3 sh $ \(Index3 iCha iRow iCol) ->
           let a = xs !! iCha
           in  index2 a (Index2 iRow iCol)

packChas4 :: Shape4 -> [Array3 Float] -> Array4 Float
packChas4 sh xs
 = let Shape4 nImgs _ _ _ = sh
   in  check (nImgs == length xs)
        $ build4 sh $ \(Index4 iImg iCha iRow iCol) ->
           let a = xs !! iImg
           in  index3 a (Index3 iCha iRow iCol)

------------------------------------------------------------------------------
-- | Retrieve the element at the given index,
--   throwing error on out of range indices.
index1  :: Elem a => Array1 a -> Index1 -> a
index1 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise
 = error $ unlines
 [ "index1: out of range"
 , "  index = " ++ show ix
 , "  shape = " ++ show sh ]

-- | Retrieve the element at the given index,
--   throwing error on out of range indices.
index2  :: Elem a => Array2 a -> Index2 -> a
index2 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise
 = error $ unlines
 [ "index2: out of range"
 , "  index = " ++ show ix
 , "  shape = " ++ show sh ]

-- | Retrieve the element at the given index,
--   throwing error on out of range indices.
index3  :: Elem a => Array3 a -> Index3 -> a
index3 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise
 = error $ unlines
 [ "index3: out of range"
 , "  index = " ++ show ix
 , "  shape = " ++ show sh ]

-- | Retrieve the element at the given index,
--   returning zero for out of range indices.
indexz3 :: Elem a => Array3 a -> Index3 -> a
indexz3 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise    = zero


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


-- | Retrieve the element at the given index,
--   throwing error on out of range indices.
index5  :: Elem a => Array5 a -> Index5 -> a
index5 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise    = error "out of range"

-- | Retrieve the element at the given index,
--   returning zero for out of range indices.
indexz5 :: Elem a => Array5 a -> Index5 -> a
indexz5 (Array sh elems) ix
 | within ix sh = elems U.! toLinear sh ix
 | otherwise    = zero


------------------------------------------------------------------------------
-- | Slice a section out of a rank-3 array,
--   given a base offset and shape of the section.
--
--   If the slice extends out side the source array then the corresponding
--   elements are set to zero.
slicez3 :: Elem a => Array3 a -> Index3 -> Shape3 -> Array3 a
slicez3 arr ixBase shResult
 = build3 shResult $ \ixResult -> indexz3 arr (ixBase + ixResult)


-- | Slice a section out of a rank-4 array,
--   given a base offset and shape of the section.
--
--   If the slice extends out side the source array then error.
slice4 :: Elem a => Array4 a -> Index4 -> Shape4 -> Array4 a
slice4 arr ixBase shResult
 = build4 shResult $ \ixResult -> index4 arr (ixBase + ixResult)


-- | Slice a section out of a rank-4 array,
--   given a base offset and shape of the section.
--
--   If the slice extends out side the source array then the corresponding
--   elements are set to zero.
slicez4 :: Elem a => Array4 a -> Index4 -> Shape4 -> Array4 a
slicez4 arr ixBase shResult
 = build4 shResult $ \ixResult -> indexz4 arr (ixBase + ixResult)


-- | Slice a section out of a rank-4 array,
--   given a base offset and shape of the section.
--
--   If the slice extends out side the source array then the corresponding
--   elements are set to zero.
slicez5 :: Elem a => Array5 a -> Index5 -> Shape5 -> Array5 a
slicez5 arr ixBase shResult
 = build5 shResult $ \ixResult -> indexz5 arr (ixBase + ixResult)

------------------------------------------------------------------------------
-- | Apply a function to every element of an array.
mapAll :: (Elem a, Elem b) => (a -> b) -> Array sh a -> Array sh b
mapAll f (Array sh elts) = Array sh $ U.map f elts


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

