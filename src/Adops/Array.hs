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


