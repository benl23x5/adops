{-# OPTIONS -fno-warn-missing-methods #-}
module Adops.Array.Shape where

-------------------------------------------------------------------------------
-- | A shape object, which describes a range of indexable elements.
class (Show sh, Eq sh) => IsShape sh where
 -- | Get the total number of indexable elements in this object.
 size  :: sh -> Int

 -- | Get the sizes of each dimension, from outer to inner.
 dims :: sh -> [Int]

 -- | Given a index and shape, check if the index is fully within the shape.
 within :: sh -> sh -> Bool

 -- | Given a bunched shape and index,
 --   yield the corresponding linear index.
 toLinear :: sh -> sh -> Int

 -- | Given a bunched shape and linear index,
 --   yield the corresponding bunched index.
 fromLinear :: sh -> Int -> sh


-------------------------------------------------------------------------------
-- | An object with attached shape information.
class IsShape (Shape a) => HasShape a where
 type Shape a
 shape  :: a -> Shape a

 extent :: a -> Int
 extent a = size (shape a)


-------------------------------------------------------------------------------
-- | An object containing elements where the index of each element can be
--   mapped to an underling linear space using the given stride information.
class IsShape sh => HasStrides sh where
 type Strides sh

 -- | Get the strides of this object.
 strides  :: sh -> Strides sh


-------------------------------------------------------------------------------
-- | An object with at least one dimension.
class Dim1 sh where
 outer0 :: sh -> Int
 inner0 :: sh -> Int


-- | An object with at least two dimensions.
class Dim1 sh => Dim2 sh where
 outer1 :: sh -> Int
 inner1 :: sh -> Int


-- | An object with at least three dimensions.
class Dim2 sh => Dim3 sh where
 outer2 :: sh -> Int
 inner2 :: sh -> Int


-- | An object with at least four dimensions.
class Dim3 sh => Dim4 sh where
 outer3 :: sh -> Int
 inner3 :: sh -> Int


-- | An object with at least five dimensions.
class Dim4 sh => Dim5 sh where
 outer4 :: sh -> Int
 inner4 :: sh -> Int


-------------------------------------------------------------------------------
-- | Flatten a shape completely.
class Flatten sh where
 type Flat sh
 flatten :: sh -> Flat sh


-------------------------------------------------------------------------------
-- | A generic shape with a single dimension.
data Shape1
 = Shape1 Int
 deriving (Eq, Show)

type Index1 = Shape1

pattern Index1 i0 = Shape1 i0

instance IsShape Shape1 where
 size (Shape1 i0) = i0
 dims (Shape1 i0) = [i0]

 within (Index1 i0) (Shape1 s0)
  = i0 >= 0 && i0 < s0

 toLinear (Shape1 s0) (Shape1 i0) = i0

 fromLinear (Shape1 _) lix = Index1 lix


instance HasStrides Shape1 where
 type Strides Shape1 = Shape1
 strides (Shape1 _) = Shape1 1


instance Dim1 Shape1 where
 inner0 (Shape1 i0) = i0
 outer0 (Shape1 o0) = o0


instance Flatten Shape1 where
 type Flat Shape1 = Shape1
 flatten (Shape1 sh1_0)
  = Shape1 sh1_0


instance Num Shape1 where
 (+) (Shape1 a0) (Shape1 b0)
  = Shape1 (a0 + b0)


-------------------------------------------------------------------------------
data Shape2
 = Shape2 Int Int
 deriving (Eq, Show)

type Index2 = Shape2

pattern Index2 i1 i0 = Shape2 i1 i0

instance IsShape Shape2 where
 size (Shape2 i1 i0) = i1 * i0
 dims (Shape2 i1 i0)  = [i1, i0]

 within (Index2 i1 i0) (Shape2 s1 s0)
  =  i0 >= 0 && i0 < s0
  && i1 >= 0 && i1 < s1

 toLinear (Shape2 _ s0) (Index2 i1 i0)
  = i1 * s0 + i0

 fromLinear (Shape2 _ s0) lix
  = Index2 i1 i0
  where p0 = s0
        i1 = lix `div` p0
        i0 = lix `mod` p0


instance HasStrides Shape2 where
 type Strides Shape2 = Shape2
 strides (Shape2 _ i0)
  = Shape2 i0 1


instance Dim1 Shape2 where
 inner0 (Shape2 _ i0)  = i0
 outer0 (Shape2 o0 _)  = o0


instance Dim2 Shape2 where
 inner1 (Shape2 i1 _)  = i1
 outer1 (Shape2 _ o1)  = o1


instance Flatten Shape2 where
 type Flat Shape2 = Shape1
 flatten (Shape2 sh2_0 sh2_1)
  = Shape1 (sh2_0 * sh2_1)


instance Num Shape2 where
 (+) (Shape2 a0 a1) (Shape2 b0 b1)
  = Shape2 (a0 + b0) (a1 + b1)


-------------------------------------------------------------------------------
data Shape3
 = Shape3 Int Int Int
 deriving (Eq, Show)

type Index3 = Shape3

pattern Index3 i2 i1 i0 = Shape3 i2 i1 i0

instance IsShape Shape3 where
 size (Shape3 i2 i1 i0) = i2 * i1 * i0
 dims (Shape3 i2 i1 i0) = [i2, i1, i0]

 within (Index3 i2 i1 i0) (Shape3 s2 s1 s0)
  =  i0 >= 0 && i0 < s0
  && i1 >= 0 && i1 < s1
  && i2 >= 0 && i2 < s2

 toLinear (Shape3 _ s1 s0) (Shape3 i2 i1 i0)
  = (i2 * s1 + i1) * s0 + i0

 fromLinear (Shape3 _ s1 s0) lix
  = Index3 i2 i1 i0
  where p0 = s0
        p1 = s1 * s0

        i2 = lix `div` p1
        r2 = lix `mod` p1

        i1 = r2  `div` p0
        i0 = r2  `mod` p0


instance HasStrides Shape3 where
 type Strides Shape3 = Shape3
 strides (Shape3 _ i1 i0)
  = Shape3 (i1 * i0) i0 1


instance Dim1 Shape3 where
 inner0 (Shape3 _ _ i0) = i0
 outer0 (Shape3 o0 _ _) = o0


instance Dim2 Shape3 where
 inner1 (Shape3 _ i1 _) = i1
 outer1 (Shape3 _ o1 _) = o1


instance Dim3 Shape3 where
 inner2 (Shape3 i2 _ _) = i2
 outer2 (Shape3 _ _ o2) = o2


instance Flatten Shape3 where
 type Flat Shape3 = Shape1
 flatten (Shape3 sh3_0 sh3_1 sh3_2)
  = Shape1 (sh3_0 * sh3_1 * sh3_2)


instance Num Shape3 where
 (+) (Shape3 a0 a1 a2) (Shape3 b0 b1 b2)
  = Shape3 (a0 + b0) (a1 + b1) (a2 + b2)


-------------------------------------------------------------------------------
data Shape4
 = Shape4 Int Int Int Int
 deriving (Eq, Show)

type Index4 = Shape4

pattern Index4 i3 i2 i1 i0 = Shape4 i3 i2 i1 i0

instance IsShape Shape4 where
 size (Shape4 i3 i2 i1 i0)
  = i3 * i2 * i1 * i0

 dims (Shape4 i3 i2 i1 i0)
  = [i3, i2, i1, i0]

 within (Index4 i3 i2 i1 i0) (Shape4 s3 s2 s1 s0)
  =  i0 >= 0 && i0 < s0
  && i1 >= 0 && i1 < s1
  && i2 >= 0 && i2 < s2
  && i3 >= 0 && i3 < s3

 toLinear (Shape4 _ s2 s1 s0) (Shape4 i3 i2 i1 i0)
  = ((i3 * s2 + i2) * s1 + i1) * s0 + i0

 fromLinear (Shape4 _ s2 s1 s0) lix
  = Index4 i3 i2 i1 i0
  where p0 = s0
        p1 = s1 * s0
        p2 = s2 * p1

        i3 = lix `div` p2
        r3 = lix `mod` p2

        i2 = r3  `div` p1
        r2 = r3  `mod` p1

        i1 = r2  `div` p0
        i0 = r2  `mod` p0


instance HasStrides Shape4 where
 type Strides Shape4 = Shape4
 strides (Shape4 _ i2 i1 i0)
  = Shape4 (i2 * i1 * i0) (i1 * i0) i0 1


instance Dim1 Shape4 where
 inner0 (Shape4 _ _ _ i0) = i0
 outer0 (Shape4 o0 _ _ _) = o0


instance Dim2 Shape4 where
 inner1 (Shape4 _ _ i1 _) = i1
 outer1 (Shape4 _ o1 _ _) = o1


instance Dim3 Shape4 where
 inner2 (Shape4 _ i2 _ _) = i2
 outer2 (Shape4 _ _ o2 _) = o2


instance Dim4 Shape4 where
 inner3 (Shape4 i3 _ _ _) = i3
 outer3 (Shape4 _ _ _ o3) = o3


instance Flatten Shape4 where
 type Flat Shape4 = Shape1
 flatten (Shape4 i h r c)
  = Shape1 (i * h * r * c)


instance Num Shape4 where
 (+) (Shape4 a0 a1 a2 a3) (Shape4 b0 b1 b2 b3)
  = Shape4 (a0 + b0) (a1 + b1) (a2 + b2) (a3 + b3)


-------------------------------------------------------------------------------
data Shape5
 = Shape5 Int Int Int Int Int
 deriving (Eq, Show)

type Index5 = Shape5

pattern Index5 i4 i3 i2 i1 i0 = Shape5 i4 i3 i2 i1 i0

instance IsShape Shape5 where
 size (Shape5 i4 i3 i2 i1 i0)
  = i4 * i3 * i2 * i1 * i0

 dims (Shape5 i4 i3 i2 i1 i0)
  = [i4, i3, i2, i1, i0]

 within (Index5 i4 i3 i2 i1 i0) (Shape5 s4 s3 s2 s1 s0)
  =  i0 >= 0 && i0 < s0
  && i1 >= 0 && i1 < s1
  && i2 >= 0 && i2 < s2
  && i3 >= 0 && i3 < s3
  && i4 >= 0 && i4 < s4

 toLinear (Shape5 _ s3 s2 s1 s0) (Shape5 i4 i3 i2 i1 i0)
  = (((i4 * s3 + i3) * s2 + i2) * s1 + i1) * s0 + i0

 fromLinear (Shape5 _ s3 s2 s1 s0) lix
  = Index5 i4 i3 i2 i1 i0
  where p0 = s0
        p1 = s1 * s0
        p2 = s2 * p1
        p3 = s3 * p2

        i4 = lix `div` p3
        r4 = lix `mod` p3

        i3 = r4  `div` p2
        r3 = r4  `mod` p2

        i2 = r3  `div` p1
        r2 = r3  `mod` p1

        i1 = r2  `div` p0
        i0 = r2  `mod` p0


instance HasStrides Shape5 where
 type Strides Shape5 = Shape5
 strides (Shape5 _ i3 i2 i1 i0)
  = Shape5 (i3 * i2 * i1 * i0) (i2 * i1 * i0) (i1 * i0) i0 1


instance Dim1 Shape5 where
 inner0 (Shape5 _ _ _ _ i0) = i0
 outer0 (Shape5 o0 _ _ _ _) = o0


instance Dim2 Shape5 where
 inner1 (Shape5 _ _ _ i1 _) = i1
 outer1 (Shape5 _ o1 _ _ _) = o1


instance Dim3 Shape5 where
 inner2 (Shape5 _ _ i2 _ _) = i2
 outer2 (Shape5 _ _ o2 _ _) = o2


instance Dim4 Shape5 where
 inner3 (Shape5 _ i3 _ _ _) = i3
 outer3 (Shape5 _ _ _ o3 _) = o3


instance Dim5 Shape5 where
 inner4 (Shape5 i4 _ _ _ _) = i4
 outer4 (Shape5 _ _ _ _ o4) = o4


instance Flatten Shape5 where
 type Flat Shape5 = Shape1
 flatten (Shape5 i d h r c)
  = Shape1 (i * d * h * r * c)


instance Num Shape5 where
 (+) (Shape5 a0 a1 a2 a3 a4) (Shape5 b0 b1 b2 b3 b4)
  = Shape5 (a0 + b0) (a1 + b1) (a2 + b2) (a3 + b3) (a4 + b4)

