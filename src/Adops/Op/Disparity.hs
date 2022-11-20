
module Adops.Op.Disparity where
import Adops.Array
import Adops.Op.Norm

-- | Disparity cost volume.
--
--   Take two arrays of multi channel 2d images, where the first contains
--   left views of the scene and the second contains right views.
--
--   For each pair of images, slice the right image over the left image,
--   and for each offset produce the L1 distance indicating how well correponding
--   multi-channel image elements in the right image match those in the left.
--
--   Described in:
--    Anytime Stereo Image Depth Estimation on Mobile Devices
--    Wang, Lai et al, ICRA 2019
--    https://arxiv.org/abs/1810.11408
--    Section III b).
--
costVolume
 :: (Elem a, Num a)
 => Int -> Int -> Array4 a -> Array4 a -> Array4 a

-- arr{L,R} :  images x channels  x height x width
-- output   :  images x disparity x height x width

costVolume iStart count arrL arrR
 = check (shape arrL == shape arrR)
 $ let  Shape4 nImgs nChas nRows nCols = shape arrL
        shOut = Shape4 nImgs count nRows nCols

   in   build4 shOut $ \(Index4 iImg iDisp iRow iCol) ->
        let arrVecL = build1 (Shape1 nChas) $ \(Index1 iCha) ->
                        index4  arrL (Index4 iImg iCha iRow iCol)

            iSrc = iCol - iStart - iDisp
            arrVecR = build1 (Shape1 nChas) $ \(Index1 iCha) ->
                        indexz4 arrR (Index4 iImg iCha iRow iSrc)

        in sumAll (aabs (arrVecL -. arrVecR))
{-# INLINE costVolume #-}


-- | Disparity regression.
--
--   Disparity regression is a reduction operator along the inner-most
--   dimension, that applies the following computation to each inner vector.
--
--   {{{
--   regression C = Σ k=0,M.  k *      exp (-C_ijk)
--                                -----------------------
--                                Σ k'=0,M. exp(-C_ijk')
--   }}}
--
--   Noticing that the right hand side is the same as the softmax
--   operator applied along the channels axis of the array, we have:
--   {{{
--   regression C = [ i j → sum [ k → k * (softmax (negate C[i,j,*]))[k] ] ]
--   }}}
--
--   Described in:
--    Anytime Stereo Image Depth Estimation on Mobile Devices
--    Wang, Lai et al, ICRA 2019
--    https://arxiv.org/abs/1810.11408
--    Section III b).
--
regression :: Array4 Float -> Array3 Float

-- output : images x height x width

regression arr
 = let  Shape4 nImgs nChas nRows nCols = shape arr
        aKs   = build1 (Shape1 nChas) $ \(Index1 iCha) ->
                  fromIntegral iCha
        shOut = Shape3 nImgs nRows nCols

   in   build3 shOut $ \(Index3 iImg iRow iCol) ->
        let aVec = build1 (Shape1 nChas) $ \(Index1 iCha) ->
                     index4 arr (Index4 iImg iCha iRow iCol)
            aNeg = mapAll (\x -> x * (-1)) aVec
        in  sumAll (aKs *. softmax aNeg)
