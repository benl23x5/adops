
module Adops.Op.Disparity where
import Adops.Array

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

costVolume iStart count arrL arrR
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

