
module Demo.Stereo where
import Adops.Array
import Adops.Codec
import qualified Adops.Op.Disparity     as Disparity
import qualified Adops.Op.Conv          as Conv
import Control.Monad

-------------------------------------------------------------------------------
runStereo :: FilePath -> FilePath -> FilePath -> IO ()
runStereo pathBmpLeft pathBmpRight dirOut
 = do
        -- Load input images.
        putStrLn "* Loading input"
        aFullL3 <- readBMP pathBmpLeft
        aFullR3 <- readBMP pathBmpRight

        -- Raw input size 540x960
        let Shape3 nChas nRows nCols = shape aFullL3
        let aFullL4 = reshape (Shape4 1 nChas nRows nCols) aFullL3
        let aFullR4 = reshape (Shape4 1 nChas nRows nCols) aFullR3

        -- Downsample input to 135x240
        let dRows  = 4
        let dCols  = 4
        let nRows_ = nRows `div` dRows
        let nCols_ = nCols `div` dCols

        let aDownL = decimate dRows dCols aFullL4
        let aDownR = decimate dRows dCols aFullR4

        -- Compute feature maps.
        let Shape4 nFeat _ _ _ = shape kernels
        let aFeatL = Conv.conv2d_pad (1, 1) kernels aDownL
        let aFeatR = Conv.conv2d_pad (1, 1) kernels aDownR

        putStrLn "* Writing feature maps"
        forM_ [0 .. nFeat - 1] $ \iFeat -> do
         writeBMP (dirOut ++ "/featureL-" ++ show iFeat ++ ".bmp")
          $ renderGrey $ takeChannel iFeat aFeatL

        forM_ [0 .. nFeat - 1] $ \iFeat -> do
         writeBMP (dirOut ++ "/featureR-" ++ show iFeat ++ ".bmp")
          $ renderGrey $ takeChannel iFeat aFeatR

        -- Compute cost volume between left and right images.
        --  The maximum disparity in the downsampled input example is about 30 pixels.
        let nCount  = 30 :: Int

        -- Compute disparity cost volume which shows how well each image element
        -- matches at each possible disparity.
        let aCost  = Disparity.costVolume 0 nCount aFeatL aFeatR

        -- Apply disparity regression to determine the position of the best
        -- match for each pixel.
        let aMin   = Disparity.regression aCost

        -- In practice the image pairs do not use the full disparity range,
        --  so use a smaller range for rendering. Large disparities will be
        --  clipped to the maximum value in the color map.
        let nRenderMax = 25 -- fromIntegral nCount
        let nRenderMin = 5
        let norm x
                | x <= nRenderMin = 0
                | x >= nRenderMax = 1
                | otherwise = (x - nRenderMin) / (nRenderMax - nRenderMin)

        let aNorm  = mapAll norm aMin

        -- Slice out the result and render it as a new image.
        let aSlice = reshape (Shape2 nRows_ nCols_)
                   $ slicez3 aNorm (Index3 0 0 0) (Index3 1 nRows_ nCols_)
        let aDisp = renderTurbo aSlice

        putStrLn "* Writing disparity"
        writeBMP (dirOut ++ "/disparity.bmp") aDisp


-- | Downsample input image in CHW order.
decimate :: Int -> Int -> Array4 Float -> Array4 Float
decimate dRows dCols arr
 = let  Shape4 nImgs nChas nRows nCols = shape arr
        nRows' = nRows `div` dRows
        nCols' = nCols `div` dCols
   in   build4 (Shape4 nImgs nChas nRows' nCols') $ \(Index4 iImg iCha iRow iCol) ->
         index4 arr (Index4 iImg iCha (iRow * dRows) (iCol * dCols))


-- | Take a single channel from a rank-3 array.
takeChannel :: Int -> Array4 Float -> Array2 Float
takeChannel iCha arr
 = let  Shape4 1 nChas nRows nCols = shape arr
   in   reshape (Shape2 nRows nCols)
         $ slicez4 arr (Index4 0 iCha 0 0) (Shape4 1 1 nRows nCols)


-------------------------------------------------------------------------------
kZero :: Array2 Float
kZero
 = fromList (Shape2 3 3)
 [ 0, 0, 0
 , 0, 0, 0
 , 0, 0, 0 ]

kPoint :: Array2 Float
kPoint
 = fromList (Shape2 3 3)
 [ 0, 0, 0
 , 0, 1, 0
 , 0, 0, 0 ]

kThird :: Array2 Float
kThird
 = fromList (Shape2 3 3) $ map (/ 3)
 [ 0, 0, 0
 , 0, 1, 0
 , 0, 0, 0 ]

kSobelX :: Array2 Float
kSobelX
 = fromList (Shape2 3 3) $ map (/4)
 [ 1, 0, -1
 , 2, 0, -2
 , 1, 0, -1 ]

kSobelY :: Array2 Float
kSobelY
 = fromList (Shape2 3 3) $ map (/4)
 [  1,  2,  1
 ,  0,  0,  0
 , -1, -2, -1 ]

kernels :: Array4 Float
kernels
 = packChas4 (Shape4 (length ks) 3 3 3) ks
 where  sh =    Shape3 3 3 3
        ks =    [ packChas3 sh [kPoint,  kZero,   kZero]
                , packChas3 sh [kZero,   kPoint,  kZero]
                , packChas3 sh [kZero,   kZero,   kPoint]
                , packChas3 sh [kThird,  kThird,  kThird]
                , packChas3 sh [kSobelX, kZero,   kZero]
                , packChas3 sh [kZero,   kSobelX, kZero]
                , packChas3 sh [kZero,   kZero,   kSobelX]
                , packChas3 sh [kSobelY, kZero,   kZero]
                , packChas3 sh [kZero,   kSobelY, kZero]
                , packChas3 sh [kZero,   kZero,   kSobelY]
                ]

