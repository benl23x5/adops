
module Demo.Stereo where
import Adops.Array
import Adops.Codec
import Adops.Params
import qualified Adops.Op.Disparity     as Disparity
import qualified Adops.Op.Conv          as Conv
import qualified Adops.Op.Norm          as Norm
import qualified Adops.Op.Activate      as Actv
import qualified Adops.Op.Pool          as Pool
import Control.Monad
import Debug.Trace
import qualified Data.Vector.Unboxed    as U



-------------------------------------------------------------------------------
runStereo :: FilePath -> FilePath -> FilePath -> FilePath -> IO ()
runStereo pathBmpLeft pathBmpRight dirParams dirOut
 = do
        -- hyper parameters of the trained model.
        let nStartChannels  = 4
        let nDispMax16      = 12
        let ixCropBase      = Index2   0   0
        let ixCropShape     = Index2 512 960

        putStrLn "* Loading model parameters"
        let s = nStartChannels
        pDown1 <- readConv3dSepNorm dirParams "down1" $ Shape5 s         3 1 3 3
        pDown2 <- readConv3dSepNorm dirParams "down2" $ Shape5 (2*s)     s 1 3 3
        pDown3 <- readConv3dSepNorm dirParams "down3" $ Shape5 (4*s) (2*s) 1 3 3
        pDown4 <- readConv3dSepNorm dirParams "down4" $ Shape5 (8*s) (4*s) 1 3 3

        pDisp1 <- readConv3dSepNorm dirParams "disp1" $ Shape5 16  1 3 3 3
        pDisp2 <- readConv3dSepNorm dirParams "disp2" $ Shape5 16 16 3 3 3
        pDisp3 <- readConv3dSepNorm dirParams "disp3" $ Shape5 16 16 3 3 3
        pDisp4 <- readConv3dSepNorm dirParams "disp4" $ Shape5 16 16 3 3 3
        pDisp5 <- readConv3dSepNorm dirParams "disp5" $ Shape5  1 16 3 3 3

        putStrLn "* Loading input"
        aFullL <- readBMP pathBmpLeft
        aFullR <- readBMP pathBmpRight
        let nRows  = outer0 ixCropShape
        let nCols  = outer1 ixCropShape
        let aCropL = slicez3 aFullL (Index3 0 0 0) (Shape3 3 nRows nCols)
        let aCropR = slicez3 aFullR (Index3 0 0 0) (Shape3 3 nRows nCols)

        -- Raw input size is 540x960
        let nChas   = 3
        let aFullL  = reshape (Shape5 1 nChas 1 nRows nCols) aCropL
        let aFullR  = reshape (Shape5 1 nChas 1 nRows nCols) aCropR

        aDown1L <- dump "down1L" $ pool $ conv3d_sep_norm pDown1 (0, 1, 1) aFullL
        aDown2L <- dump "down2L" $ pool $ conv3d_sep_norm pDown2 (0, 1, 1) aDown1L
        aDown3L <- dump "down3L" $ pool $ conv3d_sep_norm pDown3 (0, 1, 1) aDown2L
        aDown4L <- dump "down4L" $ pool $ conv3d_sep_norm pDown4 (0, 1, 1) aDown3L

        aDown1R <- dump "down1R" $ pool $ conv3d_sep_norm pDown1 (0, 1, 1) aFullR
        aDown2R <- dump "down2R" $ pool $ conv3d_sep_norm pDown2 (0, 1, 1) aDown1R
        aDown3R <- dump "down3R" $ pool $ conv3d_sep_norm pDown3 (0, 1, 1) aDown2R
        aDown4R <- dump "down4R" $ pool $ conv3d_sep_norm pDown4 (0, 1, 1) aDown3R

        aCost   <- dump "volume" $ Disparity.costVolume 0 nDispMax16 (squeeze5 aDown4L) (squeeze5 aDown4R)

        aCost1  <- dump "cost1"  $ conv3d_sep_norm pDisp1 (1, 1, 1) (unsqueeze5 aCost)
        aCost2  <- dump "cost2"  $ conv3d_sep_norm pDisp2 (1, 1, 1) aCost1
        aCost3  <- dump "cost3"  $ conv3d_sep_norm pDisp3 (1, 1, 1) aCost2
        aCost4  <- dump "cost4"  $ conv3d_sep_norm pDisp4 (1, 1, 1) aCost3
        aCost5  <- dump "cost5"  $ conv3d_sep_norm pDisp5 (1, 1, 1) aCost4

        aDisp'  <- dump "regress" $ Disparity.regression (squeeze5 aCost5)

        -- Multiply by 16 to rescale disparity to be relative to full input range.
        let aDisp = mapAll (* 16) aDisp'

        -- In practice the image pairs do not use the full disparity range,
        --  so use a smaller range for rendering. Large disparities will be
        --  clipped to the maximum value in the color map.
        let nRenderMax = 12
        let nRenderMin = 0
        let norm x
                | x <= nRenderMin = 0
                | x >= nRenderMax = 1
                | otherwise = (x - nRenderMin) / (nRenderMax - nRenderMin)

        let aNorm  = mapAll norm aDisp

        -- Slice out the result and render it as a new image.
        let aSlice = takeChannel3 0 aNorm
        aDisp  <- beep "render" $ renderTurbo aSlice
        writeBMP (dirOut ++ "/disparity.bmp") aDisp


beep :: String -> a -> IO a
beep name x
 = do   putStrLn $ "* " ++ name
        !xx <- return x
        return xx


dump :: Show sh => String -> Array sh Float -> IO (Array sh Float)
dump name arr
 = do   putStrLn $ "* " ++ name
        (Array sh elts) <- return arr
        putStrLn $ " > " ++ show sh ++ " " ++ (show $ U.take 10 elts)
        putStrLn $ " ! min = " ++ show (U.minimum elts) ++ ", max =" ++ show (U.maximum elts)
        return arr



squeeze5 :: Array5 Float -> Array4 Float
squeeze5 arr
 = let  Shape5 nCount nChas 1 nRows nCols = shape arr
        sh = Shape4 nCount nChas nRows nCols
   in   reshape sh arr


unsqueeze5 :: Array4 Float -> Array5 Float
unsqueeze5 arr
 = let  Shape4 nCount nChas nRows nCols = shape arr
        sh = Shape5 nCount nChas 1 nRows nCols
   in   reshape sh arr



-- | Take a single channel from a rank-4 array.
takeChannel4 :: Int -> Int -> Array4 Float -> Array2 Float
takeChannel4 iImg iCha arr
 = let  Shape4 nCount nChas nRows nCols = shape arr
   in   reshape (Shape2 nRows nCols)
         $ slicez4 arr (Index4 iImg iCha 0 0) (Shape4 1 1 nRows nCols)


-- | Take a single channel from a rank-3 array.
takeChannel3 :: Int -> Array3 Float -> Array2 Float
takeChannel3 iImg arr
 = let  Shape3 _ nRows nCols = shape arr
   in   reshape (Shape2 nRows nCols)
         $ slicez3 arr (Index3 iImg 0 0) (Shape3 1 nRows nCols)


-------------------------------------------------------------------------------
-- | Padded separable convolution over the volumetric dimensions of a
--   rank-5 array, followed by batch normalisation and ReLU.
--
conv3d_sep_norm
  :: Params -> (Int, Int, Int) -> Array5 Float -> Array5 Float

conv3d_sep_norm
  (Conv3dSepNorm
    aCKrn aCBia _ _ _ _ aCScale aCBias
    aPKrn aPBia _ _ _ _ aPScale aPBias)
  (nPadLay, nPadRow, nPadCol)
  arrA
  = let reshape_5_3 (Array (Shape5 nImg nCha nDep nRow nCol) v) =
          Array (Shape3 nImg nCha (nDep * nRow * nCol)) v
        aCconv = Conv.conv3d_chan  (nPadLay, nPadRow, nPadCol) aCKrn  arrA
        aCnorm = Norm.batchnorm    aCScale   aCBias (reshape_5_3 aCconv)
        aC     = Actv.relu         aCnorm
        aPconv = Conv.conv3d_point (0, 0, 0) aPKrn  (reshape (shape aCconv) aC)
        aPnorm = Norm.batchnorm    aPScale   aPBias (reshape_5_3 aPconv)
        aP     = Actv.relu         aPnorm
        arrB   = reshape (shape aPconv) aP
    in  arrB

pool :: Array5 Float -> Array5 Float
pool arr = unsqueeze5 $ Pool.maxpool (2, 2) False $ squeeze5 arr


{-


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


-------------------------------------------------------------------------------
-- | Downsample input image in CHW order.
decimate :: Int -> Int -> Array4 Float -> Array4 Float
decimate dRows dCols arr
 = let  Shape4 nImgs nChas nRows nCols = shape arr
        nRows' = nRows `div` dRows
        nCols' = nCols `div` dCols
   in   build4 (Shape4 nImgs nChas nRows' nCols') $ \(Index4 iImg iCha iRow iCol) ->
         index4 arr (Index4 iImg iCha (iRow * dRows) (iCol * dCols))



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


-}
