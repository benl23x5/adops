
module Demo.Stereo where
import Adops.Array
import Adops.Codec
import Adops.Params
import Control.Monad
import Debug.Trace
import Text.Printf
import qualified Adops.Op.Disparity     as Disparity
import qualified Adops.Op.Conv          as Conv
import qualified Adops.Op.Norm          as Norm
import qualified Adops.Op.Activate      as Actv
import qualified Adops.Op.Pool          as Pool
import qualified Data.Vector.Unboxed    as U
import qualified System.Exit            as System

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

        aDown1L <- fmap pool $ conv3d_sep_norm "aDown1L" pDown1 (0, 1, 1) aFullL
--        System.exitWith (System.ExitSuccess)

        aDown2L <- fmap pool $ conv3d_sep_norm "aDown2L" pDown2 (0, 1, 1) aDown1L
        aDown3L <- fmap pool $ conv3d_sep_norm "aDown3L" pDown3 (0, 1, 1) aDown2L
        aDown4L <- fmap pool $ conv3d_sep_norm "aDown4L" pDown4 (0, 1, 1) aDown3L

        aDown1R <- fmap pool $ conv3d_sep_norm "aDown1R" pDown1 (0, 1, 1) aFullR
        aDown2R <- fmap pool $ conv3d_sep_norm "aDown2R" pDown2 (0, 1, 1) aDown1R
        aDown3R <- fmap pool $ conv3d_sep_norm "aDown3R" pDown3 (0, 1, 1) aDown2R
        aDown4R <- fmap pool $ conv3d_sep_norm "aDown4R" pDown4 (0, 1, 1) aDown3R

        let aCost = Disparity.costVolume 0 nDispMax16 (squeeze5 aDown4L) (squeeze5 aDown4R)
        dumpNormTurbo4 aCost "output" "bCost"

        aCost1  <- conv3d_sep_norm "bCost1" pDisp1 (1, 1, 1) (unsqueeze5 aCost)
        aCost2  <- conv3d_sep_norm "bCost2" pDisp2 (1, 1, 1) aCost1
        aCost3  <- conv3d_sep_norm "bCost3" pDisp3 (1, 1, 1) aCost2
        aCost4  <- conv3d_sep_norm "bCost4" pDisp4 (1, 1, 1) aCost3
        aCost5  <- conv3d_sep_norm "bCost5" pDisp5 (1, 1, 1) aCost4

        let aDisp' = Disparity.regression (squeeze5 aCost5)

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
--        putStrLn $ " > " ++ show sh ++ " " ++ (show elts)
--        putStrLn $ " ! min = " ++ show (U.minimum elts) ++ ", max =" ++ show (U.maximum elts)
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
  :: String -> Params -> (Int, Int, Int) -> Array5 Float -> IO (Array5 Float)

conv3d_sep_norm tag
  (Conv3dSepNorm
    aCKrn aCBia _ _ _ _ aCScale aCBias
    aPKrn aPBia _ _ _ _ aPScale aPBias)
  (nPadLay, nPadRow, nPadCol)
  arrA
  = do  let reshape_5_3 (Array (Shape5 nImg nCha nDep nRow nCol) v) =
                Array (Shape3 nImg nCha (nDep * nRow * nCol)) v
        let aCconv    = Conv.conv3d_chan  (nPadLay, nPadRow, nPadCol) aCKrn aCBia arrA
        let aCnorm    = Norm.batchnorm    aCScale   aCBias (reshape_5_3 aCconv)
        let aC        = Actv.relu         aCnorm
        let aPconv    = Conv.conv3d_point (0, 0, 0) aPKrn aPBia  (reshape (shape aCconv) aC)
        let aPnorm    = Norm.batchnorm    aPScale   aPBias (reshape_5_3 aPconv)
        let aP        = Actv.relu         aPnorm
        let arrB      = reshape (shape aPconv) aP

        let aResult   = Conv.conv3d_chan (0, 1, 1) aCKrn aCBia arrA
        let aPatchOut = slicez5 aResult (Index5 0 0 0 100 100) (Index5 1 1 1 1 1)

        let aPatchIn  = slicez5 arrA  (Index5 0 0 0 99 99) (Index5 1 1 1 3 3)
        let aKrn1     = slicez5 aCKrn (Index5 0 0 0 0 0) (Index5 1 1 1 3 3)
        let aBia1     = index1 aCBia (Index1 0)
        let aAgain    = dot aPatchIn aKrn1 + aBia1

        dumpNormTurbo5 (reshape (shape aCconv) aC) "output" (tag ++ "-c")
        dumpNormTurbo5 (reshape (shape aPconv) aP) "output" (tag ++ "-p")

        return arrB

pool :: Array5 Float -> Array5 Float
pool arr = unsqueeze5 $ Pool.maxpool (2, 2) False $ squeeze5 arr

