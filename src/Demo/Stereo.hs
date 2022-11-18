
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
        let nDispMaxFull    = nDispMax16 * 16
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

        let aDown1L = pool $ conv3d_sep_norm pDown1 (0, 1, 1) aFullL
        let aDown2L = pool $ conv3d_sep_norm pDown2 (0, 1, 1) aDown1L
        let aDown3L = pool $ conv3d_sep_norm pDown3 (0, 1, 1) aDown2L
        let aDown4L = pool $ conv3d_sep_norm pDown4 (0, 1, 1) aDown3L

        let aDown1R = pool $ conv3d_sep_norm pDown1 (0, 1, 1) aFullR
        let aDown2R = pool $ conv3d_sep_norm pDown2 (0, 1, 1) aDown1R
        let aDown3R = pool $ conv3d_sep_norm pDown3 (0, 1, 1) aDown2R
        let aDown4R = pool $ conv3d_sep_norm pDown4 (0, 1, 1) aDown3R

        let aCost0_ = Disparity.costVolume 0 nDispMax16 (squeeze5 aDown4L) (squeeze5 aDown4R)
        let (Shape4 1 nChasCost nRowsCost nColsCost) = shape aCost0_
        let aCost0  = reshape (Shape5 1 1 nChasCost nRowsCost nColsCost) aCost0_

        let aCost1  = conv3d_sep_norm pDisp1 (1, 1, 1) aCost0
        let aCost2  = conv3d_sep_norm pDisp2 (1, 1, 1) aCost1
        let aCost3  = conv3d_sep_norm pDisp3 (1, 1, 1) aCost2
        let aCost4  = conv3d_sep_norm pDisp4 (1, 1, 1) aCost3
        let aCost5  = conv3d_sep_norm pDisp5 (1, 1, 1) aCost4

        let aCost5' = reshape (Shape4 1 nChasCost nRowsCost nColsCost) aCost5
        let aDisp'  = Disparity.regression aCost5'

        -- Multiply by 16 to rescale disparity to be relative to full input range.
        -- This is the final output of the model.
        let aDisp = mapAll (* 16) aDisp'

        -- Normalized to range [0,1] for rendering.
        let aNorm  = mapAll (/ fromIntegral nDispMaxFull) aDisp
        let aImage = renderTurbo $ takeChannel3 0 aNorm
        writeBMP (dirOut ++ "/disparity.bmp") aImage


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
        aCconv    = Conv.conv3d_chan  (nPadLay, nPadRow, nPadCol) aCKrn aCBia arrA
        aCnorm    = Norm.batchnorm    aCScale   aCBias (reshape_5_3 aCconv)
        aC        = Actv.relu         aCnorm
        aPconv    = Conv.conv3d_point (0, 0, 0) aPKrn aPBia  (reshape (shape aCconv) aC)
        aPnorm    = Norm.batchnorm    aPScale   aPBias (reshape_5_3 aPconv)
        aP        = Actv.relu         aPnorm
     in reshape (shape aPconv) aP

pool :: Array5 Float -> Array5 Float
pool arr = unsqueeze5 $ Pool.maxpool (2, 2) False $ squeeze5 arr


-------------------------------------------------------------------------------
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

