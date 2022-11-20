
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
-- Simple stereo vision network.
--
-- The network architecture is a simplified version of:
--  Anytime Stereo Image Depth Estimation on Mobile Devices
--  Wang, Lai, Huang et al, ICRA 2019.
--
-- This simplified version generates features at 1/16 timesthe input size,
-- using a feed forward network instead of a UNet, uses the differentiable
-- disparity cost volume and disparity regression operators, but does not
-- perform warping or upsampling.
--
runStereo :: FilePath -> FilePath -> FilePath -> FilePath -> IO ()
runStereo pathBmpLeft pathBmpRight dirParams dirOut
 = do
        -- Hyper parameters of the pre-trained model.
        --  Number of starting channels for the feature generator.
        let nStartChannels  = 4

        --  Maximum supported in 16x downsampled image.
        --  This is the number of channels we produce in the disparity cost volume.
        let nDispMax16      = 12

        --  Maximum suported disparity relative to the full resolution
        --  input image.
        let nDispMaxFull    = nDispMax16 * 16

        --  Base index for input crop.
        let ixCropBase      = Index2   0   0

        --  Size of the input crop.
        --  The network architecture requires both the number of rows
        --  and columns to be evenly divisible by 4.
        let ixCropShape     = Index2 512 960

        -- Used to dump intermediate activations.
        let dump5' name arr = dump5 True dirOut name arr

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

        putStrLn "* Loading input sample"
        aFullL <- readBMP pathBmpLeft
        aFullR <- readBMP pathBmpRight

        let nChasRGB  = 3
        let nRows     = outer0 ixCropShape
        let nCols     = outer1 ixCropShape
        let aCropL    = slicez3 aFullL (Index3 0 0 0) (Shape3 3 nRows nCols)
        let aCropR    = slicez3 aFullR (Index3 0 0 0) (Shape3 3 nRows nCols)

        putStrLn "* Evaluating network"
        aFullL  <- dump5' "s0-aFullL" $ reshape (Shape5 1 nChasRGB 1 nRows nCols) aCropL
        aFullR  <- dump5' "s0-aFullR" $ reshape (Shape5 1 nChasRGB 1 nRows nCols) aCropR

        aDown1L <- dump5' "s1-aDown1L" $ pool $ conv3d_sep_norm pDown1 (0, 1, 1) aFullL
        aDown2L <- dump5' "s1-aDown2L" $ pool $ conv3d_sep_norm pDown2 (0, 1, 1) aDown1L
        aDown3L <- dump5' "s1-aDown3L" $ pool $ conv3d_sep_norm pDown3 (0, 1, 1) aDown2L
        aDown4L <- dump5' "s1-aDown4L" $ pool $ conv3d_sep_norm pDown4 (0, 1, 1) aDown3L

        aDown1R <- dump5' "s1-aDown1R" $ pool $ conv3d_sep_norm pDown1 (0, 1, 1) aFullR
        aDown2R <- dump5' "s1-aDown2R" $ pool $ conv3d_sep_norm pDown2 (0, 1, 1) aDown1R
        aDown3R <- dump5' "s1-aDown3R" $ pool $ conv3d_sep_norm pDown3 (0, 1, 1) aDown2R
        aDown4R <- dump5' "s1-aDown4R" $ pool $ conv3d_sep_norm pDown4 (0, 1, 1) aDown3R

        let aCost0 = unsqueeze5_3
                   $ Disparity.costVolume 0 nDispMax16
                        (squeeze5_2 aDown4L) (squeeze5_2 aDown4R)

        aCost0  <- dump5' "s2-aCost0" aCost0
        aCost1  <- dump5' "s2-aCost1" $ conv3d_sep_norm pDisp1 (1, 1, 1) aCost0
        aCost2  <- dump5' "s2-aCost2" $ conv3d_sep_norm pDisp2 (1, 1, 1) aCost1
        aCost3  <- dump5' "s2-aCost3" $ conv3d_sep_norm pDisp3 (1, 1, 1) aCost2
        aCost4  <- dump5' "s2-aCost4" $ conv3d_sep_norm pDisp4 (1, 1, 1) aCost3
        aCost5  <- dump5' "s2-aCost5" $ conv3d_sep_norm pDisp5 (1, 1, 1) aCost4

        let aDisp16 = Disparity.regression $ squeeze5_3 aCost5

        -- Multiply by 16 to rescale disparity to be relative to full input range.
        -- This is the final output of the model.
        let aDispFull = mapAll (* 16) aDisp16

        -- Normalize the output disparity to the range [0,1] for rendering.
        let aNorm  = mapAll (/ fromIntegral nDispMaxFull) aDispFull

        -- Write out the rendered disparity map.
        let aImage = renderTurbo $ takeChannel3 0 aNorm
        writeBMP (dirOut ++ "/disparity.bmp") aImage


-------------------------------------------------------------------------------
-- | Padded separable convolution over the volumetric dimensions of a
--   rank-5 array, followed by batch normalisation and ReLU.
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


-- | Maxpool using 2x2 downsample.
pool :: Array5 Float -> Array5 Float
pool arr = unsqueeze5_2 $ Pool.maxpool (2, 2) False $ squeeze5_2 arr


-------------------------------------------------------------------------------
squeeze5_2 :: Array5 Float -> Array4 Float
squeeze5_2 arr
 = let  Shape5 nCount nChas 1 nRows nCols = shape arr
        sh = Shape4 nCount nChas nRows nCols
   in   reshape sh arr


unsqueeze5_2 :: Array4 Float -> Array5 Float
unsqueeze5_2 arr
 = let  Shape4 nCount nChas nRows nCols = shape arr
        sh = Shape5 nCount nChas 1 nRows nCols
   in   reshape sh arr

squeeze5_3 :: Array5 Float -> Array4 Float
squeeze5_3 arr
 = let  Shape5 nCount _ nLayers nRows nCols = shape arr
        sh = Shape4 nCount nLayers nRows nCols
   in   reshape sh arr

unsqueeze5_3 :: Array4 Float -> Array5 Float
unsqueeze5_3 arr
 = let  Shape4 nCount nLayers nRows nCols = shape arr
        sh = Shape5 nCount 1 nLayers nRows nCols
   in   reshape sh arr


-------------------------------------------------------------------------------
-- | Take a single channel from a rank-3 array.
takeChannel3 :: Int -> Array3 Float -> Array2 Float
takeChannel3 iImg arr
 = let  Shape3 _ nRows nCols = shape arr
   in   reshape (Shape2 nRows nCols)
         $ slicez3 arr (Index3 iImg 0 0) (Shape3 1 nRows nCols)


-------------------------------------------------------------------------------
-- | Dump output activations array.
dump5 :: Bool -> String -> String -> Array5 Float -> IO (Array5 Float)
dump5 enable dirOut name arr
 | enable
 = do   putStrLn $ "  - " ++ name

        -- Wait for evaluation of array.
        -- The Array constructor is strict in the element parameter,
        -- so this forces evaluation of the result before continuing.
        !x <- return arr

        dumpNormTurbo5 arr dirOut name
        return x
