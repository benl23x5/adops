
module Demo.Stereo where
import Adops.Array
import Adops.Codec
import qualified Adops.Op.Disparity as Disparity



runStereo :: FilePath -> FilePath -> FilePath -> IO ()
runStereo pathBmpLeft pathBmpRight dirOut
 = do
        -- Load input images.
        putStrLn "* Loading input"
        aLeft3  <- readBMP pathBmpLeft
        aRight3 <- readBMP pathBmpRight

        -- Raw input size 540x960
        let Shape3 nChas nRows_ nCols_ = shape aLeft3

        -- Downsampled size 135x240
        let dRows  = 4
        let dCols  = 4
        let nRows  = nRows_ `div` dRows
        let nCols  = nCols_ `div` dCols

        let aLeft4  = reshape (Shape4 1 nChas nRows nCols)
                    $ decimate dRows dCols aLeft3

        let aRight4 = reshape (Shape4 1 nChas nRows nCols)
                    $ decimate dRows dCols aRight3

        -- Compute cost volume between left and right images.
        --  The maximum possible disparity is the number of columns in the image.
        let nCount  = nCols :: Int

        -- Compute disparity cost volume which shows how well each image element
        -- matches at each possible disparity.
        let aCost  = Disparity.costVolume 0 nCount aLeft4 aRight4

        -- Apply disparity regression to determine the position of the best
        -- match for each pixel.
        let aMin   = Disparity.regression aCost

        -- In practice the image pairs do not use the full disparity range,
        --  so use a smaller range for rendering. Large disparities will be
        --  clipped to the maximum value in the color map.
        let nRender = 160
        let aNorm  = mapAll (\x -> x / fromIntegral nRender) aMin

        -- Slice out the result and render it as a new image.
        let aSlice = reshape (Shape2 nRows nCols)
                   $ slicez3 aNorm (Index3 0 0 0) (Index3 1 nRows nCols)
        let aTurbo = renderGrey aSlice

        putStrLn "* Writing raw disparity"
        writeBMP (dirOut ++ "/disparity-raw.bmp") aTurbo


-- | Downsample input image in CHW order.
decimate :: Int -> Int -> Array3 Float -> Array3 Float
decimate dRows dCols arr
 = let  Shape3 nChas nRows nCols = shape arr
        nRows' = nRows `div` dRows
        nCols' = nCols `div` dCols
   in   build3 (Shape3 nChas nRows' nCols') $ \(Index3 iCha iRow iCol) ->
         index3 arr (Index3 iCha (iRow * dRows) (iCol * dCols))

