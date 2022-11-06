
module Demo.Stereo where
import Adops.Array
import Adops.Codec


runStereo :: FilePath -> FilePath -> FilePath -> IO ()
runStereo pathBmpLeft pathBmpRight pathBmpOut
 = do   bmpLeft  <- readBMP pathBmpLeft
        bmpRight <- readBMP pathBmpRight

        let Shape3 nChas nRows nCols = shape bmpLeft
        let aRed  = reshape (Shape2 nRows nCols)
                  $ slicez3 bmpLeft (Index3 0 0 0) (Shape3 1 nRows nCols)

        let aTurbo = renderTurbo aRed

        writeBMP pathBmpOut aTurbo

