
module Demo.Stereo where
import Adops.Array
import Adops.Codec


runStereo :: FilePath -> FilePath -> FilePath -> IO ()
runStereo pathBmpLeft pathBmpRight pathBmpOut
 = do   bmpLeft  <- readBMP pathBmpLeft
        bmpRight <- readBMP pathBmpRight

        writeBMP pathBmpOut bmpLeft

