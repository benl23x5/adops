
import Adops.Array
import qualified System.Environment as S
import qualified Demo.Stereo
import qualified Demo.Sobel
import qualified Demo.Ascent

main :: IO ()
main
 = do   args <- S.getArgs
        case args of
         ["sobel", pathBmp, dirOut]
          -> Demo.Sobel.runSobel pathBmp dirOut

         ["stereo", pathBmpLeft, pathBmpRight, dirParams, dirOut]
          -> Demo.Stereo.runStereo pathBmpLeft pathBmpRight dirParams dirOut

         ["ascent", pathBmp, dirOut]
          -> Demo.Ascent.runAscent pathBmp dirOut

         _ -> putStrLn $ unlines
                [ "usage: adops stereo <LEFT.bmp> <RIGHT.bmp> <DIR_OUT>"]


