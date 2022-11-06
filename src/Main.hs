
import Adops.Array
import qualified System.Environment as S
import qualified Demo.Stereo

main :: IO ()
main
 = do   args <- S.getArgs
        case args of
         ["stereo", pathBmpLeft, pathBmpRight, dirOut]
          -> Demo.Stereo.runStereo pathBmpLeft pathBmpRight dirOut

         _ -> putStrLn $ unlines
                [ "usage: adops stereo <LEFT.bmp> <RIGHT.bmp> <DIR_OUT>"]


