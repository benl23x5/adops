
import Adops.Array
import qualified System.Environment as S
import qualified Demo.Stereo

main :: IO ()
main
 = do   args <- S.getArgs
        case args of
         ["stereo", pathBmpLeft, pathBmpRight, pathBmpOut]
          -> Demo.Stereo.runStereo pathBmpLeft pathBmpRight pathBmpOut

         _ -> putStrLn $ unlines
                [ "usage: adops stereo <LEFT.bmp> <RIGHT.bmp> <OUT.bmp>"]


