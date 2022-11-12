
module Demo.Ascent where
import Data.List
import Adops.Array
import Adops.Codec
import Adops.Op.Conv
import Demo.Sobel

-- | Given an image I and a base image B, mutate B with gradient ascent
--   such that B takes on the Sobel edges of I.
--
runAscent :: FilePath -> FilePath -> IO ()
runAscent pathBmp dirOut =
  do
    putStrLn "* Loading input"
    aInput3 <- readBMP pathBmp
    let Shape3 nChas nRows nCols = shape aInput3

    -- Apply Sobel edge detection to the given image.
    let [aOutput4_R, aOutput4_G, aOutput4_B] = applySobelRGB aInput3

    -- Create a base black image.
    let aBase = fill (Shape4 1 1 nRows nCols) 0

    -- Gradient ascent the base image for 100 iterations. It should
    -- gradually start to take on the edges of the original image.
    let aAsc4_R = doAscent 100 0.01 aOutput4_R aBase
    let aAsc4_G = doAscent 100 0.01 aOutput4_G aBase
    let aAsc4_B = doAscent 100 0.01 aOutput4_B aBase

    let aAsc3 = packChas3
          (Shape3 3 nRows nCols)
          $ map (reshape (Shape2 nRows nCols))
              [aAsc4_R, aAsc4_G, aAsc4_B]

    putStrLn "* Writing activation maximisation result"
    writeBMP (dirOut ++ "/maximise.bmp") aAsc3

doAscent :: Int -> Float -> Array4 Float -> Array4 Float -> Array4 Float
doAscent n rate aOut aBase =
  let ascend acc _ =
        let aGrad = conv2d_dInp sobelXY aOut
        in  acc +. (mapAll (* rate) aGrad)
  in  foldl' ascend aBase [0..n]