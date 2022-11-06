
module Demo.Sobel where
import Adops.Array
import Adops.Codec
import Adops.Op.Conv


-- | Apply the Sobel edge detection convolution kernels to the given image,
--   writing out the result to the given directory.
--
--   https://www.sciencedirect.com/topics/engineering/sobel-edge-detection
--
runSobel :: FilePath -> FilePath -> IO ()
runSobel pathBmp dirOut
  = do
      aOutput3 <- applySobel pathBmp
      putStrLn "* Writing Sobel edge-map"
      writeBMP (dirOut ++ "/edges.bmp") aOutput3

applySobel :: FilePath -> IO (Array3 Float)
applySobel pathBmp
  = do
      putStrLn "* Loading input"
      aInput3 <- readBMP pathBmp
      let Shape3 nChas nRows nCols = shape aInput3

      let aOutputs = applySobelRGB aInput3
      let aOutput3 = packChas3
            (Shape3 3 nRows nCols)
            $ map (reshape (Shape2 nRows nCols)) aOutputs
      return aOutput3

applySobelRGB :: Array3 Float -> [Array4 Float]
applySobelRGB aInput3
  = let
      Shape3 nChas nRows nCols = shape aInput3

      -- Separate the RGB channels.
      aInput4   = reshape (Shape4 1 nChas nRows nCols) aInput3
      aInput4_R = slicez4 aInput4 (Index4 0 0 0 0) (Index4 1 1 nRows nCols)
      aInput4_G = slicez4 aInput4 (Index4 0 1 0 0) (Index4 1 1 nRows nCols)
      aInput4_B = slicez4 aInput4 (Index4 0 2 0 0) (Index4 1 1 nRows nCols)

      -- Treat an image as a function f(x,y). Sobel filters approximate the
      -- partial gradients along x and y. Take the norm at each point to
      -- get the gradient.
      sobel arr =
        let aEdgeX = conv2d_pad (1, 1) sobelX arr
            aEdgeY = conv2d_pad (1, 1) sobelY arr
        in  zipWith4 norm aEdgeX aEdgeY

      -- Apply the Sobel kernels separately to each channel and combine
      -- the results.
      aOutput4_R = sobel aInput4_R
      aOutput4_G = sobel aInput4_G
      aOutput4_B = sobel aInput4_B

    in [aOutput4_R, aOutput4_G, aOutput4_B]


-- | Vector norm.
norm :: Float -> Float -> Float
norm x y =
  sqrt (x**2 + y**2)

-- | Horizontal Sobel kernel.
sobelX :: Array4 Float
sobelX = fromList (Shape4 1 1 3 3) $ map (/4.0)
      [  1,  0, -1
      ,  2,  0, -2
      ,  1,  0, -1 ]

-- Vertical Sobel kernel.
sobelY :: Array4 Float
sobelY = fromList (Shape4 1 1 3 3) $ map (/4.0)
      [  1,  2,  1
      ,  0,  0,  0
      , -1, -2, -1 ]

sobelXY :: Array4 Float
sobelXY =
  zipWith4 norm sobelX sobelY