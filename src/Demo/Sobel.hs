
module Demo.Sobel where
import Adops.Array
import Adops.Codec
import Adops.Op.Conv


-- | Apply the Sobel edge detection convolution kernel to the given image,
--   writing out the result to the given directory.
--
--   Sobel filters: https://en.wikipedia.org/wiki/Sobel_operator
--
runSobel :: FilePath -> FilePath -> IO ()
runSobel pathBmp dirOut
 = do
        -- Horizontal Sobel kernel.
        let krnX = fromList (Shape4 1 1 3 3) $ map (/4.0)
              [  1,  0, -1
              ,  2,  0, -2
              ,  1,  0, -1 ]

        let krnY = fromList (Shape4 1 1 3 3) $ map (/4.0)
              [  1,  2,  1
              ,  0,  0,  0
              , -1, -2, -1 ]

        -- Load the input image.
        putStrLn "* Loading input"
        aInput3 <- readBMP pathBmp

        let Shape3 nChas nRows nCols = shape aInput3

        -- Separate the RGB channels.
        let aInput4   = reshape (Shape4 1 nChas nRows nCols) aInput3
        let aInput4_R = slicez4 aInput4 (Index4 0 0 0 0) (Index4 1 1 nRows nCols)
        let aInput4_G = slicez4 aInput4 (Index4 0 1 0 0) (Index4 1 1 nRows nCols)
        let aInput4_B = slicez4 aInput4 (Index4 0 2 0 0) (Index4 1 1 nRows nCols)

        -- Sobel kernels compute an approximation of the gradient along the
        -- horizontal and vertical axes of the image. Take the magnitude of
        -- the vector at each point to get the gradient.
        let sobel arr =
              let aEdgeX = conv2d_pad (1, 1) krnX arr
                  aEdgeY = conv2d_pad (1, 1) krnY arr
              in  zipWith4
                    (\x y -> sqrt (x**2 + y**2))
                    aEdgeX aEdgeY

        -- Apply the Sobel kernels separately to each channel and combine
        -- the results.
        let aOutput2_R = reshape (Shape2 nRows nCols) $ sobel aInput4_R
        let aOutput2_G = reshape (Shape2 nRows nCols) $ sobel aInput4_G
        let aOutput2_B = reshape (Shape2 nRows nCols) $ sobel aInput4_B
        let aOutput3   = packChas3
              (Shape3 3 nRows nCols)
              [aOutput2_R, aOutput2_G, aOutput2_B]

        putStrLn "* Writing Sobel output"
        writeBMP (dirOut ++ "/sobel.bmp") aOutput3