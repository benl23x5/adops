
module Adops.Codec where
import Adops.Array
import Data.Word
import qualified Codec.BMP              as BMP
import qualified Data.ByteString        as BS
import qualified Data.Vector.Unboxed    as VU


-- | Read a BMP file as a rank3 array in CHW order
--   The result values are in the range [0,1].
readBMP :: FilePath -> IO (Array3 Float)
readBMP filePath
 = do   Right bmp <- BMP.readBMP filePath
        let (width, height) = BMP.bmpDimensions bmp
        let bsRGBA = BMP.unpackBMPToRGBA32 bmp
        let shBMP  = Index3 height width 4

        return $ build3 (Shape3 3 height width) $ \(Index3 iCha iRow iCol) ->
         let  lix = toLinear shBMP $ Index3 iRow iCol iCha
              u8  = BS.index bsRGBA lix
          in   fromIntegral u8 / 255


-- | Write a BMP file from a rank3 array in CHW order,
--   The input values are clamped to the range [0,1].
writeBMP :: FilePath -> Array3 Float -> IO ()
writeBMP filePath arr
 = do
        let Index3 3 height width = shape arr
        let elemsRGBA = width * height * 4

        let make lix =
                let shRGBA = Shape3 height width 4

                    Index3 iRow iCol iCha
                     = fromLinear shRGBA lix

                    f32 | iCha == 3     = 1
                        | otherwise     = index3 arr (Index3 iCha iRow iCol)

                    u8  | f32 <= 0      = 0
                        | f32 >= 1      = 255
                        | otherwise     = truncate (f32 * 255) :: Word8

                in Just (u8, lix + 1)

        let (bsRGBA, _) = BS.unfoldrN elemsRGBA make 0
        let bmp = BMP.packRGBA32ToBMP32 width height bsRGBA
        BMP.writeBMP filePath bmp


-- | Render single channel data using the Google Turbo Color map.
--
--   Based on:
--    https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
--    https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
--    https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
--
--   Authors:
--     Colormap Design: Anton Mikhailov (mikhailov@google.com)
--     GLSL Approximation: Ruofei Du (ruofei@google.com)
--
renderTurbo :: Array2 Float -> Array3 Float
renderTurbo arr
 = let
        kRed   = (0.13572138,  4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943)
        kBlue  = (0.10667330, 12.64194608, -60.58204836, 110.36276771,  -89.90310912, 27.34824973)
        kGreen = (0.09140261,  2.19418839,   4.84296658, -14.18503333,    4.27729857,  2.82956604)

        clamp x
         | x <= 0    = 0
         | x >= 1    = 1
         | otherwise = x

        dot (a0, a1, a2, a3, a4, a5) (b0, b1, b2, b3, b4, b5)
         = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5

        vec x
         = let  x1 = clamp x
                x2 = x1 * x1
                x3 = x2 * x1
                x4 = x3 * x1
                x5 = x4 * x1
            in  (1.0, x1, x2, x3, x4, x5)

        Shape2 nRows nCols = shape arr

   in   build3 (Shape3 3 nRows nCols) $ \(Index3 iCha iRow iCol) ->
         let    v = vec $ index2 arr $ Index2 iRow iCol
                c | iCha == 0 = dot v kRed
                  | iCha == 1 = dot v kGreen
                  | iCha == 2 = dot v kBlue
         in     c
