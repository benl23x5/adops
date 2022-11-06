
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
