
module Adops.Params where
import Adops.Array                      as A
import qualified Data.ByteString.Lazy   as BS
import qualified Data.Binary.Get        as BG
import Control.Monad


data Params
 = Conv3dSepNorm
 { paramCKrn     :: Array Shape5 Float
 , paramCBia     :: Array Shape1 Float
 , paramCBeta    :: Array Shape1 Float
 , paramCGamma   :: Array Shape1 Float

 , paramPKrn     :: Array Shape5 Float
 , paramPBia     :: Array Shape1 Float
 , paramPBeta    :: Array Shape1 Float
 , paramPGamma   :: Array Shape1 Float }
 deriving Show


readConv3dSepNorm :: FilePath -> String -> Shape5 -> IO Params
readConv3dSepNorm dir name shape
 = do   let Shape5 nChasOut nChasInp nKrnLayers nKrnRows nKrnCols = shape
        let ds = dims shape

        aCKrn   <- readFloat32LE dir (name ++ "c_krn")    (Shape5 nChasInp 1 nKrnLayers nKrnRows nKrnCols)
        aCBia   <- readFloat32LE dir (name ++ "c_bia")    (Shape1 nChasInp)
        aCBeta  <- readFloat32LE dir (name ++ "cb_beta")  (Shape1 nChasInp)
        aCGamma <- readFloat32LE dir (name ++ "cb_gamma") (Shape1 nChasInp)

        aPKrn   <- readFloat32LE dir (name ++ "p_krn")    (Shape5 nChasOut nChasInp 1 1 1)
        aPBia   <- readFloat32LE dir (name ++ "p_bia")    (Shape1 nChasOut)
        aPBeta  <- readFloat32LE dir (name ++ "pn_beta")  (Shape1 nChasOut)
        aPGamma <- readFloat32LE dir (name ++ "pn_gamma") (Shape1 nChasOut)
        return $ Conv3dSepNorm aCKrn aCBia aCBeta aCGamma aPKrn aPBia aPBeta aPGamma


readFloat32LE :: IsShape sh => FilePath -> String -> sh -> IO (Array sh Float)
readFloat32LE dirName baseName sh
 = do
        let fileName = concat
                [ dirName, "/"
                , baseName
                , "@", foldl1 (\acc d -> acc ++ "x" ++ d) $ map show $ dims sh
                , ".f32le" ]
        bs <- BS.readFile fileName
        return $ flip BG.runGet bs
         $ do   let count = size sh
                ls <- replicateM (size sh) BG.getFloatle
                return $ A.fromList sh ls
