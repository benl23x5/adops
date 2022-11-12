
module Adops.Op.Pool where
import Adops.Array
import qualified Adops.Op.Reduce     as A
import qualified Data.Vector.Unboxed as U

-- | Maxpool applied to non-overlapping patches in the spatial dimensions
--   of a rank-4 array.
--
maxpool_discrete
  :: (Int, Int) -> Bool -> Array4 Float -> Array4 Float
maxpool_discrete (hK, wK) ceil aI =
  let Shape4 nImg nCha hI wI = shape aI
      shK   = Shape4 1 1 hK wK
      hO    = hI `div` hK
      wO    = wI `div` wK
      slice = if ceil then slicez4 else slice4
  in  build4 (Shape4 nImg nCha hO wO) $ \(Index4 iImg iCha h w) ->
        let iI = Shape4 iImg iCha (h + hK) (w + wK)
            sI = slice  aI iI shK
        in  A.max sI
