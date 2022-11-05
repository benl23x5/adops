
module Adops.Array.Elem where
import Adops.Array.Shape
import qualified Data.Vector.Unboxed    as U


class U.Unbox a => Elem a where
 zero   :: a
 one    :: a

instance Elem Float where
 zero   = 0
 one    = 1

instance Elem Int where
 zero   = 0
 one    = 1



