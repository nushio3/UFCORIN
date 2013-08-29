import Control.Monad
import Data.Char
import Text.Printf

-- | count the number of alphabets

main :: IO ()
main = do
  str <- getContents
  forM_ (words str) $ printf "%s\t1\n" 
