import Control.Monad
import Data.Char
import Text.Printf

main :: IO ()
main = do
  str <- getContents
  forM_ str $ \c ->
    when (isAlphaNum c) $ printf "%c 1\n" c
