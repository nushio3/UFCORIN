import Control.Monad
import Text.Printf

main :: IO ()
main = do
  str <- getContents
  forM_ str $ \c ->
    printf "%c 1\n" c
