import Control.Monad
import Data.Function (on)
import Data.List (span, groupBy)
import Text.Printf

main :: IO ()
main = do
  str <- getContents
  mapM_ putStrLn $ map parse $ groupBy ((==) `on` fst) $ map (span (=='\t')) $ lines str
  where
    parse :: [(String, String)] -> String
    parse xs = printf "%s %d" (fst $ head xs)
      (sum $ map (read . snd) xs :: Int)
