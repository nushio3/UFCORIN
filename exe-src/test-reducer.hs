import Control.Monad
import Data.Function (on)
import Data.List (span, groupBy)
import Text.Printf

-- | sum all the numbers that belongs to a same key.

main :: IO ()
main = do
  str <- getContents
  mapM_ putStrLn $ map pprint $ groupBy ((==) `on` fst) $ 
    map (span (not . flip elem "\t ")) $ lines str
  where
    pprint :: [(String, String)] -> String
    pprint xs = printf "%s\t%d" 
      (fst $ head xs)
      (sum $ map (read . snd) xs :: Int)
