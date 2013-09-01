{-# LANGUAGE OverloadedStrings #-}

import Control.Monad
import Control.Spoon(spoon)
import Data.Function (on)
import Data.List (groupBy)
import Data.Maybe
import qualified Data.Text as T
import qualified Data.Text.IO as T


main :: IO ()
main = do
  con <- T.getContents
  mapM_ T.putStrLn $ 
    mapMaybe makeLine $ 
    groupBy ((==) `on` fst) $
    map (T.breakOn "\t") $ T.lines con

makeLine :: [(T.Text,T.Text)] -> Maybe T.Text
makeLine strs = spoon $ T.intercalate "\t" [key,T.unwords [idVal,sqVal]] 
  where
    key = fst $ head strs
    
    vals = map (T.words . snd) strs
    idVal = (!!1) $ head $ filter ((=="id") . head) vals
    sqVal = (!!1) $ head $ filter ((=="sq") . head) vals

