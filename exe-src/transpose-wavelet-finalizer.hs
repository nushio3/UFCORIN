{-# LANGUAGE OverloadedStrings #-}

import Control.Monad
import Data.Function (on)
import Data.List (groupBy)
import Data.Maybe
import qualified Data.Map as Map
import qualified Data.Text as T
import qualified Data.Text.IO as T
import System.Process (system)
import Text.Printf

import SpaceWeather.Timeline (parseTimeBin)
import SpaceWeather.Text

main :: IO ()
main = do
  let 
    outDir :: String
    outDir = "wavelet-features"
  system $ printf "mkdir -p %s" outDir
  con <- T.getContents         
  let 
    grouped =
      map compact $ 
      groupBy ((==) `on` fst) $
      map (T.breakOn " ") $
      T.lines con
    compact :: [(a,b)] -> (a,[b])
    compact xs = (fst $ head xs, map snd xs)
    
  forM_ grouped $ \(k,strs) -> do
    let 
      outFn :: FilePath
      outFn = printf "%s/%s.txt" outDir
        (T.unpack $ k)

      editLine :: T.Text -> Maybe T.Text
      editLine str = do
        let ws = T.words str
        tb <- parseTimeBin ws
        return $ T.unwords $ take 2 ws ++ [showT tb] ++ drop 2 ws
    T.writeFile outFn $ 
      T.unlines $ mapMaybe editLine $ strs