{-# LANGUAGE OverloadedStrings #-}
module Main where

import Control.Lens
import Control.Monad
import Data.Char
import Data.Maybe
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Time
import System.Environment
import Text.Printf

import SpaceWeather.Format
import SpaceWeather.TimeLine

-- AIA/AIA20100701_010006_0193.fits     5.1739596800e+09

main :: IO ()
main = getArgs >>= mapM_ process


process :: FilePath -> IO ()
process fn0 = do
  txt0 <- T.readFile fn0
  let ls0 = T.lines txt0
      lsx = mapMaybe parse ls0

      parse :: T.Text -> Maybe (TimeBin, Double)
      parse txt1 = do
        let txt2 = T.dropWhile (not . isDigit) txt1
        iyear <- readMaySubT 0 4 txt2
        imon  <- readMaySubT 4 2 txt2
        iday  <- readMaySubT 6 2 txt2
        ihour <- readMaySubT 9 2 txt2
        let day = fromGregorian iyear imon iday
            sec = secondsToDiffTime $ ihour * 3600
            t = UTCTime day sec
        val <- readAt (T.words txt1) 1
        Just (t ^. discreteTime, val)
        
  forM_ lsx $ (\(t,v) -> printf "%6d %20e\n" t v )