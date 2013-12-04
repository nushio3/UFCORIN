{-# LANGUAGE TupleSections #-}
module Main where

import Control.Monad
import Data.List
import qualified Data.Map.Strict as Map
import Data.Maybe
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Safe
import System.IO
import System.IO.Unsafe 
import System.Process
import Text.Printf

import SpaceWeather.Text
import SpaceWeather.TimeLine

-- the filename to be forecasted.
fnForecast :: FilePath
fnForecast = "forecast/forecast-goes-24.txt"

goesForecastCurve :: TimeLine Double
goesForecastCurve = unsafePerformIO $ do
  str <- T.readFile fnForecast
  let xss :: [[T.Text]]
      xss = map T.words $ T.lines str
      
      parseLine :: [T.Text] -> Maybe (TimeBin, Double)
      parseLine ws = do
        t <- readAt ws 1
        v <- readAt ws 4
        return (t,v)
  return $ Map.fromList $ catMaybes $ map parseLine xss

data FeatureCurve
  = FeatureCurve
  { tag :: String
  , range :: (Double, Double)
  , timeLine :: TimeLine Double
  }
  

featureCurves :: [FeatureCurve]
featureCurves = unsafePerformIO $ do 
  str <- readProcess "ls" [dir1] ""
  let fns0 = 
        filter (not . isInfixOf "-0-") $
        filter (not . isInfixOf "-0.txt") $
        words str
      fns1 = map (dir1++) $ fns0
      fns2 = map (dir2++) $ fns0
  zipWithM go fns1 fns2      

  where
    dir1 = "wavelet-features/"
    dir2 = "work/"    
    
    go fn1 fn2 = do
      hPutStrLn stderr $ printf "processing %s..." fn1
      str1 <- T.readFile fn1
      
      let tl :: TimeLine Double
          tl = Map.fromList $ catMaybes $ map parse $ map T.words $ T.lines str1

          parse :: [T.Text] -> Maybe (TimeBin, Double)
          parse ws = do
            t <- readAt ws 2
            v <- readAt ws 4
            return (t,v)
      
          vals :: [Double]
          vals = sort $ Map.elems tl
          
          small =  maybe 1 id $ headMay $ drop (length vals `div` 100) vals
          large =  maybe 10 id $ headMay $ drop (length vals `div` 100) $ reverse vals
          
          mixTL = Map.intersectionWith (,) goesForecastCurve tl
          
      T.writeFile fn2 $ T.unlines $ [T.pack $ printf "%f %f" v1 v2 | (_,(v1,v2)) <- Map.toList mixTL]
      return $ FeatureCurve {range = (small,large), tag = fn2, timeLine = tl}
      
      

plotCmd :: String
plotCmd = unlines $
  [ "set term postscript enhanced color solid 20"
  , "set log xy"
  , "set out 'test.eps'"
  , "set xlabel 'GOES flux (24hour forecast max)'"
  , "set ylabel 'feature vector component'"
  , "set xrange [1e-8:1e-3]" 
  ] ++ map go featureCurves
  where
    go :: FeatureCurve -> String
    go fc = 
      let (small,large) = range fc
          fn = tag fc
      in unlines
      [ printf "set title '%s'" fn
      , printf "set yrange [%f:%f]" small large
      , printf "plot '%s' u 1:2" fn]



main :: IO ()
main = do
  _ <- readProcess "gnuplot" [] plotCmd
  return ()

