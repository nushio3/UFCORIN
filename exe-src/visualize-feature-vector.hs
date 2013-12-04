module Main where

import Control.Monad
import Data.List
import qualified Data.Map.Strict as Map
import Data.Maybe
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Safe
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

fnFeatures :: [((Double,Double),FilePath)]
fnFeatures = unsafePerformIO $ do 
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
      str1 <- readFile fn1
      strF <- readFile fnForecast
      
      let stream1 :: [String]
          stream1 = catMaybes $ map (flip atMay 4) $ map words $ lines str1
      let streamF :: [String]
          streamF = catMaybes $ map (flip atMay 4) $ map words $ lines strF
      
          vals :: [Double]
          vals = sort $ map read stream1
          
          small =  maybe 1 id $ headMay $ drop (length vals `div` 100) vals
          large =  maybe 10 id $ headMay $ drop (length vals `div` 100) $ reverse vals
          
      writeFile fn2 $ unlines $ zipWith (printf "%s %s") streamF stream1
      return ((small,large), fn2)
      
      

plotCmd :: String
plotCmd = unlines $
  [ "set term postscript enhanced color solid 20"
  , "set log xy"
  , "set out 'test.eps'"
  , "set xlabel 'GOES flux (24hour forecast max)'"
  , "set ylabel 'feature vector component'"
  ] ++ map go fnFeatures 
  where
    go :: ((Double, Double),FilePath) -> String
    go ((small, large),fn) = unlines
      [ printf "set title '%s'" fn
      , printf "set yrange [%f:%f]" small large
      , printf "plot '%s' u 1:2" fn]



main :: IO ()
main = do
  _ <- readProcess "gnuplot" [] plotCmd
  return ()

