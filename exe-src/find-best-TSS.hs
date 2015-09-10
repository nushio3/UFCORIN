#!/usr/bin/env runhaskell
module Main where
import Control.Monad
import Control.Lens
import Data.Function (on)
import Data.List (sortBy)
import qualified Data.Map as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import System.Environment
import Text.Printf

import SpaceWeather.FlareClass
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.SkillScore
import System.System

readResult :: FilePath -> IO PredictionResult
readResult fn = do
  txt <- T.readFile fn
  let Right ret = decode txt
  return ret

tssOf :: FlareClass -> PredictionResult -> Double
tssOf fc res =
  case res of
   PredictionSuccess prMap ->
     prMap M.! fc M.! TrueSkillStatistic ^. scoreValue
   _ -> 0



main :: IO ()
main = do
  dirName <- fmap head getArgs
  fns <- fmap lines $ readSystem0 $ printf "find %s | grep 'result.yml'" dirName
  resultSet <- mapM readResult fns
  forM_ defaultFlareClasses $ \fc -> do
    let best = last $ sortBy (compare `on` tssOf fc) resultSet
    print (fc, tssOf fc best)
    print best
