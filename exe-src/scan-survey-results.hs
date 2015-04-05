{-# LANGUAGE OverloadedStrings #-}
module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.List
import System.Environment
import Text.Printf

import SpaceWeather.CmdArgs
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import qualified System.IO.Hadoop as HFS
import qualified Data.Text.IO as T

main :: IO ()
main = do
  argv <- getArgs
  mapM_ process argv


process :: FilePath -> IO ()
process fn = do
  Right result0 <- fmap decode $ T.readFile fn
  let result :: PredictionResult
      result = result0
  print result
