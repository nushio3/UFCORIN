{-# LANGUAGE OverloadedStrings #-}
module Main where

import Control.Monad
import Data.List
import System.Environment
       
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import qualified System.IO.Hadoop as HFS

main :: IO ()
main = do
  fns <- getArgs
  mapM_ process fns 

outputFileName :: FilePath -> FilePath
outputFileName fn
  | ".yml" `isSuffixOf` fn = (++"-result.yml") $ reverse $ drop 4 $ reverse fn  
  | otherwise              = fn ++ ".result.yml" 

process :: FilePath -> IO () 
process fn = do
  strE <- fmap decode $ HFS.readFile fn
  let resultFn = outputFileName fn
  case strE of 
    Left msg -> putStrLn msg
    Right strategy -> do
      res <- performPrediction (strategy :: PredictionStrategyG)
      HFS.writeFile resultFn $ encode res
      return ()
  