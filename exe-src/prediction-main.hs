{-# LANGUAGE OverloadedStrings #-}
module Main where

import Control.Lens
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


process :: FilePath -> IO () 
process fn = do
  strE <- fmap decode $ HFS.readFile fn
  case strE of 
    Left msg -> putStrLn msg
    Right strategy -> do
      res <- performPrediction (strategy :: PredictionStrategyG)
      let
        candFn = strategy ^. predictionResultFile 
        resultFn 
          | candFn /= "" = candFn
          | ".yml" `isSuffixOf` fn = (++"-result.yml") $ reverse $ drop 4 $ reverse fn  
          | otherwise              = fn ++ ".result.yml" 
      HFS.writeFile resultFn $ encode res
      return ()
  