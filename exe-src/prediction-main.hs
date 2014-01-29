{-# LANGUAGE OverloadedStrings #-}
module Main where

import Control.Lens
import Control.Monad
import Data.List
import System.Environment
       
import SpaceWeather.CmdArgs
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import qualified System.IO.Hadoop as HFS

main :: IO ()
main = do
  fns <- getArgs
  mapM_ process fns 


process :: FilePath -> IO () 
process fn = withWorkDir $ do
  strE <- fmap decode $ HFS.readFile fn
  case strE of 
    Left msg -> putStrLn msg
    Right strategy -> do
      res <- performPrediction (strategy :: PredictionStrategyG)
      let
        candSesFn = strategy ^. predictionSessionFile 
        candResFn = strategy ^. predictionResultFile 
        finalSesFn 
          | candSesFn /= "" = candSesFn
          | ".yml" `isSuffixOf` fn = (++"-session.yml") $ reverse $ drop 4 $ reverse fn  
          | otherwise              = fn ++ ".result.yml" 
        finalResFn
          | candResFn /= "" = candResFn
          | ".yml" `isSuffixOf` fn = (++"-result.yml") $ reverse $ drop 4 $ reverse fn  
          | otherwise              = fn ++ ".result.yml" 
      HFS.writeFile finalSesFn $ encode (res ^. predictionResult)
      HFS.writeFile finalResFn $ encode res
      return ()
  