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
import qualified Data.Text.IO as T

main :: IO ()
main = do
  fns <- getArgs
  mapM_ process fns 


process :: FilePath -> IO () 
process fn = withWorkDir $ do
  strE <- fmap decode $ T.readFile fn
  case strE of 
    Left msg -> putStrLn msg
    Right strategy -> do
      let
        strategy2 = strategy 
          & predictionSessionFile .~ finalSesFn        
          & predictionResultFile .~ finalResFn        
          & predictionRegressionFile .~ finalRegFn        

        candSesFn = strategy ^. predictionSessionFile 
        candResFn = strategy ^. predictionResultFile 
        candRegFn = strategy ^. predictionRegressionFile 

        finalSesFn 
          | candSesFn /= "" = candSesFn
          | ".yml" `isSuffixOf` fn = (++"-session.yml") $ reverse $ drop 4 $ reverse fn  
          | otherwise              = fn ++ ".session.yml" 

        finalResFn
          | candResFn /= "" = candResFn
          | ".yml" `isSuffixOf` fn = (++"-result.yml") $ reverse $ drop 4 $ reverse fn  
          | otherwise              = fn ++ ".result.yml" 

        finalRegFn
          | candRegFn /= "" = candRegFn
          | ".yml" `isSuffixOf` fn = (++"-regres.txt") $ reverse $ drop 4 $ reverse fn  
          | otherwise              = fn ++ ".regress.txt" 


      res <- performPrediction (strategy2 :: PredictionStrategyGS)


      putStrLn $ "FN: " ++ finalResFn
      T.writeFile finalResFn $ encode (res ^. predictionResult)
      T.writeFile finalSesFn $ encode res
      return ()
  
