module Main where

import Control.Lens
import qualified Data.Map as M
import qualified Data.Text.IO as T

import SpaceWeather.Feature
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General


main :: IO ()
main = do
  Right strategy <- fmap decode $ T.readFile "resource/strategy-template-local.yml"
  Right goesFeature <- loadFeatureWithSchema
                       (strategy ^. predictionTargetSchema)
                       (strategy ^. predictionTargetFile)
  return (strategy :: PredictionStrategyGS)

  print $ length goesFeature
  print $ sum $ map fst $ M.toList $ goesFeature
