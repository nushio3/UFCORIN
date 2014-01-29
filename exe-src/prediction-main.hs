module Main where
       
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General

main :: IO ()
main = do
  performPrediction defaultPredictionStrategy
  return ()
  