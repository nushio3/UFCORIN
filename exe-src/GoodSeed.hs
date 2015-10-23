module GoodSeed where

import Control.Lens
import qualified Data.Map as M
import qualified Data.Text.IO as T
import System.Random

import SpaceWeather.TimeLine
import SpaceWeather.FlareClass
import SpaceWeather.Feature
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General


goodSeeds :: Int -> IO [Int]
goodSeeds n = do
  Right strategy <- fmap decode $ T.readFile "resource/strategy-template-local.yml"
  Right goesFeature <- loadFeatureWithSchema
                       (strategy ^. predictionTargetSchema)
                       (strategy ^. predictionTargetFile)
  return (strategy :: PredictionStrategyGS)
  putStrLn "PROG: begin generate seed"
  collectIO n $ getGoodSeed goesFeature

collectIO :: Int -> IO [a] -> IO [a]
collectIO n m
  | n <= 0    = return []
  | otherwise = do
      xs <- m
      ys <- collectIO (n - length xs) m
      return $ xs ++ ys

getGoodSeed :: Feature -> IO [Int]
getGoodSeed goesFeature = do
  seed <- randomIO

  let
    cvstr = CVShuffled seed CVWeekly
    pred :: TimeBin -> Bool
    pred = inTrainingSet cvstr

    (trainSet, testSet) = M.partitionWithKey (\k _ -> pred k) goesFeature

    countBalance :: FlareClass -> (Double, Double)
    countBalance fc = (n1,n2) where
      countX = length . filter (\v -> v >= log (xRayFlux fc) / log 10) . map snd . M.toList
      n1 = fromIntegral $ countX trainSet
      n2 = fromIntegral $ countX testSet

    isBalanced (n1,n2) = n1 < 1.1 * n2 && n2 < 1.1 * n1

    balances = fmap countBalance defaultFlareClasses
  if all isBalanced balances then print balances >> print seed >> return [seed]
    else return []
