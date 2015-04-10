module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.List
import qualified Data.Text.IO as T
import System.Environment
import System.Process
import Text.Printf

import SpaceWeather.CmdArgs
import SpaceWeather.FeaturePack
import SpaceWeather.FlareClass
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import qualified System.IO.Hadoop as HFS

surveyDir = "survey-noise"

main :: IO ()
main = do
  system $ "mkdir -p " ++ surveyDir
  process defaultFlareClasses

process :: FlareClass -> IO ()
process className = do
  strE <- fmap decode $ T.readFile $ printf "resource/best-strategy/%s.yml" className
  case strE of
    Left msg -> putStrLn msg
    Right strategy -> do

genStrategy :: FlareClass -> PredictionStrategyGS -> Int -> IO ()
genStrategy className strategy0 iterNum = do
  let
    strategy2 :: PredictionStrategyGS
    strategy2 = strategy0
      & featureSchemaPackUsed . fspFilenamePairs %~ (++ fnPairs)


    fn :: String
    fn = printf "%s/%s-%02d-strategy.yml" surveyDir className iterNum

  T.writeFile fn $ encode (strategy2)

  return ()
