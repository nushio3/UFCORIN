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

pprClass :: FlareClass -> String
pprClass = take 6 . show

main :: IO ()
main = do
  system $ "mkdir -p " ++ surveyDir
  mapM_ process defaultFlareClasses

process :: FlareClass -> IO [FilePath]
process className = do
  strE <- fmap decode $ T.readFile $ printf "resource/best-strategies-3/%s.yml" (pprClass className)
  case strE of
    Left msg -> putStrLn msg >> return []
    Right strategy -> sequence $
      [genStrategy className strategy i | i <- [0..9]]

genStrategy :: FlareClass -> PredictionStrategyGS -> Int -> IO FilePath
genStrategy className strategy0 iterNum = do
  let
    strategy2 :: PredictionStrategyGS
    strategy2 = strategy0
          & predictionSessionFile .~ ""
          & predictionResultFile .~ ""
          & predictionRegressionFile .~ ""

    fn :: String
    fn = printf "%s/%s-%02d-strategy.yml" surveyDir (pprClass className) iterNum

  T.writeFile fn $ encode (strategy2)

  return fn
