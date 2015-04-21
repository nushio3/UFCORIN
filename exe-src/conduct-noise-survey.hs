module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.List
import qualified Data.Map as M
import qualified Data.Text.IO as T
import qualified Statistics.Sample as Stat
import qualified Data.Vector.Unboxed as V
import System.Environment
import System.Process
import System.System
import Text.Printf

import SpaceWeather.CmdArgs
import SpaceWeather.FeaturePack
import SpaceWeather.FlareClass
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import SpaceWeather.SkillScore
import qualified System.IO.Hadoop as HFS

meanDevi :: [Double] -> (Double, Double)
meanDevi xs = let vx = V.fromList xs in (Stat.mean vx, sqrt $ Stat.varianceUnbiased vx)

surveyDir = "survey-noise"

pprClass :: FlareClass -> String
pprClass = take 6 . show

data Mode = SummarizeMode | GenerateMode

main :: IO ()
main = do
  system $ "mkdir -p " ++ surveyDir
  (modeStr:_) <- getArgs
  let mode = case modeStr of
        "-S" -> SummarizeMode
        "-G" -> GenerateMode
        _ -> error "specify either -S or -G"

  cmds <- mapM (process mode) defaultFlareClasses
--  readSystem "xargs -P 8 -L 1 -d  '\\n' ./dist/build/prediction-main/prediction-main" $ unlines $ concat cmds
  return ()

process :: Mode -> FlareClass -> IO [String]
process mode className = do
  strE <- fmap decode $ T.readFile $ printf "resource/best-strategies-3/%s.yml" (pprClass className)
  case strE of
    Left msg -> putStrLn msg >> return []
    Right strategy -> do
      let spNoises :: [Double]
          spNoises = map (/1000) [0..9] ++ map (/100) [1..10] ++ map (/10) [1..10]
          tpNoises :: [Double]
          tpNoises = [0..9] ++ map (*10) [1..9] ++ map (*100) [1..9] ++ map (*1000) [1..20]
      case mode of
        GenerateMode -> do
          xs <- sequence $
             [genStrategy className ("--sn" ++ show opt) strategy i
             | i <- [0..9], opt <- spNoises]
          xt <- sequence $
             [genStrategy className ("--tn" ++ show opt) strategy i
             | i <- [0..9], opt <- tpNoises]
          return $ xs++xt
        SummarizeMode -> do
          summarize className "--sn" spNoises [0..9]
          summarize className "--tn" tpNoises [0..9]
          return []

summarize :: FlareClass -> String -> [Double] -> [Int] -> IO ()
summarize className optionString noiseArgs iters = do
  statLines <- forM noiseArgs $ \noiseArg -> do
    scores <- fmap concat $ forM iters $ \iterNum -> do
      let fnResult = mkFnBody className (optionString ++ show noiseArg) iterNum ++ "-result.yml"
      resE <- fmap decode $ T.readFile fnResult
      case resE of
        Left msg -> putStrLn msg >> return []
        Right result -> do
          let
              prm = (result :: PredictionResult) ^. predictionResultMap
              xcm = prm M.! className M.! TrueSkillStatistic
              xcsv = xcm ^. scoreValue
          return [xcsv]
    let (m0,d0) = meanDevi scores
    return $ unwords $ map show [noiseArg, m0, d0]
  writeFile (mkFnBody className (optionString ++"-summary") 0 ++ ".txt") $
    unlines statLines
  return ()

genStrategy :: FlareClass -> String -> PredictionStrategyGS -> Int -> IO String
genStrategy className optionString strategy0 iterNum = do
  let
    strategy2 :: PredictionStrategyGS
    strategy2 = strategy0
          & predictionSessionFile .~ ""
          & predictionResultFile .~ ""
          & predictionRegressionFile .~ ""

    fn = fnBody ++ "-strategy.yml"
    fnSh = fnBody ++ ".sh"
    fnBody :: String
    fnBody = mkFnBody className optionString  iterNum

  T.writeFile fn $ encode (strategy2)
  writeFile fnSh $ printf "./dist/build/prediction-main/prediction-main %s %s" optionString fn
  return $ printf "%s %s" optionString fn

mkFnBody :: FlareClass -> String -> Int -> String
mkFnBody className optionString iterNum
  = printf "%s/%s[%s]-%02d" surveyDir (pprClass className) optionString iterNum
