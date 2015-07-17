{-# LANGUAGE TupleSections #-}
import Control.Lens
import Control.Monad

import qualified Data.Map as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Traversable (traverse)
import System.Random

import SpaceWeather.CmdArgs
import SpaceWeather.FlareClass
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import SpaceWeather.FeaturePack
import SpaceWeather.SkillScore
import System.System
import System.IO.Unsafe

type Genome = M.Map (String, FilePath) Bool

-- data Statistics :±

genome :: Lens' PredictionStrategyGS Genome
genome = featureSchemaPackUsed . fspFilenamePairs . l
  where
    l :: Lens' [(String, FilePath)] Genome
    l = lens g s
    g :: [(String, FilePath)] -> Genome
    g xs = M.union (M.fromList $ zip xs $ repeat True) defaultGenome

    s :: [(String, FilePath)] -> Genome -> [(String, FilePath)]
    s _ xs = map fst $ filter snd $ M.toList xs

mutate :: Genome -> IO Genome
mutate g = traverse flipper g
  where
    flipper :: Bool -> IO Bool
    flipper x = do
      r <- randomRIO (0,1 :: Double)
      return $ if r < 0.01 then (not x) else x

crossover :: Genome -> Genome -> IO Genome
crossover g1 g2 = traverse selecter $ M.unionWith justSame (setJust g1) (setJust g2)
  where
    selecter :: Maybe Bool -> IO Bool
    selecter (Just x) = return x
    selecter Nothing = do
      r <- randomRIO (0,1)
      return $ r < (0.5 :: Double)

    justSame :: Maybe Bool ->  Maybe Bool ->  Maybe Bool
    justSame a b
      | a==b      = a
      | otherwise = Nothing

    setJust = M.map Just

defaultGenome :: Genome
defaultGenome = unsafePerformIO $ do
  filesW <- readSystem0 "ls wavelet-features/*.txt"
  let
    featuresW :: [(String, FilePath)]
    featuresW = map ("f35Log",) $  map ("file://./" ++ ) $ words filesW
  filesB <- readSystem0 "ls forecast-features/backcast*.txt"
  let
    featuresB :: [(String, FilePath)]
    featuresB = map ("f25Log",) $  map ("file://./" ++ ) $ words filesB
  return $
    M.fromList $
    map (,False) $
    featuresB ++ featuresW

defaultStrategy :: PredictionStrategyGS
defaultStrategy = unsafePerformIO $ do
  Right s <- fmap decode $ T.readFile "resource/best-strategies-3/CClass.yml"
  return s

evaluate :: Genome -> IO Double
evaluate g = do
  ses <- performPrediction $ defaultStrategy & genome .~ g
  let res :: PredictionResult
      res = ses ^. predictionResult
      PredictionSuccess prMap = res
      val = prMap M.! MClassFlare M.! TrueSkillStatistic ^. scoreValue
  return val

main :: IO ()
main = withWorkDir $ do
  Right cclass <- fmap decode $ T.readFile "resource/best-strategies-3/CClass.yml"
  Right mclass <- fmap decode $ T.readFile "resource/best-strategies-3/MClass.yml"
  Right xclass <- fmap decode $ T.readFile "resource/best-strategies-3/XClass.yml"

  let population :: [Genome]
      population = map (^. genome) [cclass, mclass, xclass :: PredictionStrategyGS]
---  mapM_ T.putStrLn $ map encode
  vs <- mapM_ evaluate population
  print vs
  return ()
