{-# LANGUAGE TupleSections #-}
import Control.Lens
import Control.Monad
import qualified Control.Concurrent.ParallelIO.Global as P

import Data.Function (on)
import Data.List
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

pprGenome :: Genome -> String
pprGenome g = concat $ map (show . fromEnum) $ map snd $ M.toList g

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

chooseN :: Int -> [a] -> IO [a]
chooseN n xs = do
  rxs <- forM xs $ \x -> do
    r <- randomRIO (0,1 :: Double)
    return (r,x)
  return $ take n $ map snd $ sortBy (compare `on` fst) rxs


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
  Right s <- fmap decode $ T.readFile "resource/best-strategies-local/CClass.yml"
  return s

evaluate :: Genome -> IO Double
evaluate g = do
  ses <- performPrediction $ defaultStrategy & genome .~ g
  let res :: PredictionResult
      res = ses ^. predictionResult
      PredictionSuccess prMap = res
      val = prMap M.! MClassFlare M.! TrueSkillStatistic ^. scoreValue
  return val

proceed :: [Genome] -> IO [Genome]
proceed gs = do
  mutatedGs   <- mapM mutate gs
  crossoverGs <- replicateM 20 $ do
    [g1,g2] <- chooseN 2 gs
    crossover g1 g2
  let newPopulation = nub $ gs ++ mutatedGs ++ crossoverGs
  mapM_ putStrLn $ map pprGenome newPopulation

  egs <- forM newPopulation $ \g -> do
    e <- evaluate g
    return (e,g)
  let top10 = take 10 $ reverse $ sort egs
  print $ map fst top10
  appendFile "genetic-algorithm.txt" $ show $ map fst top10
  return $ map snd top10

main :: IO ()
main = withWorkDir $ do
  Right cclass <- fmap decode $ T.readFile "resource/best-strategies-local/CClass.yml"
  Right mclass <- fmap decode $ T.readFile "resource/best-strategies-local/MClass.yml"
  Right xclass <- fmap decode $ T.readFile "resource/best-strategies-local/XClass.yml"

  let population :: [Genome]
      population = map (^. genome) [cclass, mclass, xclass :: PredictionStrategyGS]
  vs <- P.parallel $ map evaluate population
  print vs

  loop population
  return ()

loop :: [Genome] -> IO ()
loop gs = do
  next <- proceed gs
  loop next
