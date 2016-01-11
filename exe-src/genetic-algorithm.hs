{-# LANGUAGE TupleSections #-}
import Control.Lens
import Control.Monad
import qualified Control.Concurrent.ParallelIO.Global as P

import Data.Function (on)
import Data.List
import qualified Data.Map as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Vector.Unboxed as V
import Data.Traversable (traverse)
import qualified Statistics.Sample as Stat
import System.Random
import System.System
import System.IO.Unsafe

import SpaceWeather.CmdArgs
import SpaceWeather.FlareClass
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import SpaceWeather.FeaturePack
import SpaceWeather.SkillScore

import qualified GoodSeed


stubMode :: Bool
stubMode = False

genLimit :: Int
genLimit = 20

mutationRateDefault :: Double
mutationRateDefault = 0.01

mutationRateThreshold :: Double
mutationRateThreshold = 0.001

mutationRateOfChange :: Double
mutationRateOfChange = 2.0

goodSeeds :: Int -> IO [Int]
goodSeeds n
  | stubMode  = return $ replicate n 0
  | otherwise = GoodSeed.goodSeeds n

statisticSize :: Int
statisticSize = 1

populationSize :: Int
populationSize = 2

infix 6 :±
data Statistics = Double :± Double deriving (Eq, Show, Ord)

meanDevi :: [Double] -> Statistics
meanDevi xs = let vx = V.fromList xs
                  (m,v) = Stat.meanVarianceUnb vx
              in m :± sqrt v

type Genome = M.Map (String, FilePath) Bool
data Individual = Individual {individualGenome :: Genome, individualStatistics :: Statistics}
                  deriving (Eq, Show)
type Population = [Individual]

pprGenome :: Genome -> String
pprGenome g = concat $ map (show . fromEnum) $ map snd $ M.toList g


genome :: Lens' PredictionStrategyGS Genome
genome = featureSchemaPackUsed . fspFilenamePairs . l
  where
    l :: Lens' [(String, FilePath)] Genome
    l = lens g s
    g :: [(String, FilePath)] -> Genome
    g xs = M.union (M.fromList $ zip xs $ repeat True) defaultGenome

    s :: [(String, FilePath)] -> Genome -> [(String, FilePath)]
    s _ xs = map fst $ filter snd $ M.toList xs

mutate :: (Int, Double) -> Genome -> IO Genome
mutate (genCt, bfValueGap) g = traverse flipper g
  where
    flipper :: Bool -> IO Bool
    flipper x = do
      r <- randomRIO (0,1 :: Double)
      return $ if r < mutateProb then (not x) else x
    mutateProb :: Double
    mutateProb = min 0.5 (mRate * (1 + fromIntegral genCt) / 4)
    mRate :: Double
    mRate
      | bfValueGap >= mutationRateThreshold     = mutationRateDefault
      | otherwise                               = mutationRateDefault * mutationRateOfChange

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

{-# NOINLINE defaultStrategy #-}
defaultStrategy :: PredictionStrategyGS
defaultStrategy = unsafePerformIO $ do
  Right s <- fmap decode $ T.readFile "resource/best-strategies-local/CClass.yml"
  return s
{-
evaluateWithSeed :: Int -> Genome -> IO Double
evaluateWithSeed s g = do
  r <- randomRIO (-0.1, 0.1)
  let cnt = length $ filter id $ map snd $ M.toList g
      cntMax = length $ M.toList $ defaultGenome
  return $ fromIntegral cnt / fromIntegral cntMax + r
-}


evaluateWithSeed :: Int -> Genome -> IO Double
evaluateWithSeed seed g
  | stubMode = do
    r <- randomRIO (-0.1, 0.1)
    let cnt = length $ filter id $ map snd $ M.toList g
        cntMax = length $ M.toList $ defaultGenome
    return $ fromIntegral cnt / fromIntegral cntMax + r
  | otherwise = do
    r <- replicateM 32 $ randomRIO ('a','z')
    let wd = workDir ++ r
        rsc = defaultPredictorResource { _workingDirectory = wd }
    withWorkDirOf wd $  do
      let pStrategy :: PredictionStrategyGS
          pStrategy = defaultStrategy
                      & genome .~ g
                      & crossValidationStrategy .~ (CVShuffled (seed) CVWeekly)
      ses <- performPrediction rsc $ pStrategy

      let res :: PredictionResult
          res = ses ^. predictionResult
          val = case res of
            PredictionSuccess prMap ->
              prMap M.! MClassFlare M.! TrueSkillStatistic ^. scoreValue
            _ -> 0
      return val


evaluate :: [Int] -> Genome -> IO Individual
evaluate seeds g = do
  vals <- P.parallel [evaluateWithSeed s g | s <- seeds]
  return $ Individual g (meanDevi vals)

proceed :: (Int, (Double, Double)) -> [Genome] -> Population -> IO (Population, [Genome], (Double, Double))
proceed (genCt, (p2BfValue, p1BfValue)) tabooList pop = do
  appendFile "genetic-algorithm.txt" $ (++"\n") $show $ map individualStatistics pop

  let gs :: [Genome]
      gs = map individualGenome pop
      bfValueGap :: Double
      bfValueGap = p1BfValue - p2BfValue
  mutatedGs   <- mapM (mutate (genCt, bfValueGap)) gs
  crossoverGs <- replicateM populationSize $ do
    [g1,g2] <- chooseN 2 gs
    crossover g1 g2
  let newGenomes = filter (not. isTaboo) $ nub $ mutatedGs ++ crossoverGs  --nub:delete same genomes
      isTaboo :: Genome -> Bool --Taboo List
      isTaboo _ = False
  seeds <- goodSeeds statisticSize

  newIndividuals <- P.parallel [evaluate seeds g | g <- newGenomes]

  let tops = take populationSize $ reverse $ sortBy (compare `on` individualStatistics) $ pop ++ newIndividuals
      (newBfValue :± xs) = individualStatistics $ head $ take 1 tops
  print $ map individualStatistics tops
  return $ (tops, tabooList, (p1BfValue, newBfValue))
  

main :: IO ()
main = do
  Right cclass <- fmap decode $ T.readFile "resource/best-strategies-local/CClass.yml"
  Right mclass <- fmap decode $ T.readFile "resource/best-strategies-local/MClass.yml"
  Right xclass <- fmap decode $ T.readFile "resource/best-strategies-local/XClass.yml"

  let initialGenomes :: [Genome]
      initialGenomes = map (^. genome) [cclass, mclass, xclass :: PredictionStrategyGS]

  seeds <- goodSeeds statisticSize

  initialPopulation <- P.parallel $ map (evaluate seeds) initialGenomes
  print initialPopulation

  loop (1, (0.0, 0.0)) [] initialPopulation
  P.stopGlobalPool
  return ()
  
loop :: (Int, (Double, Double)) -> [Genome] -> Population -> IO ()
loop (genCt, (bfValue, newBfValue)) tabooList gs = do
  (next, newTabooList, (bfValue, newBfValue)) <- proceed (genCt, (bfValue, newBfValue)) tabooList  gs
  when (genCt < genLimit) $ loop (genCt +1, (bfValue, newBfValue)) newTabooList next
