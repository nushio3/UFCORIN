{-# LANGUAGE DeriveFunctor, FlexibleContexts, FlexibleInstances, FunctionalDependencies, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.Prediction where

import Control.Lens
import qualified Data.ByteString.Char8 as BS
import qualified Data.Aeson.TH as Aeson
import qualified Data.Map as Map
import qualified Data.Text as T
import qualified Data.Yaml as Yaml
import Data.Maybe
import System.Random

import SpaceWeather.DefaultFeatureSchemaPack
import SpaceWeather.Feature
import SpaceWeather.FeaturePack
import SpaceWeather.FlareClass
import SpaceWeather.Format
import SpaceWeather.SkillScore
import SpaceWeather.TimeLine

xor = (/=)

data CrossValidationStrategy = CVWeekly | CVMonthly | CVYearly  | CVNegate CrossValidationStrategy | CVShuffled Int CrossValidationStrategy
  deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions ''CrossValidationStrategy


inTrainingSet :: CrossValidationStrategy -> TimeBin -> Bool
inTrainingSet s n = inTrainingSetInner (repeat False) s n

inTrainingSetInner :: [Bool] -> CrossValidationStrategy -> TimeBin -> Bool
inTrainingSetInner shuffler CVWeekly  n = let i = (n `div` (24*7))   in even i `xor` (shuffler !! fromIntegral i)
inTrainingSetInner shuffler CVMonthly n = let i = (n `div` (24*31))  in even i `xor` (shuffler !! fromIntegral i)
inTrainingSetInner shuffler CVYearly  n = let i = (n `div` (24*366)) in even i `xor` (shuffler !! fromIntegral i)
inTrainingSetInner shuffler (CVNegate s) n = not $ inTrainingSetInner shuffler s n
inTrainingSetInner _ (CVShuffled seed s) n = inTrainingSetInner newShuffler s n
  where
    newShuffler = map fst $ drop 1 $ iterate f (False, mkStdGen seed)
    f :: (Bool, StdGen) -> (Bool,StdGen)
    f (_, g) = random g

-- | Using the functor instance you can change the regressor to other types.

data PredictionStrategy a = PredictionStrategy
  { _spaceWeatherLibVersion :: String
  , _regressorUsed :: a
  , _featureSchemaPackUsed  :: FeatureSchemaPack
  , _crossValidationStrategy :: CrossValidationStrategy
  , _predictionTargetSchema :: FeatureSchema
  , _predictionTargetFile :: FilePath
  , _predictionResultFile :: FilePath
  , _predictionRegressionFile :: FilePath
  , _predictionSessionFile :: FilePath
  } deriving (Eq, Ord, Show, Read, Functor)

makeClassy ''PredictionStrategy
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''PredictionStrategy

instance (Yaml.ToJSON a, Yaml.FromJSON a) => Format (PredictionStrategy a) where
  encode = T.pack . BS.unpack . Yaml.encode
  decode = Yaml.decodeEither . BS.pack . T.unpack


data PredictionResult
  = PredictionFailure { _predictionFailureMessage :: String }
  | PredictionSuccess
    { _predictionResultMap :: Map.Map FlareClass ScoreMap
    }deriving (Eq, Ord, Show, Read)
makeClassy ''PredictionResult
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = filter (/= '_')} ''PredictionResult
instance Format PredictionResult where
  encode = T.pack . BS.unpack . Yaml.encode
  decode = Yaml.decodeEither . BS.pack . T.unpack



data PredictionSession a = PredictionSession
  { _predictionStrategyUsed  :: PredictionStrategy a
  , _predictionSessionResult :: PredictionResult
  } deriving (Eq, Ord, Show, Read, Functor)
makeClassy ''PredictionSession
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''PredictionSession
instance HasPredictionResult (PredictionSession a) where
  predictionResult = predictionSessionResult


instance (Yaml.ToJSON a, Yaml.FromJSON a) => Format (PredictionSession a) where
  encode = T.pack . BS.unpack . Yaml.encode
  decode = Yaml.decodeEither . BS.pack . T.unpack

data LearningInput = LearningInput
  { _learningTrainSet :: FeatureIOPair
  , _learningTestSet :: FeatureIOPair
   }
makeClassy ''LearningInput

data LearningOutput = LearningOutput
  { _learningPredictionResults :: [Double]
  }
makeClassy ''LearningOutput



class Predictor a where
  performPrediction :: PredictionStrategy a -> IO (PredictionSession a)
  preprocessLearning :: PredictionStrategy a -> IO (Either String LearningInput)
  performLearning :: a -> LearningInput -> LearningOutput
  postprocessLearning :: PredictionStrategy a -> LearningOutput -> IO (PredictionSession a)



prToDouble :: PredictionResult -> Double
prToDouble = prToDoubleWith f where
  f _ TrueSkillStatistic = 1
  f _ _                  = 0


prToDoubleWith :: (FlareClass -> ScoreMode -> Double) -> PredictionResult -> Double
prToDoubleWith _  (PredictionFailure _) = 0
prToDoubleWith wf (PredictionSuccess m) = Prelude.sum $ do -- List Monad
  (fc, prmap) <- Map.toList m
  (sm, sr) <- Map.toList prmap
  return $ wf fc sm * _scoreValue sr
