{-# LANGUAGE DeriveFunctor, FlexibleContexts, FlexibleInstances, FunctionalDependencies, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.Prediction where

import Control.Lens 
import qualified Data.ByteString.Char8 as BS
import qualified Data.Aeson.TH as Aeson
import qualified Data.Map as Map
import qualified Data.Text as T
import qualified Data.Yaml as Yaml


import SpaceWeather.DefaultFeatureSchemaPack
import SpaceWeather.FeaturePack
import SpaceWeather.FlareClass
import SpaceWeather.Format
import SpaceWeather.SkillScore       
import SpaceWeather.TimeLine

data CrossValidationStrategy = CVWeekly | CVMonthly | CVYearly  | CVNegate CrossValidationStrategy
  deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions ''CrossValidationStrategy

inTrainingSet :: CrossValidationStrategy -> TimeBin -> Bool
inTrainingSet CVWeekly n = even (n `div` (24*7))
inTrainingSet CVMonthly n = even (n `div` (24*31))
inTrainingSet CVYearly n = even (n `div` (24*366))


-- | Using the functor instance you can change the regressor to other types.

data PredictionStrategy a = PredictionStrategy 
  { _spaceWeatherLibVersion :: String
  , _regressorUsed :: a
  , _featureSchemaPackUsed  :: FeatureSchemaPack
  , _crossValidationStrategy :: CrossValidationStrategy
  , _predictionTargetSchema :: FeatureSchema
  , _predictionTargetFile :: FilePath
  , _predictionResultFile :: FilePath
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


class Predictor a where
  performPrediction :: PredictionStrategy a -> IO (PredictionSession a)

