{-# LANGUAGE FlexibleContexts, FlexibleInstances, FunctionalDependencies, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.Prediction where

import Control.Lens 
import qualified Data.ByteString.Char8 as BS
import qualified Data.Aeson.TH as Aeson
import qualified Data.Text as T
import qualified Data.Yaml as Yaml


import SpaceWeather.DefaultFeatureSchemaPack
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.TimeLine

data CrossValidationStrategy = CVWeekly | CVMonthly | CVYearly deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''CrossValidationStrategy

inTrainingSet :: CrossValidationStrategy -> TimeBin -> Bool
inTrainingSet CVWeekly n = even (n `div` (24*7))
inTrainingSet CVMonthly n = even (n `div` (24*31))
inTrainingSet CVYearly n = even (n `div` (24*366))

data PredictionStrategy a = PredictionStrategy 
  { _regressorUsed :: a
  , _featureSchemaPackUsed  :: FeatureSchemaPack
  , _crossValidationStrategy :: CrossValidationStrategy
  , _predictionTargetSchema :: FeatureSchema
  , _predictionResultFile :: FilePath
  } deriving (Eq, Ord, Show, Read)

makeClassy ''PredictionStrategy
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''PredictionStrategy

instance (Yaml.ToJSON a, Yaml.FromJSON a) => Format (PredictionStrategy a) where
  encode = T.pack . BS.unpack . Yaml.encode
  decode = Yaml.decodeEither . BS.pack . T.unpack



data PredictionSession a = PredictionSession
  { _predictionStrategyUsed :: PredictionStrategy a
  , _heidkeSkillScore       :: Double
  , _trueSkillScore         :: Double
  } deriving (Eq, Ord, Show, Read)
makeClassy ''PredictionSession
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''PredictionSession

class Predictor a where
  performPrediction :: PredictionStrategy a -> IO (PredictionSession a)

