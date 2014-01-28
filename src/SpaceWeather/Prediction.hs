{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.Prediction where

import Control.Lens 
import qualified Data.ByteString.Char8 as BS
import qualified Data.Aeson.TH as Aeson
import qualified Data.Text as T
import qualified Data.Yaml as Yaml


import SpaceWeather.DefaultFeatureSchemaPack
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.MachineLearningEngine

data PredictionStrategy = PredictionStrategy 
  { _regressorUsed :: Regressor
  , _featureSchemaPackUsed  :: FeatureSchemaPack
  , _crossValidationStrategy :: CrossValidationStrategy
  , _predictionTargetSchema :: FeatureSchema
  , _predictionResultFile :: FilePath
  }

makeClassy ''PredictionStrategy
Aeson.deriveJSON Aeson.defaultOptions ''PredictionStrategy

instance Format PredictionStrategy where
  encode = T.pack . BS.unpack . Yaml.encode
  decode = Yaml.decodeEither . BS.pack . T.unpack


defaultPredictionStrategy :: PredictionStrategy
defaultPredictionStrategy = PredictionStrategy
  { _regressorUsed = LibSVM defaultLibSVMOption
  , _featureSchemaPackUsed = defaultFeatureSchemaPack
  , _crossValidationStrategy = CVWeekly
  , _predictionTargetSchema = goes24max
  , _predictionResultFile = ""}

goes24max :: FeatureSchema
goes24max = FeatureSchema
  { _colX = 2
  , _colY = 5
  , _weight = 1
  , _isLog = True}

data PredictionSession = PredictionSession
  { _predictionStrategyUsed :: PredictionStrategy
  , _heidkeSkillScore       :: Double
  , _trueSkillScore         :: Double
  }
