{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.Prediction where

import Control.Lens 
import qualified Data.Aeson.TH as Aeson
import qualified Data.Yaml as Yaml

import SpaceWeather.FeaturePack
import SpaceWeather.MachineLearningEngine

data PredictionStrategy = PredictionStrategy 
  { _regressorUsed :: Regressor
  , _featureFilesUsed  :: FeatureSchemaPack
  , _predictionTargetFile :: FilePath
  , _predictionResultFile :: FilePath
  }

makeClassy ''PredictionStrategy
Aeson.deriveJSON Aeson.defaultOptions ''PredictionStrategy

defaultPredictionStrategy :: PredictionStrategy
defaultPredictionStrategy = PredictionStrategy
  { _regressorUsed = LibSVM defaultLibSVMOption
  , _featureFilesUsed = undefined
  , _predictionTargetFile = ""
  , _predictionResultFile = ""}

data PredictionSession = PredictionSession
  { _predictionStrategyUsed :: PredictionStrategy
  , _heidkeSkillScore       :: Double
  , _trueSkillScore         :: Double
  }
