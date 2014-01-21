{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.PredictionStrategy where

import Control.Lens 
import Data.Aeson.TH
import Data.Yaml

import SpaceWeather.FeaturePack
import SpaceWeather.MachineLearningEngine

data PredictionStrategy = PredictionStrategy 
  { _regressorUsed :: Regressor
  , _featuresUsed  :: FeaturePack

  }

makeClassy ''PredictionStrategy
