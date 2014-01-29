{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.Regressor.General where

import Control.Lens
import qualified Data.Aeson.TH as Aeson

import SpaceWeather.DefaultFeatureSchemaPack
import SpaceWeather.FeaturePack
import SpaceWeather.Regressor.LibSVM
import SpaceWeather.Regressor.Linear
import SpaceWeather.Prediction
import SpaceWeather.TimeLine




-- | Choice for regression engine.

data GeneralRegressor = LibSVM LibSVMOption | Linear LinearOption deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions ''GeneralRegressor

instance Predictor GeneralRegressor where
  performPrediction ps = let optG = ps^.regressorUsed in case optG of
    LibSVM opt -> fmap (fmap (const optG)) $ performPrediction $ fmap (const opt) ps
    Linear opt -> fmap (fmap (const optG)) $ performPrediction $ fmap (const opt) ps

defaultPredictionStrategy :: PredictionStrategy GeneralRegressor
defaultPredictionStrategy = PredictionStrategy 
  { _regressorUsed = LibSVM defaultLibSVMOption
  , _featureSchemaPackUsed = defaultFeatureSchemaPack
  , _crossValidationStrategy = CVWeekly
  , _predictionTargetSchema = goes24max
  , _predictionTargetFile = "/user/nushio/forecast/forecast-goes-24.txt"
  , _predictionResultFile = ""}

goes24max :: FeatureSchema
goes24max = FeatureSchema
  { _colX = 2
  , _colY = 5
  , _weight = 1
  , _isLog = True}
