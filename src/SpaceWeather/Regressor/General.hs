{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.Regressor.General where

import Paths_spaceweather_wavelet
import Data.Version (showVersion)
import Control.Lens
import qualified Data.Aeson.TH as Aeson

import SpaceWeather.DefaultFeatureSchemaPack
import SpaceWeather.FeaturePack
import SpaceWeather.Regressor.LibSVM
import SpaceWeather.Regressor.Linear
import SpaceWeather.Prediction
import SpaceWeather.TimeLine




-- | Choice for regression engine.

data GeneralRegressor = LibSVMRegressor LibSVMOption | LinearRegressor LinearOption deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions ''GeneralRegressor

type PredictionStrategyG = PredictionStrategy GeneralRegressor
type PredictionSessionG = PredictionSession GeneralRegressor

instance Predictor GeneralRegressor where
  performPrediction ps = let optG = ps^.regressorUsed in case optG of
    LibSVMRegressor opt -> fmap (fmap LibSVMRegressor) $ performPrediction $ fmap (const opt) ps
    LinearRegressor opt -> fmap (fmap LinearRegressor) $ performPrediction $ fmap (const opt) ps

defaultPredictionStrategy :: PredictionStrategy GeneralRegressor
defaultPredictionStrategy = PredictionStrategy 
  { _spaceWeatherLibVersion = "version " ++ showVersion version
  , _regressorUsed = LibSVMRegressor defaultLibSVMOption
  , _featureSchemaPackUsed = defaultFeatureSchemaPack
  , _crossValidationStrategy = CVWeekly
  , _predictionTargetSchema = goes24max
  , _predictionTargetFile = "/user/nushio/forecast/forecast-goes-24.txt"
  , _predictionResultFile = ""
  , _predictionSessionFile = ""}

goes24max :: FeatureSchema
goes24max = FeatureSchema
  { _colX = 2
  , _colY = 5
  , _scaling = 1
  , _isLog = True}
