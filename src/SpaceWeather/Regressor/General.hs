{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.Regressor.General where

import Control.Lens
import Control.Monad
import Data.Version (showVersion)
import Data.Function (on)
import Data.List
import qualified Data.Aeson.TH as Aeson

import Paths_spaceweather_wavelet
import SpaceWeather.DefaultFeatureSchemaPack
import SpaceWeather.FeaturePack
import SpaceWeather.Regressor.LibSVM
import SpaceWeather.Regressor.Linear
import SpaceWeather.Prediction
import SpaceWeather.TimeLine




-- | Choice for regression engine.

data GeneralRegressor = LibSVMRegressor LibSVMOption | LinearRegressor LinearOption deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions ''GeneralRegressor

type PredictionStrategyGS = PredictionStrategy [GeneralRegressor]
type PredictionSessionGS = PredictionSession [GeneralRegressor]

instance Predictor GeneralRegressor where
  performPrediction ps = let optG = ps^.regressorUsed in case optG of
    LibSVMRegressor opt -> fmap (fmap LibSVMRegressor) $ performPrediction $ fmap (const opt) ps
    LinearRegressor opt -> fmap (fmap LinearRegressor) $ performPrediction $ fmap (const opt) ps

instance Predictor [GeneralRegressor] where
  performPrediction ps = do
    let 
        rs :: [GeneralRegressor]
        rs = ps^.regressorUsed 
        singleStrts :: [PredictionStrategy GeneralRegressor]
        singleStrts = [fmap (const r) ps | r <- rs]

        cmp = compare `on` (prToDouble . _predictionSessionResult)
    sessions <- mapM performPrediction $ singleStrts
    return $ fmap (:[]) $ last $ sortBy cmp sessions
    

defaultPredictionStrategy :: PredictionStrategy [GeneralRegressor]
defaultPredictionStrategy = PredictionStrategy 
  { _spaceWeatherLibVersion = "version " ++ showVersion version
  , _regressorUsed = 
   [ LibSVMRegressor $ defaultLibSVMOption{_libSVMCost = c} 
   | c <- [1]]
  , _featureSchemaPackUsed = defaultFeatureSchemaPack
  , _crossValidationStrategy = CVWeekly
  , _predictionTargetSchema = goes24max
  , _predictionTargetFile = "/user/nushio/forecast/forecast-goes-24.txt"
  , _predictionResultFile = ""
  , _predictionRegressionFile = ""
  , _predictionSessionFile = ""}

biggerPredictionStrategy :: PredictionStrategy [GeneralRegressor]
biggerPredictionStrategy = PredictionStrategy 
  { _spaceWeatherLibVersion = "version " ++ showVersion version
  , _regressorUsed = 
   [ LibSVMRegressor $
       defaultLibSVMOption{
         _libSVMCost = 10**(c/10) ,
         _libSVMGamma = 10**(g/10) } 
   | c <- [0..30], g <- [-70.. -30]]
  , _featureSchemaPackUsed = defaultFeatureSchemaPack
  , _crossValidationStrategy = CVWeekly
  , _predictionTargetSchema = goes24max
  , _predictionTargetFile = "/user/nushio/forecast/forecast-goes-24.txt"
  , _predictionResultFile = ""
  , _predictionRegressionFile = ""
  , _predictionSessionFile = ""}


goes24max :: FeatureSchema
goes24max = FeatureSchema
  { _colT = 2
  , _colX = 5
  , _scaling = 1
  , _isLog = True}
