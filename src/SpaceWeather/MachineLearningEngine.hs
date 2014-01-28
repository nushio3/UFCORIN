{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.MachineLearningEngine where

import qualified Data.Aeson.TH as Aeson
import SpaceWeather.TimeLine

data LinearOption = LinearOption  deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions ''LinearOption


data LibSVMOption = LibSVMOption 
  { _libSVMType       :: Int
  , _libSVMKernelType :: Int
  , _libSVMCost       :: Double  
  , _libSVMEpsilon    :: Double
  , _libSVMGamma      :: Maybe Double
  , _libSVMNu         :: Double
  } deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 7} ''LibSVMOption

defaultLibSVMOption :: LibSVMOption
defaultLibSVMOption = LibSVMOption
  { _libSVMType       = 3
  , _libSVMKernelType = 2
  , _libSVMCost       = 1  
  , _libSVMEpsilon    = 0.001
  , _libSVMGamma      = Nothing
  , _libSVMNu         = 0.5
  }


-- | Choice for regression engine.

data Regressor = LibSVM LibSVMOption | Linear LinearOption deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''Regressor

data CrossValidationStrategy = CVWeekly | CVMonthly | CVYearly deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''CrossValidationStrategy

inTrainingSet :: CrossValidationStrategy -> TimeBin -> Bool
inTrainingSet CVWeekly n = even (n `div` (24*7))
inTrainingSet CVMonthly n = even (n `div` (24*31))
inTrainingSet CVYearly n = even (n `div` (24*366))
