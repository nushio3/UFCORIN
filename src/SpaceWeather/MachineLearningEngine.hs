{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.MachineLearningEngine where

import qualified Data.Aeson.TH as Aeson

data LinearOption = LinearOption 
Aeson.deriveJSON Aeson.defaultOptions ''LinearOption


data LibSVMOption = LibSVMOption 
  { _libSVMType       :: Int
  , _libSVMKernelType :: Int
  , _libSVMCost       :: Double  
  , _libSVMEpsilon    :: Double
  , _libSVMGamma      :: Maybe Double
  , _libSVMNu         :: Double
  }
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


data Regressor = LibSVM LibSVMOption | Linear LinearOption
Aeson.deriveJSON Aeson.defaultOptions ''Regressor

