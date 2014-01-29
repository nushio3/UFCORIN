{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, OverloadedStrings, TemplateHaskell, TypeSynonymInstances #-}
module SpaceWeather.Regressor.LibSVM where

import Control.Lens
import Control.Monad
import qualified Data.Aeson.TH as Aeson
import qualified Data.Map.Strict as Map
import Data.Monoid
import qualified Data.Text as T
import Test.QuickCheck.Arbitrary
import Test.QuickCheck.Gen

import SpaceWeather.TimeLine
import SpaceWeather.Format
import SpaceWeather.Feature

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

newtype LibSVMFeatures = LibSVMFeatures {
      _libSVMIOPair :: FeatureIOPair
   }
makeClassy  ''LibSVMFeatures
makeWrapped ''LibSVMFeatures

instance Arbitrary LibSVMFeatures where
  arbitrary = 
    sized $ \sz -> do
      nrow <- fmap abs arbitrary
      let ncol :: Int
          ncol = sz `div` nrow
      tmp <- replicateM nrow $ do
        t <- arbitrary
        xo <- arbitrary
        xis <- vectorOf ncol arbitrary
        return (t,(xis, xo))
      return $ LibSVMFeatures $ Map.fromList tmp


instance Format LibSVMFeatures where
  decode _ = Left "LibSVMFeatures is read only."
  encode lf = T.unlines $ map mkLibSVMLine $ Map.toList $ lf ^. libSVMIOPair
    where
      mkLibSVMLine :: (TimeBin,  ([Double],Double)) -> T.Text
      mkLibSVMLine (_, (xis,xo)) = 
        ((showT xo <> " ") <> ) $ 
        T.unwords $ 
        zipWith (\i x -> showT i <> ":" <> showT x) [1..] $ xis


