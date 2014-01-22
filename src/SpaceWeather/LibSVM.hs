{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, OverloadedStrings, TemplateHaskell, TypeSynonymInstances #-}
module SpaceWeather.LibSVM where

import Control.Lens
import Control.Monad
import qualified Data.Map.Strict as Map
import Data.Monoid
import qualified Data.Text as T
import Test.QuickCheck.Arbitrary
import Test.QuickCheck.Gen

import SpaceWeather.TimeLine
import SpaceWeather.Format
import SpaceWeather.Feature


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


mkLibSVMLine :: (TimeBin,  ([Double],Double)) -> T.Text
mkLibSVMLine (_, (xis,xo)) = 
  ((showT xo <> " ") <> ) $ 
  T.unwords $ 
  zipWith (\i x -> showT i <> ":" <> showT x) [1..] $ xis
