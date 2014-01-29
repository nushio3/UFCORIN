{-# LANGUAGE FlexibleInstances, TemplateHaskell #-}
module SpaceWeather.FlareClass where

import Control.Lens
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.TH as Aeson
import qualified Data.Map as Map

data FlareClass = CClassFlare | MClassFlare | XClassFlare
  deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions ''FlareClass

instance Aeson.ToJSON a => Aeson.ToJSON (Map.Map FlareClass a) where
  -- define the instance via Map String a
  toJSON = Aeson.toJSON . Map.fromList . (map (_1 %~ show)) . Map.toList

instance Aeson.FromJSON a => Aeson.FromJSON (Map.Map FlareClass a) where
  parseJSON = fmap go . Aeson.parseJSON
    where
      go :: Map.Map String a -> Map.Map FlareClass a  
      go = Map.fromList . (map (_1 %~ read)) . Map.toList

xRayFlux :: FlareClass -> Double
xRayFlux CClassFlare = 1e-6
xRayFlux MClassFlare = 1e-5
xRayFlux XClassFlare = 1e-4

defaultFlareClasses :: [FlareClass]
defaultFlareClasses = [CClassFlare, MClassFlare, XClassFlare]

