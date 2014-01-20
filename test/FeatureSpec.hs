module FeatureSpec where

import SpaceWeather.Feature
import SpaceWeather.Format
import Test.Hspec


spec :: Spec
spec = do
  describe "Feature" $ do
    prop "is encoded/decoded back properly." $ \feature -> 
      decode $ encode (feature :: Feature) == Right feature
