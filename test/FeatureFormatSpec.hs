{-# LANGUAGE OverloadedStrings #-}
module FeatureFormatSpec where

import qualified Data.Text as T
import qualified Data.Map as Map
import SpaceWeather.Feature
import SpaceWeather.Format
import Test.Hspec
import Test.Hspec.QuickCheck


testFeature :: Feature
testFeature = Map.fromList [(0,1.2),(3,4.5)]

testEncode1 :: T.Text
testEncode1 = "0 1.2\n3 4.5\n"

testEncode2 :: T.Text
testEncode2 = "#comment\n0 1.2\n  #  another comment \n  3 \t  4.5 \n"

spec :: Spec
spec = do
  describe "Feature" $ do
    it "is encoded as expected." $
      encode testFeature `shouldBe` testEncode1
    it "is robust against comment and spaces." $
      decode testEncode2 `shouldBe` testFeature
    prop "is encoded/decoded back properly." $ \feature -> 
      (decode $ encode (feature :: Feature)) == Right feature
