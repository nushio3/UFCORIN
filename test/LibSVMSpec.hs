{-# LANGUAGE OverloadedStrings #-}
module LibSVMSpec where

import qualified Data.Text as T
import qualified Data.Map as Map
import SpaceWeather.Feature
import SpaceWeather.Format
import SpaceWeather.LibSVM
import Test.Hspec
import Test.Hspec.QuickCheck
import Test.QuickCheck.Arbitrary

testObj :: LibSVMFeatures
testObj = LibSVMFeatures $ Map.fromList $ 
  [(0,([1,2],3))
  ,(4,([5,6],7))]

testEnco :: T.Text
testEnco = T.unlines
  ["3.0 1:1.0 2:2.0"
  ,"7.0 1:5.0 2:6.0"]


spec :: Spec
spec = do
  describe "LibSVM" $ do
    it "encodes properly." $
      encode testObj `shouldBe` testEnco
