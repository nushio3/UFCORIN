{-# LANGUAGE OverloadedStrings #-}
module FeatureSpec where

import qualified Data.Text as T
import qualified Data.Map as Map
import SpaceWeather.Feature
import SpaceWeather.Format
import Test.Hspec
import Test.Hspec.QuickCheck
import Test.QuickCheck.Arbitrary

newtype SmallPositive = SmallPositive Int deriving (Eq,Show)
instance Arbitrary SmallPositive where
  arbitrary = do
    n <- arbitrary
    return $ SmallPositive $ 1 + mod n 10
  shrink (SmallPositive n) = map SmallPositive $ filter (>0) $ shrink n


testFeature :: Feature
testFeature = Map.fromList [(0,1.2),(3,4.5)]

testFeature2 :: Feature
testFeature2 = Map.fromList $ zip [1..11] [3..33]

testFeature3 :: Feature
testFeature3 = Map.fromList $ zip [2..22] [44..444]

testFeatures23 :: Features
testFeatures23 = Map.fromList $ [(t, [fromIntegral t+2, fromIntegral t+42])|t<-[2..11]]

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
      decode testEncode2 `shouldBe` Right testFeature
    prop "is encoded/decoded back properly." $ \feature -> 
      (decode $ encode (feature :: Feature)) == Right feature

  describe "Features" $ do
    it "cats as expected." $
      catFeatures [testFeature2, testFeature3] `shouldBe` testFeatures23
    prop "cat is idempotent." $ \(SmallPositive n, feature)-> 
      catFeatures (replicate n feature) == Map.map (replicate n) feature 
    prop "every row of cat of N features contain N elements." $ \fs -> 
      all ((== length fs) . length)  $ Map.elems $ catFeatures fs
