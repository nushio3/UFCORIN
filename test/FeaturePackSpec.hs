module FeaturePackSpec where

import Control.Monad.IO.Class
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as Map
import SpaceWeather.Feature
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import Test.Hspec
import Test.Hspec.QuickCheck
import Test.QuickCheck.Arbitrary

testFS1 :: FeatureSchema
testFS1 = FeatureSchema 
  { _colX = 3
  , _colY = 5
  , _weight = 1
  , _isLog = True
  , _schemaFilename = "/user/nushio/wavelet-features/haarC-2-S-0000-0000.txt"
  }

testFP :: FeatureSchemaPack
testFP = FeatureSchemaPack [testFS1]

spec::Spec
spec = do
  describe "A" $ do
    it "B" $ do
      liftIO $ T.putStr $ encode testFP
      (1+1) `shouldBe` 2
