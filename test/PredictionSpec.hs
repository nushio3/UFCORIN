module PredictionSpec where

import Control.Lens
import Control.Monad.IO.Class
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as Map
import SpaceWeather.Feature
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.Prediction
import System.IO.Unsafe
import Test.Hspec
import Test.Hspec.QuickCheck
import Test.QuickCheck.Arbitrary



spec::Spec
spec = do
  describe "Prediction Pack" $ do
    it "generates Yaml." $ do
      liftIO $ T.writeFile "resource/sample-predictor.yml" $ encode defaultPredictionStrategy
      (decode $ encode defaultPredictionStrategy) `shouldBe` Right defaultPredictionStrategy

