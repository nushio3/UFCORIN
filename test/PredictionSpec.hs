module PredictionSpec where

import Control.Lens
import Control.Monad.IO.Class
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as Map
import SpaceWeather.DefaultFeatureSchemaPack
import SpaceWeather.Feature
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import System.IO.Unsafe
import Test.Hspec
import Test.Hspec.QuickCheck
import Test.QuickCheck.Arbitrary



spec::Spec
spec = do
  describe "Prediction Pack" $ do
    it "generates Yaml." $ do
      liftIO $ T.writeFile "resource/sample-strategy.yml" $ encode defaultPredictionStrategy
      liftIO $ T.writeFile "resource/big-strategy.yml" $ encode $ 
        defaultPredictionStrategy & featureSchemaPackUsed .~ defaultFeatureSchemaPackBig
      (decode $ encode defaultPredictionStrategy) `shouldBe` Right defaultPredictionStrategy
    it "accepts easy Yaml." $ do
      res <- liftIO $ decodeFile "resource/sample-strategy-beautiful.yml"
      res `shouldBe` Right defaultPredictionStrategy

