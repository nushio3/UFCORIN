module FeaturePackSpec where

import Control.Lens
import Control.Monad.IO.Class
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as Map
import SpaceWeather.Feature
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import System.IO.Unsafe
import Test.Hspec
import Test.Hspec.QuickCheck
import Test.QuickCheck.Arbitrary

testFS1 :: FeatureSchema
testFS1 = FeatureSchema 
  { _colT = 3
  , _colX = 5
  , _scaling = 1
  , _isLog = True
  }


testFSP :: FeatureSchemaPack
testFSP = FeatureSchemaPack
  { _fspSchemaDefinitions = Map.fromList
      [("f35L", testFS1)]
  , _fspFilenamePairs = 
      [("f35L",   "/user/nushio/wavelet-features/haarC-2-S-0000-0000.txt")
      ,("f35L",   "/user/nushio/wavelet-features/bsplC-301-S-0000-0000.txt")
      ,("f35L",   "/user/shibayama/sdo/hmi/hmi_totalflux.txt")]}

testFSP2 :: FeatureSchemaPack
testFSP2 = ret
  where
    Right ret = 
      unsafePerformIO $ 
      decodeFile "resource/sample.fsp"




spec::Spec
spec = do
  describe "FeaturePack" $ do
    it "accepts easy Yaml." $ do
      liftIO $ T.writeFile "resource/sample-featurepack.yml" $ encode testFSP
      testFSP `shouldBe` testFSP2
