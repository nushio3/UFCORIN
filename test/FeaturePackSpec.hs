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
  { _colX = 3
  , _colY = 5
  , _weight = 1
  , _isLog = True
  }


testFSP :: FeatureSchemaPack
testFSP = FeatureSchemaPack
  { _fspSchemaDefinitions = Map.fromList
      [("35L", testFS1)]
  , _fspFilenamePairs = 
      [("35L",   "/user/nushio/wavelet-features/haarC-2-S-0000-0000.txt")
      ,("35L",   "/user/nushio/wavelet-features/bsplC-2-S-0000-0000.txt")
      ,("35L",   "/user/nushio/wavelet-features/bsplC-2-S-0001-0001.txt")]}

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
      liftIO $ T.writeFile "test.fsp" $ encode testFSP
      testFSP `shouldBe` testFSP2
