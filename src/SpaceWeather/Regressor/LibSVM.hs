{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, OverloadedStrings, TemplateHaskell, TypeSynonymInstances #-}
module SpaceWeather.Regressor.LibSVM where

import Control.Lens
import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.Trans.Either
import qualified Data.Aeson.TH as Aeson
import qualified Data.ByteString.Char8 as BS
import qualified Data.Yaml as Yaml
import qualified Data.Map.Strict as Map
import Data.Monoid
import qualified Data.Text as T
import qualified Data.Text.IO as T
import System.IO
import System.Process
import Test.QuickCheck.Arbitrary
import Test.QuickCheck.Gen
import Text.Printf

import SpaceWeather.CmdArgs
import SpaceWeather.TimeLine
import SpaceWeather.FlareClass
import SpaceWeather.Format
import SpaceWeather.Feature
import SpaceWeather.FeaturePack
import SpaceWeather.SkillScore
import SpaceWeather.Prediction

data LibSVMOption = LibSVMOption 
  { _libSVMType       :: Int
  , _libSVMKernelType :: Int
  , _libSVMCost       :: Double  
  , _libSVMEpsilon    :: Double
  , _libSVMGamma      :: Maybe Double
  , _libSVMNu         :: Double
  } deriving (Eq, Ord, Show, Read)
makeClassy ''LibSVMOption
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 7} ''LibSVMOption

defaultLibSVMOption :: LibSVMOption
defaultLibSVMOption = LibSVMOption
  { _libSVMType       = 3
  , _libSVMKernelType = 2
  , _libSVMCost       = 1  
  , _libSVMEpsilon    = 0.001
  , _libSVMGamma      = Nothing
  , _libSVMNu         = 0.5
  }

libSVMOptionCmdLine :: LibSVMOption -> String
libSVMOptionCmdLine opt = 
  printf "-s %d -t %d -c %f -e %f %s -n %f"
    (opt^.libSVMType) (opt^.libSVMKernelType) (opt^.libSVMCost) (opt^.libSVMEpsilon) gammaStr (opt^.libSVMNu)
  where
    gammaStr = case opt ^. libSVMGamma of
      Nothing -> ""
      Just x  -> "-g " ++ show x
newtype LibSVMFeatures = LibSVMFeatures {
      _libSVMIOPair :: FeatureIOPair
   }
makeClassy  ''LibSVMFeatures
makeWrapped ''LibSVMFeatures

instance Arbitrary LibSVMFeatures where
  arbitrary = 
    sized $ \sz -> do
      nrow <- fmap abs arbitrary
      let ncol :: Int
          ncol = sz `div` nrow
      tmp <- replicateM nrow $ do
        t <- arbitrary
        xo <- arbitrary
        xis <- vectorOf ncol arbitrary
        return (t,(xis, xo))
      return $ LibSVMFeatures $ Map.fromList tmp


instance Format LibSVMFeatures where
  decode _ = Left "LibSVMFeatures is read only."
  encode lf = T.unlines $ map mkLibSVMLine $ Map.toAscList $ lf ^. libSVMIOPair
    where
      mkLibSVMLine :: (TimeBin,  ([Double],Double)) -> T.Text
      mkLibSVMLine (_, (xis,xo)) = 
        ((showT xo <> " ") <> ) $ 
        T.unwords $ 
        zipWith (\i x -> showT i <> ":" <> showT x) [1..] $ xis


instance Predictor LibSVMOption where
  performPrediction strategy = do
    e <- runEitherT $ libSVMPerformPrediction strategy
    let res = case e of
          Left msg -> PredictionFailure msg
          Right x  -> x
    return $ PredictionSession {
      _predictionStrategyUsed = strategy
    , _predictionSessionResult = res
    }


libSVMPerformPrediction :: PredictionStrategy LibSVMOption -> EitherT String IO PredictionResult
libSVMPerformPrediction strategy = do
  let fsp :: FeatureSchemaPack
      fsp = strategy ^. featureSchemaPackUsed

  featurePack0 <- EitherT $ loadFeatureSchemaPack fsp

  let tgtSchema = strategy ^. predictionTargetSchema
      tgtFn     = strategy ^. predictionTargetFile

  tgtFeature <- loadFeatureWithSchemaT tgtSchema tgtFn


  let fioPair0 :: FeatureIOPair
      fioPair0 = catFeaturePair fs0 tgtFeature
      
      fs0 :: [Feature]
      fs0 = view unwrapped featurePack0

      pred :: TimeBin -> Bool
      pred = inTrainingSet $ strategy ^. crossValidationStrategy

      (fioTrainSet, fioTestSet) = Map.partitionWithKey (\k _ -> pred k) fioPair0

      svmTrainSet = LibSVMFeatures fioTrainSet
      svmTestSet = LibSVMFeatures fioTestSet

      fnTrainSet = workDir ++ "/train.txt" 
      fnModel =  workDir ++ "/train.txt.model" 
      fnTestSet = workDir ++ "/test.txt" 
      fnPrediction = workDir ++ "/test.txt.prediction" 
 
      svmTrainCmd = printf "./svm-train %s %s %s"
        (libSVMOptionCmdLine $ strategy ^. regressorUsed)
        fnTrainSet fnModel

      svmPredictCmd = printf "./svm-predict %s %s %s"
        fnTestSet
        fnModel
        fnPrediction

  liftIO $ do
    encodeFile fnTrainSet svmTrainSet
    encodeFile fnTestSet svmTestSet

    system "hadoop fs -get /user/nushio/libsvm-3.17/svm-train ."
    system "hadoop fs -get /user/nushio/libsvm-3.17/svm-predict ."
    system $ svmTrainCmd
    system $ svmPredictCmd

  predictionStr <- liftIO $ readFile fnPrediction  

  let predictions :: [Double]
      predictions = map read $ lines predictionStr

      observations :: [Double]
      observations = map (snd . snd) $ Map.toAscList fioTestSet

      poTbl = zip predictions observations

      resultMap0 = Map.fromList $ 
        [ let logXRF = log (xRayFlux flare1) / log 10 in (flare1 , makeScoreMap poTbl logXRF)
        | flare1 <- defaultFlareClasses]
  
      ret = PredictionSuccess resultMap0


  liftIO $ BS.hPutStrLn stderr $ Yaml.encode ret
  return ret
