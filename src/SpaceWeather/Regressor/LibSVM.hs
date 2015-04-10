{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable, FlexibleContexts, FlexibleInstances, FunctionalDependencies, MultiParamTypeClasses, MultiWayIf, OverloadedStrings, TemplateHaskell, TypeFamilies, TypeSynonymInstances #-}
module SpaceWeather.Regressor.LibSVM where

import Control.Lens
import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.Trans.Either
import Data.Foldable
import Data.Maybe
import Data.Traversable
import qualified Data.Aeson.TH as Aeson
import qualified Data.Yaml as Yaml
import qualified Data.Map.Strict as Map
import Data.Monoid
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Numeric.Optimization.Algorithms.CMAES as CMAES
import System.IO
import System.Random
import qualified System.IO.Hadoop as HFS
import System.Process
import System.Timeout
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

-- | record field marked by `a' are subject to machine-learning parameter tuning.
data LibSVMOptionOf a = LibSVMOption
  { _libSVMType       :: Int
  , _libSVMKernelType :: Int
  , _libSVMCost       :: a
  , _libSVMEpsilon    :: Maybe a
  , _libSVMGamma      :: a
  , _libSVMNu         :: Maybe a
  , _libSVMAutomationLevel :: Int
  , _libSVMAutomationPopSize :: Int
  , _libSVMAutomationTolFun :: Double
  , _libSVMAutomationScaling :: Double
  , _libSVMAutomationNoise :: Bool
  } deriving (Eq, Ord, Show, Read, Functor, Foldable, Traversable)
type LibSVMOption = LibSVMOptionOf Double
makeClassy ''LibSVMOptionOf
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 7} ''LibSVMOptionOf

defaultLibSVMOption :: LibSVMOption
defaultLibSVMOption = LibSVMOption
  { _libSVMType       = 3
  , _libSVMKernelType = 2
  , _libSVMCost       = 1
  , _libSVMEpsilon    = Nothing -- 0.001
  , _libSVMGamma      = 0.01
  , _libSVMNu         = Nothing -- 0.5
  , _libSVMAutomationLevel = 0
  , _libSVMAutomationPopSize = 10
  , _libSVMAutomationTolFun = 1e-3
  , _libSVMAutomationScaling = 2
  , _libSVMAutomationNoise = False
  }

libSVMOptionCmdLine :: LibSVMOption -> String
libSVMOptionCmdLine opt =
  printf "-s %d -t %d -c %f %s %s %s"
    (opt^.libSVMType) (opt^.libSVMKernelType) (opt^.libSVMCost) epsilonStr gammaStr nuStr
  where
    gammaStr = "-g " ++ show (opt ^. libSVMGamma)
    epsilonStr = maybe "" (\v -> "-e " ++ show v) (opt^.libSVMEpsilon)
    nuStr = maybe "" (\v -> "-n " ++ show v) (opt^.libSVMNu)

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
    return $ case e of
      Right x  -> x
      Left msg ->
        PredictionSession {
          _predictionStrategyUsed = strategy
        , _predictionSessionResult = PredictionFailure msg
        }


libSVMPerformPrediction :: PredictionStrategy LibSVMOption -> EitherT String IO (PredictionSession LibSVMOption)
libSVMPerformPrediction strategy = do
  let fsp :: FeatureSchemaPack
      fsp = strategy ^. featureSchemaPackUsed
      opt0 :: LibSVMOption
      opt0 = strategy ^. regressorUsed

  featurePack0 <- EitherT $ loadFeatureSchemaPack fsp

  let tgtSchema = strategy ^. predictionTargetSchema
      tgtFn     = strategy ^. predictionTargetFile

  tgtFeature <- loadFeatureWithSchemaT tgtSchema tgtFn

  cvstrSeed <- liftIO randomIO

  let fioPair0 :: FeatureIOPair
      fioPair0 = catFeaturePair fs0 tgtFeature

      fs0 :: [Feature]
      FeaturePack fs0 = featurePack0

      cvstr0 = strategy ^. crossValidationStrategy

      cvstr :: CrossValidationStrategy
      cvstr
        | crossValidationNoise = CVShuffled cvstrSeed cvstr0
        | otherwise            = cvstr0

      pred :: TimeBin -> Bool
      pred = inTrainingSet cvstr

      (fioTrainSet, fioTestSet) = Map.partitionWithKey (\k _ -> pred k) fioPair0

      svmTrainSet = LibSVMFeatures fioTrainSet
      svmTestSet = LibSVMFeatures fioTestSet

      fnTrainSet = workDir ++ "/train.txt"
      fnModel =  workDir ++ "/train.txt.model"
      fnTestSet = workDir ++ "/test.txt"
      fnPrediction = workDir ++ "/test.txt.prediction"

  liftIO $ do
    print cvstr
    encodeFile fnTrainSet svmTrainSet
    encodeFile fnTestSet svmTestSet

--     system "hadoop fs -get /user/nushio/libsvm-3.17/svm-train ."
--     system "hadoop fs -get /user/nushio/libsvm-3.17/svm-predict ."
--     system "hadoop fs -get /user/nushio/executables/cmaes_wrapper.py ."
--     system "hadoop fs -get /user/nushio/executables/cma.py ."

  let
    evaluate :: FilePath -> LibSVMOption -> IO PredictionResult
    evaluate fnDebugFn opt = do
      hPutStrLn stderr $ "testing: " ++ show opt
      hPutStrLn stderr ".";  hFlush stderr
      {- for hadoop log retrieval system tends to replicate the last non-blank line of stderr
         every one second, this is done to cleanse the error log as good as possible. -}

      let
          svmTrainCmd = printf "svm-train -h 0 %s %s %s"
            (libSVMOptionCmdLine opt) fnTrainSet fnModel

          svmPredictCmd = printf "svm-predict %s %s %s"
            fnTestSet fnModel fnPrediction
      isSuccess <- timeout (30 * 60 * 1000000) $ do
        system $ svmTrainCmd
        system $ svmPredictCmd
        return True
      case isSuccess of
        Nothing -> return $ PredictionFailure "LibSVM timeout"
        _ -> evaluateCont fnDebugFn opt

    evaluateCont :: FilePath -> LibSVMOption -> IO PredictionResult
    evaluateCont fnDebugFn opt = do
      predictionStr <- readFile fnPrediction

      let predictions :: [Double]
          predictions = map read $ lines predictionStr

          observations :: [Double]
          observations = map (snd . snd) $ Map.toAscList fioTestSet

          poTimeLine :: TimeLine (Double, Double)
          poTimeLine = Map.fromList $
            zipWith (\p (t, (_,o)) -> (t, (p,o)) ) predictions $
            Map.toList fioTestSet

          poTbl = zip predictions observations

          resultMap0 = Map.fromList $
            [ let logXRF = log (xRayFlux flare1) / log 10 in (flare1 , makeScoreMap poTbl logXRF)
            | flare1 <- defaultFlareClasses]
          ret = PredictionSuccess resultMap0
      hPutStrLn stderr $ "sum TSS : " ++ (show $ prToDouble ret)

      when (fnDebugFn/="") $
         T.writeFile fnDebugFn $ T.unlines $
           ("#time\tprediction\tobservation" :) $
           map (T.pack) $
           map (\(t, (p,o)) -> printf "%d\t%f\t%f" t p o) $
           Map.toList poTimeLine


      return $ ret

  let logOpt0 = fmap log opt0
      lv = opt0 ^. libSVMAutomationLevel
      minimizationTgt :: LibSVMOption -> IO Double
      minimizationTgt =
        fmap (negate . prToDoubleWith (wf lv)) .
        (evaluate "") .
        fmap exp

      wf :: Int -> FlareClass -> ScoreMode -> Double
      wf 2 XClassFlare TrueSkillStatistic = 1
      wf 2 _           _                  = 0
      wf 3 MClassFlare TrueSkillStatistic = 1
      wf 3 _           _                  = 0
      wf 4 CClassFlare TrueSkillStatistic = 1
      wf 4 _           _                  = 0
      wf _ _           TrueSkillStatistic = 1
      wf _ _           _                  = 0

  let
    goBestOpt
      | lv <= 0 = return $ opt0
      | lv >= 1 = do
          logBestOpt <- CMAES.run $ problem
          return $ fmap exp logBestOpt

    problem = (CMAES.minimizeTIO minimizationTgt logOpt0)
      { CMAES.sigma0 = opt0 ^. libSVMAutomationScaling
      , CMAES.tolFun = Just $ opt0 ^. libSVMAutomationTolFun
      , CMAES.scaling = Just $ repeat (log 10)
      , CMAES.noiseHandling = opt0 ^. libSVMAutomationNoise
      , CMAES.otherArgs =
          [("popsize", show $ opt0 ^. libSVMAutomationPopSize)]
      , CMAES.pythonPath = Just "/usr/bin/python"
      , CMAES.cmaesWrapperPath = Just "./cmaes_wrapper.py"}
  bestOpt <- liftIO goBestOpt

  ret <- liftIO $ evaluate (strategy ^. predictionRegressionFile) bestOpt

  liftIO $ do
      T.hPutStrLn stderr $ encode ret

  return $ PredictionSession
     (strategy & regressorUsed .~ bestOpt)
     ret
