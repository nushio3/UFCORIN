{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable, FlexibleContexts, FlexibleInstances, FunctionalDependencies, MultiParamTypeClasses, MultiWayIf, OverloadedStrings, TemplateHaskell, TypeSynonymInstances #-}
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

data LibSVMOptionOf a = LibSVMOption 
  { _libSVMType       :: Int
  , _libSVMKernelType :: Int
  , _libSVMCost       :: a
  , _libSVMEpsilon    :: a
  , _libSVMGamma      :: Maybe a
  , _libSVMNu         :: a
  , _libSVMAutomationLevel :: Int
  } deriving (Eq, Ord, Show, Read, Functor, Foldable, Traversable)
type LibSVMOption = LibSVMOptionOf Double
makeClassy ''LibSVMOptionOf
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 7} ''LibSVMOptionOf

defaultLibSVMOption :: LibSVMOption
defaultLibSVMOption = LibSVMOption
  { _libSVMType       = 3
  , _libSVMKernelType = 2
  , _libSVMCost       = 1  
  , _libSVMEpsilon    = 0.001
  , _libSVMGamma      = Nothing
  , _libSVMNu         = 0.5
  , _libSVMAutomationLevel = 0
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
 
  liftIO $ do
    encodeFile fnTrainSet svmTrainSet
    encodeFile fnTestSet svmTestSet

    system "hadoop fs -get /user/nushio/libsvm-3.17/svm-train ."
    system "hadoop fs -get /user/nushio/libsvm-3.17/svm-predict ."
    system "cabal install cmaes"
  
  let 
    evaluate :: LibSVMOption -> IO PredictionResult
    evaluate opt = do
      hPutStrLn stderr $ "testing: " ++ show opt
      let
          svmTrainCmd = printf "./svm-train %s %s %s"
            (libSVMOptionCmdLine opt) fnTrainSet fnModel
    
          svmPredictCmd = printf "./svm-predict %s %s %s"
            fnTestSet fnModel fnPrediction
      system $ svmTrainCmd
      system $ svmPredictCmd

      predictionStr <- readFile fnPrediction  

      let predictions :: [Double]
          predictions = map read $ lines predictionStr
    
          observations :: [Double]
          observations = map (snd . snd) $ Map.toAscList fioTestSet
    
          poTbl = zip predictions observations
    
          resultMap0 = Map.fromList $ 
            [ let logXRF = log (xRayFlux flare1) / log 10 in (flare1 , makeScoreMap poTbl logXRF)
            | flare1 <- defaultFlareClasses]
          ret = PredictionSuccess resultMap0
      hPutStrLn stderr $ "sum TSS : " ++ (show $ prToDouble ret)
      return $ ret

  let logOpt0 = fmap log (opt0 & libSVMGamma .~ Just 0.01)
      minimizationTgt :: LibSVMOption -> IO Double
      minimizationTgt = 
        fmap (negate . prToDouble) .            
        evaluate .               
        fmap exp  
  let 
    lv = opt0 ^. libSVMAutomationLevel
    goBestOpt
      | lv <= 0 = return $ opt0
      | lv >= 1 = do
          logBestOpt <- CMAES.run $ problem
          return $ fmap exp logBestOpt

    problem = (CMAES.minimizeTIO minimizationTgt logOpt0)
      { CMAES.sigma0 = 1 -- we need big sigma to find out the best config!
      , CMAES.tolFun = Just 1e-4
      , CMAES.scaling = Just $ repeat (log 1e3) }  
  bestOpt <- liftIO goBestOpt

  ret <- liftIO $ evaluate bestOpt

  liftIO $ T.hPutStrLn stderr $ encode ret
  return $ PredictionSession
     (strategy & regressorUsed .~ bestOpt) 
     ret


prToDouble :: PredictionResult -> Double
prToDouble (PredictionFailure _) = 0
prToDouble (PredictionSuccess m) = 
  Prelude.sum $
  map _scoreValue $
  catMaybes $
  map (Map.lookup TrueSkillStatistic )$ 
  map snd $ 
  Map.toList m

