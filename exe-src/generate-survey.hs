module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.List
import qualified Data.Map as M
import qualified Data.Text.IO as T
import System.Environment
import System.Process
import System.Random
import Text.Printf

import SpaceWeather.CmdArgs
import SpaceWeather.TimeLine
import SpaceWeather.FlareClass
import SpaceWeather.Feature
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import qualified System.IO.Hadoop as HFS

surveyDir = "survey-cvn-6"

testStrategy :: PredictionStrategyGS
testStrategy = defaultPredictionStrategy  & crossValidationStrategy .~ CVShuffled 12345 CVWeekly

main :: IO ()
main = withWorkDir $ do
  seeds <- goodSeeds
  let perfLimit = 99
  system $ "mkdir -p " ++ surveyDir
  sequence_ [process seeds "bsplC-301" True (2^i) (2^j) (2^iy) (2^jy)
            | i <- [0..9], j <- [i..9],  iy <- [0..9], jy <- [iy..9], (jy-iy)*(j-i)<perfLimit ]
  sequence_ [process seeds "bsplC-301" False (2^i) (2^j) (2^iy) (2^jy)
            | i <- [0..9], j <- [i..9],  iy <- [0..9], jy <- [iy..9] , (jy-iy)*(j-i)<perfLimit]
  sequence_ [process seeds "haarC-2" True (2^i) (2^j) (2^iy) (2^jy)
            | i <- [0..9], j <- [i..9],  iy <- [0..9], jy <- [iy..9] , (jy-iy)*(j-i)<perfLimit]
  sequence_ [process seeds "haarC-2" False (2^i) (2^j) (2^iy) (2^jy)
            | i <- [0..9], j <- [i..9],  iy <- [0..9], jy <- [iy..9] , (jy-iy)*(j-i)<perfLimit]

process :: [Int] -> String -> Bool -> Int -> Int -> Int -> Int -> IO ()
process seeds basisName isStd lower upper lowerY0 upperY0 = do
  strE <- fmap decode $ T.readFile "resource/strategy-template.yml"
  case strE of
    Left msg -> putStrLn msg
    Right strategy -> forM_ [0..9 :: Int] $ \iterID -> do
      let
        (iterIDDiv, iterIDMod) = divMod iterID 2
        seedParity = case iterIDMod of
          0 -> id
          1 -> CVNegate

        strategy2 :: PredictionStrategyGS
        strategy2 = strategy
          & predictionSessionFile .~ ""
          & predictionResultFile .~ ""
          & predictionRegressionFile .~ "/dev/null"
          & featureSchemaPackUsed . fspFilenamePairs %~ (++ fnPairs)
          & crossValidationStrategy .~ seedParity (CVShuffled (seeds !! iterIDDiv) CVWeekly)

        lowerY = if isStd then lowerY0 else lower
        upperY = if isStd then upperY0 else upper

        coordList :: [Int]
        coordList = [2^i | i <- [0..9], lower <= 2^i , 2^i <= upper  ]
        coordListY :: [Int]
        coordListY = [2^i | i <- [0..9], lowerY <= 2^i , 2^i <= upperY  ]

        fnPairs
          | isStd     = genFn <$> coordList <*> coordListY
          | otherwise = coordList >>= genFnN

        genFn :: Int -> Int -> (String,FilePath)
        genFn x y = ("f35Log", printf "/user/nushio/wavelet-features/%s-%04d-%04d.txt" basisString x y)

        genFnN :: Int -> [(String,FilePath)]
        genFnN x  =
          [ ("f35Log", printf "/user/nushio/wavelet-features/%s-%04d-%04d.txt" basisString (0::Int) x)
          , ("f35Log", printf "/user/nushio/wavelet-features/%s-%04d-%04d.txt" basisString x (0::Int))
          , ("f35Log", printf "/user/nushio/wavelet-features/%s-%04d-%04d.txt" basisString x x)]

        basisString :: String
        basisString = printf "%s-%s" basisName (if isStd then "S" else "N")

        candSesFn = strategy ^. predictionSessionFile
        candResFn = strategy ^. predictionResultFile
        candRegFn = strategy ^. predictionRegressionFile

        surveySubDirID :: Integer
        surveySubDirID = (read $ concat $ map show [lower,upper,lowerY,upperY,iterID]) `mod` 997
        surveySubDir :: String
        surveySubDir = printf "%03d" surveySubDirID

        surveyDir2 :: String
        surveyDir2 = surveyDir -- ++ "/" ++ surveySubDir

        fn :: String
        fn = printf "%s/%s-%04d-%04d-%04d-%04d-[%02d]-strategy.yml"
          surveyDir2 basisString
          (lower :: Int) (upper :: Int) lowerY upperY iterID

      system $ printf "mkdir -p %s" surveyDir2
      T.writeFile fn $ encode (strategy2)

      return ()

goodSeeds :: IO [Int]
goodSeeds = do
  Right strategy <- fmap decode $ T.readFile "resource/strategy-template.yml"
  Right goesFeature <- loadFeatureWithSchema
                       (strategy ^. predictionTargetSchema)
                       (strategy ^. predictionTargetFile)
  return (strategy :: PredictionStrategyGS)
  collectIO 5 $ getGoodSeed goesFeature

collectIO :: Int -> IO [a] -> IO [a]
collectIO n m
  | n <= 0    = return []
  | otherwise = do
      xs <- m
      ys <- collectIO (n - length xs) m
      return $ xs ++ ys

getGoodSeed :: Feature -> IO [Int]
getGoodSeed goesFeature = do
  seed <- randomIO
  let
    cvstr = CVShuffled seed CVWeekly
    pred :: TimeBin -> Bool
    pred = inTrainingSet cvstr

    (trainSet, testSet) = M.partitionWithKey (\k _ -> pred k) goesFeature

    countBalance :: FlareClass -> (Double, Double)
    countBalance fc = (n1,n2) where
      countX = length . filter (\v -> v >= log (xRayFlux fc) / log 10) . map snd . M.toList
      n1 = fromIntegral $ countX trainSet
      n2 = fromIntegral $ countX testSet

    isBalanced (n1,n2) = n1 < 1.1 * n2 && n2 < 1.1 * n1

    balances = fmap countBalance defaultFlareClasses
  if all isBalanced balances then print balances >> print seed >> return [seed]
    else return []
