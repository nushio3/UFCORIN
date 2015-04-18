module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.List
import qualified Data.Text.IO as T
import System.Environment
import System.Process
import System.Random
import Text.Printf

import SpaceWeather.CmdArgs
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import qualified System.IO.Hadoop as HFS

surveyDir = "survey-cvn-4"

testStrategy :: PredictionStrategyGS
testStrategy = defaultPredictionStrategy  & crossValidationStrategy .~ CVShuffled 12345 CVWeekly

main :: IO ()
main = withWorkDir $ do
  let perfLimit = 99
  system $ "mkdir -p " ++ surveyDir
  sequence_ [process "bsplC-301" True (2^i) (2^j) (2^iy) (2^jy) | i <- [0..9], j <- [i..9],  iy <- [0..9], jy <- [iy..9], (jy-iy)*(j-i)<perfLimit ]
  sequence_ [process "bsplC-301" False (2^i) (2^j) (2^iy) (2^jy) | i <- [0..9], j <- [i..9],  iy <- [0..9], jy <- [iy..9] , (jy-iy)*(j-i)<perfLimit]
  sequence_ [process "haarC-2" True (2^i) (2^j) (2^iy) (2^jy) | i <- [0..9], j <- [i..9],  iy <- [0..9], jy <- [iy..9] , (jy-iy)*(j-i)<perfLimit]
  sequence_ [process "haarC-2" False (2^i) (2^j) (2^iy) (2^jy) | i <- [0..9], j <- [i..9],  iy <- [0..9], jy <- [iy..9] , (jy-iy)*(j-i)<perfLimit]

process :: String -> Bool -> Int -> Int -> Int -> Int -> IO ()
process basisName isStd lower upper lowerY0 upperY0 = do
  strE <- fmap decode $ T.readFile "resource/strategy-template.yml"
  seeds <- replicateM 5 randomIO
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
