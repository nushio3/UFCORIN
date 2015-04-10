#!/usr/bin/env runhaskell
module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.Char
import Data.List
import qualified Data.Map as M
import System.Environment
import System.IO
import Text.Printf

import SpaceWeather.CmdArgs
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.FlareClass
import SpaceWeather.SkillScore
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import qualified System.IO.Hadoop as HFS
import System.System
import qualified Data.Text.IO as T



imgPerSolarDiameter :: Double
imgPerSolarDiameter = 1.144

main :: IO ()
main = do
  (dirName: _)  <- getArgs
  sequence_ $ do -- List monad
    waveletNames <- ["bsplC-301", "haarC-2"]
    fc <- defaultFlareClasses
    [   study dirName waveletNames 'S' fc 'x'
      , study dirName waveletNames 'S' fc 'y'
      , study dirName waveletNames 'N' fc 'x'
      ]
  sequence_ $ do -- List monad
    fc <- defaultFlareClasses
    dc <- ['x', 'y']
    [study dirName "*" '*' fc dc]


study :: String -> String -> Char -> FlareClass -> Char -> IO ()
study dirName waveletName stdChar fc directionChar = do
  lsRet <- readSystem0 $ printf "ls %s/%s-%c*-result.yml" dirName waveletName stdChar
  let matchedFns = lines lsRet

  bars <- mapM (process fc directionChar) $ matchedFns

  let nm = foldr (\(k,v) -> M.insertWith max k v) M.empty bars

      tmpFn = "tmp.dat"
      ppr (k,v) = printf "%s %f\n" (unwords $ map show k) v :: String

      changeAsterisk :: Char -> String
      changeAsterisk '*' = "all"
      changeAsterisk c = [c]

      figFn :: FilePath
      figFn = printf "figure/%s-%s-%s-%s-%c.eps" (filter (/='/') dirName)
              (concat $ map changeAsterisk waveletName)
              (changeAsterisk stdChar) (take 6 $ show fc) directionChar

      xlabel = case (stdChar, directionChar) of
        ('N', _) -> "Scale of the feature"
        (_  , 'x') -> "Horizontal scale of the feature"
        (_  , 'y') -> "Vertical scale of the feature"
        x          -> error $ "Unknown xlabel: " ++ show x

  hPutStrLn stderr figFn

  let bestFn = snd $ maximum $ zip (map snd bars) matchedFns

  hPutStrLn stderr $ "best fn: " ++ bestFn

  writeFile tmpFn $ unlines $
     map ppr $ M.toList nm

  _ <- readSystem "gnuplot" $ unlines
        [ "set term postscript landscape enhanced color 20"
        , printf "set out '%s'" figFn
        , "set log x; set grid "
        , "set xrange [0.001:1]"
--        , "set yrange [0.6:0.72]"
        , printf "set xlabel '%s'" xlabel
        , "set ylabel 'True Skill Statistic'"
        , printf "plot '%s' u (($1+$2)/2):3:(0.99*$1):(1.01*$2) w xerr t '' pt 0 lw 3" tmpFn
        ]
  return ()

process :: FlareClass -> Char -> FilePath -> IO ([Double], Double)
process fc keyword fn = do
  Right result0 <- fmap decode $ T.readFile fn
  let result :: PredictionResult
      result = result0
      prm = result ^. predictionResultMap
      xcm = prm M.! fc M.! TrueSkillStatistic
      xcsv = xcm ^. scoreValue

      nums0 :: [Int]
      nums0 = reverse $ take 4 $ reverse $ map read$ words $ map eraseChar fn
      eraseChar c
        | isDigit c = c
        | otherwise = ' '
      nums = map (\n -> imgPerSolarDiameter / fromIntegral n) $ filter0 nums0
      filter0 = case keyword of
        'x' -> take 2
        'y' -> drop 2
        d   -> error $ "undefined filtering direction: " ++ show d

  return (nums, xcsv)
