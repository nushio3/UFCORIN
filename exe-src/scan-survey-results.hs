#!/usr/bin/env runhaskell
module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.Char
import Data.List
import qualified Data.Map as M
import System.Environment
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
  sequence_ $ do
    waveletNames <- ["bsplC-301", "haarC-2"]
    [   study dirName waveletNames 'S' 'x'
      , study dirName waveletNames 'S' 'y'
      , study dirName waveletNames 'N' 'x'
      ]

{-
-}

study :: String -> String -> Char -> Char -> IO ()
study dirName waveletName stdChar directionChar = do
  lsRet <- readSystem0 $ printf "ls %s/%s-%c*-result.yml" dirName waveletName stdChar
  bars <- mapM (process directionChar) $ lines lsRet

  let nm = foldr (\(k,v) -> M.insertWith max k v) M.empty bars

      tmpFn = "tmp.dat"
      ppr (k,v) = printf "%s %f\n" (unwords $ map show k) v :: String

      figFn :: FilePath
      figFn = printf "figure/%s-%s-%c-%c.eps" (filter (/='/') dirName) waveletName stdChar directionChar

      xlabel = case (stdChar, directionChar) of
        ('S', 'x') -> "Horizontal scale of the feature"
        ('S', 'y') -> "Vertical scale of the feature"
        ('N', _) -> "Scale of the feature"

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

process :: Char -> FilePath -> IO ([Double], Double)
process keyword fn = do
  Right result0 <- fmap decode $ T.readFile fn
  let result :: PredictionResult
      result = result0
      prm = result ^. predictionResultMap
      xcm = prm M.! MClassFlare M.! TrueSkillStatistic
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
        _   -> error "undefined filtering direction"

  return (nums, xcsv)
