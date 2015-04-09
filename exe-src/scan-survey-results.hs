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
import qualified Data.Text.IO as T

main :: IO ()
main = do
  argv <- getArgs
  ls <- mapM process argv

  let nm = foldr (\(k,v) -> M.insertWith max k v) M.empty ls
  forM_ (M.toList nm) $ \(k,v) -> do
    printf "%s %f\n" (unwords $ map show k) v




process :: FilePath -> IO ([Int], Double)
process fn = do
  Right result0 <- fmap decode $ T.readFile fn
  let result :: PredictionResult
      result = result0
      prm = result ^. predictionResultMap
      xcm = prm M.! XClassFlare M.! TrueSkillStatistic
      xcsv = xcm ^. scoreValue

      nums0 :: [Int]
      nums0 = reverse $ take 4 $ reverse $ map read$ words $ map eraseChar fn
      eraseChar c
        | isDigit c = c
        | otherwise = ' '
      nums = drop 2 nums0
  return (nums, xcsv)
