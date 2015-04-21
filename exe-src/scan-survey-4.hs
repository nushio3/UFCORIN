#!/usr/bin/env runhaskell
module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.Char
import Data.List
import Data.List.Split
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as M
import System.Environment
import System.IO
import Text.Printf
import qualified Statistics.Sample as Stat
import qualified Data.Vector.Unboxed as V

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

meanDevi :: [Double] -> (Double, Double)
meanDevi xs = let vx = V.fromList xs
                  (m,v) = Stat.meanVarianceUnb vx
              in (m,sqrt v)


data Sample = Sample
  { key :: String ,
    lowerX :: Int,
    upperX :: Int,
    lowerY :: Int,
    upperY :: Int,
    scoreMap :: M.Map FlareClass (Double, Double)}



imgPerSolarDiameter :: Double
imgPerSolarDiameter = 1024/959

type FileSet = (String, [FilePath])
type Category = (Int,Int)


main :: IO ()
main = do
  dirName <- fmap head getArgs
  fns <- fmap lines $ readSystem0 $ printf "find %s | grep 'result.yml'" dirName

  print $ length fns

  let fileSet :: [FileSet]
      fileSet = M.toList $ M.map sort $ M.unionsWith (++) $ map setKey fns
  dataSet <- forM fileSet $ analyze

  forM_ "xy" $ \dirChar -> do
    forM_ defaultFlareClasses $ \fc -> do
      let chosens :: M.Map Category Sample
          chosens = M.unionsWith (better fc) $ map (makeCand dirChar) dataSet

          ppr :: (Category,Sample) -> String
          ppr ((lo,hi),s) =
            let (m,d) = scoreMap s M.! fc
            in printf "%f %f %f %f" (inS lo) (inS hi) m d

          inS :: Int -> Double
          inS x = imgPerSolarDiameter / fromIntegral x

          tmpFn = "tmp.txt"

          xlabel = case dirChar of
            'x' -> "Horizontal Scale"
            'y' -> "Vertical Scale"

          figFn :: FilePath
          figFn = printf "figure/wavelet-range-%s-%c.eps" (show fc) dirChar

      writeFile tmpFn $ unlines $ map ppr $ M.toList chosens

      _ <- readSystem "gnuplot" $ unlines
           [ "set term postscript landscape enhanced color 20"
           , printf "set out '%s'" figFn
           , "set log x; set grid "
           , "set xrange [0.001:1]"
           , printf "set xlabel '%s'" xlabel
           , "set ylabel 'True Skill Statistic'"
           , printf "plot '%s' u (($1+$2)/2):3:(0.99*$1):(1.01*$2):($3-$4):($3+$4) w xyerr t '' pt 0 lw 3" tmpFn
           ]
      return ()
  where
    setKey :: String -> M.Map String [String]
    setKey x = M.singleton (takeWhile (/='[') x) [x]

    makeCand :: Char -> Sample -> M.Map Category Sample
    makeCand 'x' s = M.singleton (lowerX s, upperX s) s
    makeCand 'y' s = M.singleton (lowerY s, upperY s) s
    makeCand c   _ = error $ "unsupported one direction " ++ [c]

    better :: FlareClass -> Sample -> Sample -> Sample
    better fc a b
      | s a > s b = a
      | otherwise = b
      where s x = fst (scoreMap x M.! fc)

analyze :: FileSet -> IO Sample
analyze (keyStr, fns) = do
  let
      keys :: [Int]
      keys = map read $ reverse $  take 4 $ drop 1 $reverse $ splitOn "-" keyStr
      [lx,ux,ly,uy] = keys
  let fn2pr :: FilePath -> IO PredictionResult
      fn2pr fn = do
        txt <- T.readFile fn
        let Right ret = decode txt
        return ret

      mRet :: PredictionResult ->  M.Map FlareClass [Double]
      mRet pr = let  prm = pr ^. predictionResultMap
        in M.fromList [(fc, [(prm M.! fc M.! TrueSkillStatistic)^. scoreValue ]) | fc <- defaultFlareClasses]

  prs <- mapM fn2pr fns
  let pRets :: M.Map FlareClass [Double]
      pRets = M.map couple $ M.unionsWith (++) (map mRet prs)
      couple (a:b:rest) = ((a+b)/2):couple rest
      couple rest = rest


  return Sample{key = keyStr, lowerX = lx, upperX = ux, lowerY = ly, upperY = uy,
                scoreMap = M.map meanDevi pRets}
