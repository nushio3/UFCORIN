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
    scoreMap :: M.Map FlareClass [Double]}
            deriving Show


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

  let chosens :: Char -> FlareClass -> M.Map Category Sample
      chosens dirChar fc = M.unionsWith (better fc) $ map (makeCand dirChar) dataSet


  forM_ "xy" $ \dirChar -> do
    forM_ defaultFlareClasses $ \fc -> do
      let theBest = foldr1 (better fc)  dataSet
      print theBest

      let bothChosens = concat [M.elems $ chosens d0 fc | d0 <- "xy"]
          scoreStat = [meanDevi$  scoreMap sample M.! fc| sample <- bothChosens]
          yrangeLo = minimum [m-d | (m,d) <- scoreStat] - 0.002
          yrangeUp = maximum [m+d | (m,d) <- scoreStat] + 0.002
      let
          ppr :: (Category,Sample) -> String
          ppr ((lo,hi),s) =
            let (m,d) = meanDevi $ scoreMap s M.! fc
            in printf "%f %f %f %f" (inS lo) (inS hi) m d

          inS :: Int -> Double
          inS x = imgPerSolarDiameter / fromIntegral x

          tmpFn = "tmp.txt"

          xlabel = case dirChar of
            'x' -> "Horizontal Scale"
            'y' -> "Vertical Scale"

          figFn :: FilePath
          figFn = printf "figure/wavelet-range-%s-%c.eps" (show fc) dirChar

      writeFile tmpFn $ unlines $ map ppr $ M.toList (chosens dirChar fc)

      let plot1 :: String
          plot1 = printf "'%s' u (($1+$2)/2):3:($3-$4):($3+$4) w yerr t '' pt 0 lw 2 lt 2 lc rgb '%s'" tmpFn lcstr
          lcstr = case fc of
            XClassFlare -> "#FF8080"
            MClassFlare -> "#80FF80"
            CClassFlare -> "#8080FF"
      let plot2 :: String
          plot2 = printf "'%s' u (($1+$2)/2):3:(0.99*$1):(1.01*$2) w xerr t '' pt 0 lt 1 lw 4 lc rgb '%s'" tmpFn lcstr
          lcstr = case fc of
            XClassFlare -> "#FF0000"
            MClassFlare -> "#008000"
            CClassFlare -> "#0000FF"

      _ <- readSystem "gnuplot" $ unlines
           [ "set term postscript landscape enhanced color 25"
           , printf "set out '%s'" figFn
           , "set log x; set grid "
           , "set xrange [0.001:1]"
           , printf "set yrange [%f:%f]" yrangeLo yrangeUp
           , printf "set xlabel '%s'" xlabel
           , "set ylabel 'True Skill Statistic'"
           , printf "plot %s, %s" plot1 plot2
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
      where s x = fst $ meanDevi $ scoreMap x M.! fc

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
      pRets = -- M.map couple $
        M.unionsWith (++) (map mRet prs)
--       couple (a:b:rest) = ((a+b)/2):couple rest
--       couple rest = rest


  return Sample{key = keyStr, lowerX = lx, upperX = ux, lowerY = ly, upperY = uy,
                scoreMap = pRets}
