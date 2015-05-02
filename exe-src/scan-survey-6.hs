#!/usr/bin/env runhaskell
module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.Char
import Data.Function (on)
import Data.List
import Data.Maybe
import Data.List.Split
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as M
import qualified Data.Set as S
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

mean :: [Double] -> Double
mean = fst . meanDevi

devi :: [Double] -> Double
devi = snd . meanDevi


data Key = Key { strategy :: Strategy,     flareClass :: FlareClass,   cvIndex :: CV}
            deriving (Eq, Ord, Show)

data Strategy = Strategy
  { strategyStr :: String ,
    lowerX :: Int,
    upperX :: Int,
    lowerY :: Int,
    upperY :: Int
    }
            deriving (Eq, Ord, Show)

type CV = Int
cvSet = [0..9] :: [CV]

imgPerSolarDiameter :: Double
imgPerSolarDiameter = 1024/959

type Category = (Int,Int)


main :: IO ()
main = do
  dirName <- fmap head getArgs
  fns <- fmap lines $ readSystem0 $ printf "find %s | grep 'result.yml'" dirName

  print $ length fns

  dataSet  <- fmap (M.unionsWith better) $ forM fns $ analyze

  let strategySet :: [Strategy]
      strategySet =
        S.toList $ S.fromList $
        map strategy $ M.keys dataSet

  let dataByFC :: M.Map (FlareClass,CV) [Double]
      dataByFC = M.mapKeysWith (++) (\k -> (flareClass k, cvIndex k)) dataSet

      statByFC :: M.Map (FlareClass,CV) (Double,Double)
      statByFC = M.map meanDevi dataByFC

      meanFC :: FlareClass -> CV -> Double
      meanFC f c = snd $ statByFC M.! (f,c)

  let dataBySF :: M.Map (Strategy, FlareClass) [Double]
      dataBySF = M.mapKeysWith (++) (\k -> (strategy k, flareClass k)) dataSet

      statBySF :: M.Map (Strategy, FlareClass) (Double,Double)
      statBySF = M.map meanDevi dataBySF

      meanSF :: Strategy -> FlareClass -> Double
      meanSF s f  = snd $ statBySF M.! (s,f)

      dataBySFA :: M.Map (Strategy,FlareClass) [Double]
      dataBySFA = M.fromList
        [((s,f), [meanSF s f - meanFC f c | c <- cvSet, x <- fromMaybe [] (M.lookup (Key s f c) dataSet )])
        | s <- strategySet, f <- defaultFlareClasses]

      statBySFA :: M.Map (Strategy, FlareClass) (Double,Double)
      statBySFA = M.map meanDevi dataBySFA

  let
      statBySFB :: M.Map (Strategy,FlareClass) (Double, Double)
      statBySFB =
        M.fromList
          [ ((s,f), (m,d))
          | s <- strategySet, f <- defaultFlareClasses,
            (m,_) <- maybeToList $ M.lookup (s,f) statBySF,
            (_,d) <- maybeToList $ M.lookup (s,f) statBySFA
            ]


  let theBest :: M.Map FlareClass (Strategy, [Double])
      theBest = M.mapKeysWith better2 flareClass $
                M.mapWithKey (\k xs -> (strategy k, xs)) dataSet
  mapM_ print $ M.toList theBest


  let chosens :: Char -> FlareClass -> M.Map Category (Strategy, (Double,Double))
      chosens d0 fc =
        M.mapKeysWith better1 (\(s,f) -> makeCat d0 $ s) $
        M.mapWithKey (\(s,f) md -> (s , md)) $
        M.filterWithKey (\(_,f) _ -> f==fc) $
        statBySFB

  forM_ defaultFlareClasses $ \fc -> do
    let tmpFn = "tmp.txt" :: FilePath
    writeFile tmpFn $ unlines
      [ printf "%d %f %f" c m d
      | c <- cvSet, let (m,d) = statByFC M.! (fc,c)
      ]

    let figFn = printf "figure/CV-dependency-%s.txt" (show fc) :: FilePath
        lcstr = case fc of
          XClassFlare -> "#FF0000"
          MClassFlare -> "#008000"
          CClassFlare -> "#0000FF"

    _ <- readSystem "gnuplot" $ unlines
       [ "set term postscript landscape enhanced color 25"
       , printf "set out '%s'" figFn
       , "set xrange [0.5:10.5]"
       , "set xlabel 'CV Data Index'"
       , "set ylabel 'True Skill Statistic'"
       , printf "plot '%s' u ($1+1):2:3 w yerr t '' pt 0 lw 4 lc rgb '%s'" tmpFn  lcstr
       ]
    return ()



  forM_ "xy" $ \dirChar -> do
    forM_ defaultFlareClasses $ \fc -> do
      let mds :: [(Double, Double)]
          mds = concat [map snd $ M.elems $ chosens d0 fc | d0 <- "xy"]
          yrangeLo = minimum [m-d | (m,d) <- mds] - 0.002
          yrangeUp = maximum [m+d | (m,d) <- mds] + 0.002

      let
          ppr :: (Category,(Double,Double)) -> String
          ppr ((lo,hi),(m,d)) =
             printf "%f %f %f %f" (inS lo) (inS hi) m d

          inS :: Int -> Double
          inS x = imgPerSolarDiameter / fromIntegral x

          tmpFn = "tmp.txt"

          xlabel = case dirChar of
            'x' -> "Horizontal Scale"
            'y' -> "Vertical Scale"

          figFn :: FilePath
          figFn = printf "figure/wavelet-range-%s-%c.eps" (show fc) dirChar

      writeFile tmpFn $ unlines $ map ppr $ M.toList $ M.map snd (chosens dirChar fc)

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

    makeCat :: Char -> Strategy -> Category
    makeCat 'x' s = (lowerX s, upperX s)
    makeCat 'y' s = (lowerY s, upperY s)
    makeCat c   _ = error $ "unsupported one direction " ++ [c]

    better :: [Double] -> [Double] -> [Double]
    better a b
      | mean a > mean b = a
      | otherwise       = b

    better2 :: (a, [Double]) -> (a, [Double]) -> (a, [Double])
    better2 a b
      | mean (snd a) > mean (snd b) = a
      | otherwise                   = b

    better1 :: (a, (Double, Double)) -> (a, (Double, Double)) -> (a, (Double, Double))
    better1 a b
      | fst (snd a) > fst (snd b) = a
      | otherwise                 = b



analyze :: FilePath -> IO (M.Map Key [Double])
analyze fn = do
  let
      keys :: [Int]
      keys = map read $ reverse $  take 4 $ drop 1 $reverse $ splitOn "-" keyStr
      keyStr = takeWhile (/='[') fn

      cvIndex0 = read $ takeWhile isDigit $ drop 1$ dropWhile (/='[') fn
      [lx,ux,ly,uy] = keys
  let fn2pr :: FilePath -> IO PredictionResult
      fn2pr fn = do
        txt <- T.readFile fn
        let Right ret = decode txt
        return ret

      mRet :: PredictionResult ->  M.Map Key Double
      mRet pr = let  prm = pr ^. predictionResultMap
        in M.fromList [(mkKey fc, (prm M.! fc M.! TrueSkillStatistic)^. scoreValue) | fc <- defaultFlareClasses]

      mkKey :: FlareClass -> Key
      mkKey fc = Key{strategy = s, cvIndex = cvIndex0, flareClass = fc}
        where
          s = Strategy{strategyStr = keyStr, lowerX = lx, upperX = ux, lowerY = ly, upperY = uy}

  pr <- fn2pr fn

  return $ M.map (:[])$ mRet pr
