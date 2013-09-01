{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}
module Main where


import           Control.Lens (iso, Iso', (^.), from)
import           Control.Lens.TH
import           Control.Monad
import           Data.List (intercalate)
import qualified Data.Map as Map
import           Data.Maybe
import qualified Data.Set as Set
import qualified Data.Text as Text
import qualified Data.Text.IO as Text
import           Data.Time
import           Safe (atMay, readMay)
import           Text.Printf


import SpaceWeather.Timeline

main :: IO ()
main = do
  goesStr <- Text.readFile "resource/goes.dat"
  hmiStr <- Text.readFile "resource/hmi_mag_flux.dat"
  let
    goesData :: Map.Map TimeBin GoesFeature
    goesData = Map.fromList $ catMaybes $ map parseGoesLine $ Text.lines goesStr
    
    hmiData :: Map.Map TimeBin HmiFeature
    hmiData = Map.fromList $ catMaybes $ map parseHmiLine $ Text.lines hmiStr

    ticks :: [TimeBin]
    ticks = Set.toList $ Set.union
            (Set.fromList $ Map.keys goesData)
            (Set.fromList $ Map.keys hmiData)


  forM_ [0,24,48,72] $ \n -> do
    let
      goesData2 = forecast n catGoesFeatures goesData
      hmiData2 = forecast n catHmiFeatures hmiData

      goesFn :: FilePath
      goesFn = printf "forecast-goes-%d.txt" n
      hmiFn :: FilePath
      hmiFn = printf "forecast-hmi-%d.txt" n
    writeFile goesFn $ unlines $ map pprintGoes $ Map.toList goesData2
    writeFile hmiFn $ unlines $ map pprintHmi $ Map.toList hmiData2
  
