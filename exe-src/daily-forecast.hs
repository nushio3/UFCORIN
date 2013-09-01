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

----------------------------------------------------
-- Utility Functions
----------------------------------------------------

average :: Fractional a => [a] -> a
average xs = sum xs / (fromIntegral $ length xs)

----------------------------------------------------
-- Time features
----------------------------------------------------

epoch :: UTCTime
epoch = UTCTime (fromGregorian 2011 1 1) 0

type TimeBin = Integer

discreteTime :: Iso' UTCTime TimeBin
discreteTime = iso f g
  where
    f t  = round $ toRational (diffUTCTime t epoch) / 3600
    g tb = addUTCTime (fromIntegral (tb * 3600))  epoch

----------------------------------------------------
-- The features
----------------------------------------------------

data GoesFeature
  = GoesFeature
  { _avgXray :: Double
  , _maxXray :: Double}

$(makeLenses ''GoesFeature)

data HmiFeature
  = HmiFeature
  { _sphereAvg :: Double
  , _sphereAbsSum :: Double}

$(makeLenses ''HmiFeature)

----------------------------------------------------
-- Parsers
----------------------------------------------------

readMayT :: Read a => Text.Text -> Maybe a
readMayT = readMay . Text.unpack

readAt xs n = atMay xs n >>= readMayT


parseTimeBin :: [Text.Text] -> Maybe TimeBin
parseTimeBin ws = do
  dayStr <- ws `atMay` 0

  let dayParts = Text.splitOn "-" dayStr 

  yearVal <- dayParts `readAt` 0
  monthVal <- dayParts `readAt` 1
  dayVal <- dayParts `readAt` 2

  hourVal <- ws `readAt` 1
  
  let day = fromGregorian yearVal monthVal dayVal
      sec = secondsToDiffTime $ hourVal * 3600
      t = UTCTime day sec
      
  return $ t ^. discreteTime

parseGoesLine :: Text.Text -> Maybe (TimeBin, GoesFeature)
parseGoesLine str = do -- maybe monad
  let ws = Text.words str

  t <- parseTimeBin ws
  
  avgXray0 <- ws `readAt` 2 
  maxXray0 <- ws `readAt` 3 

  guard $ avgXray0 > 1e-8
  guard $ maxXray0 > 1e-8

  return (t, GoesFeature avgXray0 maxXray0)

parseHmiLine :: Text.Text -> Maybe (TimeBin, HmiFeature)
parseHmiLine str = do -- maybe monad
  let ws = Text.words str

  t <- parseTimeBin ws
  
  avgXray0 <- ws `readAt` 2 
  absSum0 <- ws `readAt` 3 

  guard $ abs avgXray0 < 1.5e7
  guard $ absSum0  < 3e8

  return (t, HmiFeature avgXray0 absSum0)

----------------------------------------------------
-- Create forecasting data
----------------------------------------------------

forecast :: Int -> ([a]->a) -> Map.Map TimeBin a -> Map.Map TimeBin a 
forecast span f src = ret
  where
    ts = Map.keys src
    ret = Map.fromList $ catMaybes $ map go ts

    go t = do -- maybe monad
      let futureTs = [t..t+fromIntegral span]

      -- require all data to be present
      --vals <- forM futureTs (\t -> Map.lookup t src)

      -- require some of the data to be present.
      let vals = catMaybes $ map (\t -> Map.lookup t src) futureTs
      guard $ length vals > 0
      
      return (t,f vals)

catGoesFeatures :: [GoesFeature] -> GoesFeature
catGoesFeatures xs =
  GoesFeature
  { _avgXray = average $ map (^. avgXray) xs
  , _maxXray = maximum $ map (^. maxXray) xs             
  }

catHmiFeatures :: [HmiFeature] -> HmiFeature
catHmiFeatures xs =
  HmiFeature
  { _sphereAvg = average $ map (^. sphereAvg) xs
  , _sphereAbsSum = average $ map (^. sphereAbsSum) xs
  }



----------------------------------------------------
-- The main program
----------------------------------------------------

pprintTime :: TimeBin -> String
pprintTime t = unwords
  [ intercalate "-" $ take 2 $ words $ show $ t ^. from discreteTime
  , printf "%8d" t]

pprintGoes :: (TimeBin, GoesFeature) -> String
pprintGoes (t,gd) = printf "%s\t%e\t%e" (pprintTime t) (gd ^. avgXray) (gd ^. maxXray)

pprintHmi :: (TimeBin, HmiFeature) -> String
pprintHmi (t,hd) = printf "%s\t%e\t%en" (pprintTime t) (hd ^. sphereAvg) (hd ^. sphereAbsSum)

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
  
