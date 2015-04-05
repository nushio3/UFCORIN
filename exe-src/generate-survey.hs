{-# LANGUAGE OverloadedStrings #-}
module Main where
import Control.Applicative
import Control.Lens
import Control.Monad
import Data.List
import System.Environment
import Text.Printf

import SpaceWeather.CmdArgs
import SpaceWeather.FeaturePack
import SpaceWeather.Format
import SpaceWeather.Prediction
import SpaceWeather.Regressor.General
import qualified System.IO.Hadoop as HFS
import qualified Data.Text.IO as T

main :: IO ()
main = do
  sequence_ [process (2^i) (2^j) (2^iy) (2^jy) | i <- [0..9], j <- [i+1..9],  iy <- [0..9], jy <- [iy+1..9] ]


process :: Int -> Int -> Int -> Int -> IO () 
process lower upper lowerY upperY = withWorkDir $ do
  strE <- fmap decode $ T.readFile "resource/strategy-template.yml"
  case strE of 
    Left msg -> putStrLn msg
    Right strategy -> do
      let
        strategy2 :: PredictionStrategyGS
        strategy2 = strategy 
          & predictionSessionFile .~ finalSesFn        
          & predictionResultFile .~ finalResFn        
          & predictionRegressionFile .~ finalRegFn        
          & featureSchemaPackUsed . fspFilenamePairs %~ (++  (genFn <$> coordList <*> coordListY))

        coordList :: [Int]
        coordList = [2^i | i <- [0..9], lower <= 2^i , 2^i <= upper  ]
        coordListY :: [Int]
        coordListY = [2^i | i <- [0..9], lowerY <= 2^i , 2^i <= upperY  ]

        genFn :: Int -> Int -> (String,FilePath)
        genFn x y = ("f35Log", printf "/user/nushio/wavelet-features/bsplC-301-S-%04d-%04d.txt" x y)
        
        candSesFn = strategy ^. predictionSessionFile 
        candResFn = strategy ^. predictionResultFile 
        candRegFn = strategy ^. predictionRegressionFile

        fn :: String
        fn = printf "survey2/%04d-%04d-%04d-%04d.yml" (lower :: Int) (upper :: Int) lowerY upperY

        finalSesFn 
          | candSesFn /= "" = candSesFn
          | ".yml" `isSuffixOf` fn = (++"-session.yml") $ reverse $ drop 4 $ reverse fn  
          | otherwise              = fn ++ ".session.yml" 

        finalResFn
          | candResFn /= "" = candResFn
          | ".yml" `isSuffixOf` fn = (++"-result.yml") $ reverse $ drop 4 $ reverse fn  
          | otherwise              = fn ++ ".result.yml" 

        finalRegFn
          | candRegFn /= "" = candRegFn
          | ".yml" `isSuffixOf` fn = (++"-regres.txt") $ reverse $ drop 4 $ reverse fn  
          | otherwise              = fn ++ ".regress.txt" 

      T.writeFile fn $ encode (strategy2)

      return ()
  
