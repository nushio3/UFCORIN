{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TypeSynonymInstances #-}
module SpaceWeather.Feature where

import Control.Lens
import Control.Monad
import qualified Data.Map.Strict as Map
import Data.Monoid
import qualified Data.Text as T
import SpaceWeather.Format
import SpaceWeather.TimeLine
import Text.Printf

type Feature = TimeLine Double

type Features = TimeLine [Double]
 
instance Format Feature where
  encode = 
    T.unlines .
    map (\(t,a) -> T.unwords [showT t, showT a] ) .
    Map.toList

  decode txt0 = do
    -- (Either String) monad
    let xs = linesWithComment txt0
        parseLine :: (Int, T.Text) -> Either String (TimeBin, Double)
        parseLine (lineNum, txt) = 
          maybe (Left $ printf "parse error on line %d" lineNum) Right $ do
            -- maybe monad here
            let wtxt = T.words txt
            t <- readAt wtxt 0
            a <- readAt wtxt 1
            return (t,a)
    fmap Map.fromList $ mapM parseLine xs

catFeatures :: [Feature] -> Features
catFeatures [] = Map.empty
catFeatures xs = let (fs1: fss) = reverse xs in
  foldr (Map.intersectionWith (:)) (Map.map (:[]) fs1) fss