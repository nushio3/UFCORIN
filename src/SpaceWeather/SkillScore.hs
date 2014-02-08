{-# LANGUAGE FlexibleInstances, TemplateHaskell #-}
module SpaceWeather.SkillScore where

import Control.Lens
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.TH as Aeson
import           Data.List
import qualified Data.Map as Map

data ScoreMode 
  = HeidkeSkillScore 
  | TrueSkillStatistic
  | ContingencyTableElem Bool Bool

  deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions ''ScoreMode

-- ScoreMode can be a key to JSON map.
instance Aeson.ToJSON a => Aeson.ToJSON (Map.Map ScoreMode a) where
  toJSON = Aeson.toJSON . Map.fromList . (map (_1 %~ show)) . Map.toList
instance Aeson.FromJSON a => Aeson.FromJSON (Map.Map ScoreMode a) where
  parseJSON = fmap go . Aeson.parseJSON
    where
      go :: Map.Map String a -> Map.Map ScoreMode a  
      go = Map.fromList . (map (_1 %~ read)) . Map.toList



data ScoreReport = ScoreReport
  { _scoreValue :: Double
  , _maximizingThreshold :: Double
  , _contingencyTable :: Map.Map String Double
  }
  deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''ScoreReport

type ScoreMap = Map.Map ScoreMode ScoreReport




type BinaryPredictorScore =  [(Bool, Bool)] -> Double

evalScore :: ScoreMode -> BinaryPredictorScore
evalScore mode arg = 
  case mode of
    HeidkeSkillScore -> hss
    TrueSkillStatistic -> tss
    ContingencyTableElem a b -> count a b

  where
      predictions = map fst arg
      observations = map snd arg

      count :: Bool -> Bool -> Double
      count bx by = 
        fromIntegral $
        length $ 
        filter (\(x,y) -> x==bx && y==by) $
        arg
      
      nTP = count True  True
      nFN = count False True            
      nFP = count True  False
      nTN = count False False
      
      hss = 2*(nTP*nTN - nFN*nFP) //
            ((nTP+nFN)*(nFN+nTN) + (nTP+nFP)*(nFP+nTN))

      tss = nTP//(nTP+nFN) - nFP//(nFP+nTN)

      x//y 
        | y==0 = 0
        | otherwise = x/y


poTblToBools :: Double -> Double -> [(Double, Double)] -> [(Bool,Bool)]
poTblToBools threP threO tbl = 
  [(xp > threP, xo > threO) | (xp, xo) <- tbl]


-- | Returns the pair of the maximum found and the threshold
searchThreshold :: [(Double,Double)] -> BinaryPredictorScore -> Double -> (Double, Double)
searchThreshold tbl score threO = 
  maximum $ take (length thres0 + 40) stPairs 
  where
    scoreOf threP = score $ poTblToBools threP threO tbl

    stPairs = [(scoreOf t1, t1) | t1 <- thres]
    thres = thres0
      ++ concat [ mkThre i | i <- [0..]]
    thres0 = [threO + dt | dt <- [-2, -1.98 .. 2]]

    mkThre i = [tbsf-dt,tbsf+dt]
      where 
        (_, tbsf) = maximum $ take (i+length thres0) stPairs
        dt = 0.02 * exp (negate $ fromIntegral i / 5)

makeScoreMap :: [(Double,Double)] -> Double -> ScoreMap
makeScoreMap tbl threO = Map.fromList
  [ (mode1, go mode1)
  | mode1 <- [HeidkeSkillScore, TrueSkillStatistic]]
  where 
    eSX threP mode = evalScore mode $ poTblToBools threP threO tbl

    go :: ScoreMode -> ScoreReport
    go mode = let
      (val,threP) = searchThreshold tbl (evalScore mode) threO
      in ScoreReport
        { _scoreValue = val
        , _maximizingThreshold = threP
        , _contingencyTable = 
          Map.fromList
            [ ("pToT", eSX threP $ ContingencyTableElem True  True )
            , ("pToF", eSX threP $ ContingencyTableElem True  False)
            , ("pFoT", eSX threP $ ContingencyTableElem False True )
            , ("pFoF", eSX threP $ ContingencyTableElem False False)
            ]
        }
