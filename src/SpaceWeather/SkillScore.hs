{-# LANGUAGE FlexibleInstances, TemplateHaskell #-}
module SpaceWeather.SkillScore where

import Control.Lens
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.TH as Aeson
import qualified Data.Map as Map

data ScoreMode = HeidkeSkillScore | TrueSkillStatistic
  deriving (Eq, Ord, Show, Read)
Aeson.deriveJSON Aeson.defaultOptions ''ScoreMode

instance Aeson.ToJSON a => Aeson.ToJSON (Map.Map ScoreMode a) where
  toJSON = Aeson.toJSON . Map.fromList . (map (_1 %~ show)) . Map.toList


instance Aeson.FromJSON a => Aeson.FromJSON (Map.Map ScoreMode a) where
  parseJSON = fmap go . Aeson.parseJSON
    where
      go :: Map.Map String a -> Map.Map ScoreMode a  
      go = Map.fromList . (map (_1 %~ read)) . Map.toList



type ScoreMap = Map.Map ScoreMode Double

type BinaryPredictorScore =  [(Bool, Bool)] -> Double

evalScore :: ScoreMode -> BinaryPredictorScore
evalScore mode arg = 
  case mode of
    HeidkeSkillScore -> hss
    TrueSkillStatistic -> tss
    
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
      
      hss = 2*(nTP*nTN - nFN*nFP)/
            ((nTP+nFN)*(nFN+nTN) + (nTP+nFP)*(nFP+nTN))

      tss = nTP/(nTP+nFN) - nFP/(nFP+nTN)




-- | Returns the pair of the maximum found and the threshold

searchThreshold :: [(Double,Double)] -> BinaryPredictorScore -> Double -> (Double, Double)
searchThreshold tbl score thre0 = maximum $
    [(scoreOf t1, t1) | t1 <- thres]
  where
    scoreOf thre1 = score [(xp > thre1, xo > thre0) | (xp, xo) <- tbl]

    thres = map ((+thre0) . (/50) . fromInteger) [-100 .. 100]

makeScoreMap :: [(Double,Double)] -> Double -> ScoreMap
makeScoreMap tbl thre0 = Map.fromList
  [(mode, fst $ searchThreshold tbl (evalScore mode) thre0) | mode <- [HeidkeSkillScore, TrueSkillStatistic]]