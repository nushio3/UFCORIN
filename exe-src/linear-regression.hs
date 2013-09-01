{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeSynonymInstances #-}
import Control.Applicative
import Control.Monad
import Control.Spoon
import Data.Maybe
import qualified Data.Map as Map
import qualified Numeric.Optimization.Algorithms.CMAES as Opt
import System.IO
import System.Process
import Text.Printf
import SpaceWeather.Timeline



readInteractiveCommand :: String -> IO String
readInteractiveCommand cmd = do
  (_, stdout, _, _) <- runInteractiveCommand cmd
  hGetContents stdout

log10 x = log x / log 10

upperLimit = 720

targetRange :: [TimeBin]
targetRange = [0..upperLimit]

type Vector a = Map.Map TimeBin a

unitVector :: Num a => Vector a
unitVector = Map.fromList [(t,1)| t<-targetRange]

average :: Fractional a => Vector a -> a
average vx = sum xs / fromIntegral (length xs)
  where
    xs = Map.elems vx



zipV :: (a->b->c) -> Vector a -> Vector b -> Vector c
zipV f va vb = Map.fromList $ catMaybes [(t,) <$> (f <$> Map.lookup t va <*> Map.lookup t vb) |t <- targetRange]

infix 7 .* -- dot product

scale :: Num a => a -> Vector a -> Vector a
scale x = Map.map (*x)

instance Num a => Num (Vector a) where
  (+) = zipV (+)
  (-) = zipV (-)

(.*) :: Num a => Vector a ->  Vector a -> a
(.*) va vb = sum $ Map.elems $ zipV (*) va vb

norm :: Fractional a => Vector a -> a
norm v = v .* v
  
dist :: Floating a => Vector a -> a 
dist v = sqrt $ norm v

pprint :: Show a => Vector a -> IO ()
pprint v = do
  mapM_ putStrLn [printf "%d %s" t (show x)| t <- targetRange, x <- maybeToList $ Map.lookup t v]

loadFile :: FilePath -> Int -> Int -> (Double->Double) -> IO (String, Vector Double)
loadFile fn x y f = do
  con <- readFile fn
  let strs = lines con
      parse :: String -> Maybe (Integer, Double)
      parse str = join $ spoon $ do
        let ws = words str 
        guard $ (read $ ws!!x) <= upperLimit
        return $ (read $ ws!!x , f$read $ ws!!y)
  return $ (fn,) $ Map.fromList $ mapMaybe parse strs


main :: IO ()
main = do
  (_,forecastTarget) <- loadFile "forecast-goes-24.txt" 1 4 log10

  featureFiles <- lines <$> readInteractiveCommand "ls wavelet-features/*.txt"

  features <- forM featureFiles $ \fn -> loadFile fn 2 4 id

  forM features $ \ (fn, vx) -> do
    let 
        f [a,b] = norm (forecastTarget - (scale b unitVector + scale a vx))
        nm = vx .* vx
    guard $ nm > 0

    goal <- Opt.run $ Opt.minimize f [1,0]

    printf "%f\t%s\t%s\n" (f goal) fn (show goal)
  hFlush stdout

  return ()