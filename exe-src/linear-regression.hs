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
import System.Random
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
  (_,forecastTarget) <- loadFile "forecast-goes-72.txt" 1 4 log10
  (_,backcast) <- loadFile "forecast-goes-24.txt" 1 2 log10

  featureFiles <- lines <$> readInteractiveCommand "ls wavelet-features/*.txt"

  features <- forM featureFiles $ \fn -> loadFile fn 2 4 id

  forever $ do
    i1 <- randomRIO (0,length features - 1)
    i2 <- randomRIO (0,length features - 1)
    i3 <- randomRIO (0,length features - 1)
    i4 <- randomRIO (0,length features - 1)
    let 
        (fn1, vx1) = features !! i1
        (fn2, vx2) = features !! i2
        (fn3, vx3) = features !! i3
        (fn4, vx4) = features !! i4

        f [a0, a1,b, a2,a3,a4] = norm (forecastTarget -
          (scale a0 (zipV (*) backcast (scale (1e-10 * abs a3) vx3 + unitVector) )
            + zipV (/) (scale (1e-10*a1) vx1 + scale b unitVector) (scale (1e-10 * abs a2) vx2 + unitVector)  )
            + (scale (1e-10 * abs a4) vx4))
        
        
        ppGoal :: [Double] -> String
        ppGoal [a0, a1,b,a2,a3,a4] = printf
          "\"<paste forecast-goes-24.txt %s \" u 2:( %e *  (%e * $20 + 1) *log10($3) + (%e * $10 + %e) / (%e * $15 + 1)) + %e*$25"
                        (unwords [fn1,fn2,fn3,fn4])  a0     (1e-10*a3)                 (1e-10*a1)     b      (1e-10*abs a2)  (1e-10*abs a4)    
 
        nm1 = vx1 .* vx1
        nm2 = vx2 .* vx2

        
    when (nm1 > 0 && nm2 > 0 && Map.size vx1 >= 360 && Map.size vx2 >= 360) $ do
      goal <- Opt.run $ (Opt.minimize f [1,1,0,1,1,1]) -- {Opt.scaling=Just [1e-10,1]}
      printf "%f\t%s\n" (f goal) (ppGoal goal)
    hFlush stdout

  return ()