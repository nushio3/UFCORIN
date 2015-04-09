import Control.Lens
import Control.Monad
import Data.Function (on)
import Data.List (sortBy)
import System.Random

randSort :: Int -> StdGen -> [a] -> [a]
randSort width gen0 xs = map snd $ sortBy (compare `on` fst) $ zip rands xs
  where
    rands :: [Double]
    rands = zipWith (+) [0..] $ map (mix.fst) $ drop 1 $ iterate f (0,gen0)

    f (_,g) = randomR (0,width) g

    -- given integer between (0,width), return double that does not exceed width+1
    mix :: Int -> Double
    mix x = (1 + 1 / (1 + fromIntegral width)) * fromIntegral x

main :: IO ()
main = do
  seed <- randomIO
  let gen = mkStdGen seed
      xs = zip [1..] ['a'..'z']
  print $ xs & partsOf (each._2) %~ (randSort 1 gen)
  replicateM_ 100 $ do
    seed <- randomIO
    putStrLn $ randSort 10 (mkStdGen seed) (replicate 30 ' ' ++ "#" ++ replicate 30 '.')
