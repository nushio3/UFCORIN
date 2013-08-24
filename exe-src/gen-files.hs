import Control.Monad
import System.Process
import Text.Printf

main :: IO ()
main = do
  forM_ [2011,2012 :: Int] $ \year -> do
    forM_ [1..12 :: Int] $ \mo -> do
      forM_ [0..9 :: Int] $ \d1 -> do
        let fn :: String
            fn = printf "%d-%02d-x%d.txt" year mo d1
        system $ printf "hadoop fs -ls /user/shibayama/sdo/hmi/%04d/%02d/*%01d > filelist/%s" year mo d1 fn
