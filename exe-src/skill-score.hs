#!/usr/bin/env runhaskell

import System.Environment
import Text.Printf

main :: IO ()
main = do
  (fnPredict: fnObserve: _) <- getArgs
  predStr <- readFile fnPredict
  obsvStr <- readFile fnObserve
  let preds = map (head . words) $ lines predStr
      obsvs = map (head . words) $ lines obsvStr
      
      count :: Bool -> Bool -> Double
      count bx by = 
        fromIntegral $
        length $ 
        filter (\(x,y) -> (x/="0")==bx && (y/="0")==by)$
        zip preds obsvs
      
      nTP = count True  True
      nFN = count False True            
      nFP = count True  False
      nTN = count False False
      
      hss = 2*(nTP*nTN - nFN*nFP)/
            ((nTP+nFN)*(nFN+nTN) + (nTP+nFP)*(nFP+nTN))
      tss = nTP/(nTP+nFN) - nFP/(nFP+nTN)
      
  printf "%f\t%f\n" nTP nFP
  printf "%f\t%f\n" nFN nTN
  printf "hss=%f\n" hss
  printf "tss=%f\n" tss  
  