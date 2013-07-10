module Main where

import Bindings.Gsl.WaveletTransforms
import Control.Applicative
import Control.Monad
import Foreign.Marshal.Alloc (mallocBytes)
import Foreign.Marshal.Utils (new)
import Foreign.Storable (peek, poke, sizeOf)
import Foreign.Ptr (plusPtr)
import Foreign.C.String (peekCString)
import Foreign.C.Types (CDouble)

main :: IO ()
main = do
  putStrLn "hi"
  
  let n :: Int
      n = 256
  
  pData <- mallocBytes (sizeOf (0::CDouble) * n)
  pAbsCoeff <- mallocBytes (sizeOf (0::CDouble) * n)
  pWavDau <- peek p'gsl_wavelet_daubechies

  pWork <- c'gsl_wavelet_workspace_alloc (fromIntegral n)

  pWavelet <- c'gsl_wavelet_alloc pWavDau 4

  nam <- peekCString =<< c'gsl_wavelet_name pWavelet
  print nam
  
  forM_ [0..(n-1)] $ \i -> 
    poke (plusPtr pData (i * sizeOf (0::CDouble)))
         (fromIntegral i :: CDouble) 
          
  flag <- c'gsl_wavelet_transform_forward pWavelet pData 
          1 (fromIntegral n) pWork

  forM_ [0..(n-1)] $ \i -> do
    x <- peek (plusPtr pData (i * sizeOf (0::CDouble)))
    print (x :: CDouble)     

  print flag

