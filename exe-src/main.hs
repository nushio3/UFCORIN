module Main where

import Bindings.Gsl.WaveletTransforms
import Foreign.Marshal.Alloc (mallocBytes)
import Foreign.Marshal.Utils (new)
import Foreign.Storable (peek, sizeOf)
import Foreign.C.String (peekCString)
import Foreign.C.Types (CDouble)

main :: IO ()
main = do
  putStrLn "hi"
  
  let n :: Int
      n = 256
  
  ptrData <- mallocBytes (sizeOf (0::CDouble) * n)
  
  pgwDau <- peek p'gsl_wavelet_daubechies

  ret <- c'gsl_wavelet_alloc pgwDau 4

  nam <- peekCString =<< c'gsl_wavelet_name ret

  print nam