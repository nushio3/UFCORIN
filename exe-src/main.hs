module Main where

import Bindings.Gsl.WaveletTransforms
import Foreign.Marshal.Utils (new)
import Foreign.Storable (peek)
import Foreign.C.String (peekCString)

main :: IO ()
main = do
  putStrLn "hi"
  pgwDau <- peek p'gsl_wavelet_daubechies

  ret <- c'gsl_wavelet_alloc pgwDau 4

  nam <- peekCString =<< c'gsl_wavelet_name ret

  print nam