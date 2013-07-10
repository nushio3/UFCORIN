module Main where

import Bindings.Gsl.WaveletTransforms
import Control.Applicative
import Control.Monad
import qualified Data.ByteString as BS
import Foreign.Marshal.Alloc (mallocBytes)
import Foreign.Marshal.Utils (new)
import Foreign.Ptr (plusPtr, castPtr, Ptr(..))
import Foreign.Storable (peek, poke, sizeOf)
import Foreign.C.String (peekCString)
import Foreign.C.Types (CDouble, CChar)

main :: IO ()
main = do
  putStrLn "hi"
  origData <- BS.readFile "resource/sample.fits"
  BS.useAsCString origData analyze
  
analyze origDataStr = do
  let n :: Int
      n = 4096
      sizeOfHeader :: Int
      sizeOfHeader = 8640
      pOrigData :: Ptr CDouble
      pOrigData = castPtr $ plusPtr origDataStr sizeOfHeader
  
  forM_ [0..(n*n-1)] $ \i -> do
    let pHead :: Ptr CDouble
        pHead = plusPtr pOrigData (i * sizeOf (0::CDouble))
        pHeadW :: Ptr CChar
        pHeadW = castPtr pHead
        pBytes = map (plusPtr pHeadW) [0..3]
    byteDatas <- mapM peek pBytes
    zipWithM_ poke pBytes $ reverse (map (*0) byteDatas :: [CChar])
    
    
    x <-peek pHead
    when (x > 500) $ print x
  
  
  pData <- mallocBytes (sizeOf (0::CDouble) * n)
  pAbsCoeff <- mallocBytes (sizeOf (0::CDouble) * n)
  pWavDau <- peek p'gsl_wavelet_haar

  pWork <- c'gsl_wavelet_workspace_alloc (fromIntegral n)

  pWavelet <- c'gsl_wavelet_alloc pWavDau 2

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

