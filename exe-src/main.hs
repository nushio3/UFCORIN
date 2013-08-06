{-# LANGUAGE OverloadedStrings #-}
module Main where

import Bindings.Gsl.WaveletTransforms
import Control.Applicative
import Control.Monad
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BSL
import Data.Binary (get)
import Data.Binary.Get (runGet)
import Data.Monoid ((<>))
import Data.Word(Word32)
import Data.Int(Int32)
import qualified Data.Text as Text
import qualified Data.Text.IO as Text
import Foreign.Marshal.Alloc (mallocBytes)
import Foreign.Marshal.Utils (new)
import Foreign.Ptr (plusPtr, castPtr, Ptr(..))
import Foreign.Storable (peek, poke, sizeOf, peekElemOff, pokeElemOff)
import Foreign.C.String (peekCString)
import Foreign.C.Types (CDouble, CFloat, CChar)
import System.Endian (fromBE32, fromLE32)
import Text.Printf
import Unsafe.Coerce (unsafeCoerce)

sq x = x*x

main :: IO ()
main = do
  putStrLn "hi"
  origData <- BS.readFile "resource/sample.fits"

  -- prepare wavelet transformation
  let n :: Int
      n = 1024
  pData <- mallocBytes (sizeOf (0::CDouble) * n * n)
  pAbsCoeff <- mallocBytes (sizeOf (0::CDouble) * n * n)
  pWavDau <- peek p'gsl_wavelet_haar

  pWork <- c'gsl_wavelet_workspace_alloc (fromIntegral $ n*n)

  pWavelet <- c'gsl_wavelet_alloc pWavDau 2

  nam <- peekCString =<< c'gsl_wavelet_name pWavelet
  print nam

  -- zero initialize the data
  forM_ [0..n*n-1] $ \i ->
    pokeElemOff pData i (0 :: CDouble)

  -- read the fits file and shrink it
  let nOfInput :: Int
      nOfInput = 4096

      rOfInput = nOfInput `div` 2
      sizeOfHeader :: Int
      sizeOfHeader = 8640

  forM_ [0..(nOfInput-1)] $ \iy -> do
    forM_ [0..(nOfInput-1)] $ \ix -> do
      let
        addr0 :: Int
        addr0 = sizeOfHeader + 4* (iy*nOfInput + ix)
        binVal :: BSL.ByteString
        binVal = BSL.pack
          [ BS.index origData (fromIntegral $ addr0+di) | di <- [0,1,2,3]]


        val' :: Int32
        val' = runGet get binVal


        tobas = 1000000

        val3, val :: CDouble
        val3 =  fromIntegral $ val'

        val
          | sq(ix-rOfInput) + sq(iy-rOfInput) > sq 1792 = 0
          | otherwise         =  val3



        ix' = ix `div` (nOfInput `div`n)
        iy' = iy `div` (nOfInput `div`n)
      pokeElemOff pData (iy'*n+ix') val


  -- perform wavelet transformation.
  ret <- c'gsl_wavelet2d_nstransform pWavelet pData
         (fromIntegral n) (fromIntegral n) (fromIntegral n)
         (fromIntegral 1)
         pWork
  print ret

  -- write out the text for use in gnuplot.
  bitmapText <- fmap Text.concat $ forM [0..(n-1)] $ \iy -> do
    lineText <- fmap Text.concat $ forM [0..(n-1)] $ \ix -> do
      val <- peekElemOff pData (iy*n + ix)
      return $ Text.pack $ printf "%i %i %s\n" ix iy (show (val :: CDouble))
    return $ lineText <> "\n"

  Text.writeFile "test-wavelet.txt" bitmapText


  -- perform inverse transformation.
  ret <- c'gsl_wavelet2d_nstransform pWavelet pData
         (fromIntegral n) (fromIntegral n) (fromIntegral n)
         (fromIntegral (-1))
         pWork
  print ret



  -- write out the text for use in gnuplot.
  bitmapText <- fmap Text.concat $ forM [0..(n-1)] $ \iy -> do
    lineText <- fmap Text.concat $ forM [0..(n-1)] $ \ix -> do
      val <- peekElemOff pData (iy*n + ix)
      return $ Text.pack $ printf "%i %i %s\n" ix iy (show (val :: CDouble))
    return $ lineText <> "\n"

  Text.writeFile "test-back.txt" bitmapText
