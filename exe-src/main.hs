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

main :: IO ()
main = do
  putStrLn "hi"
  origData <- BS.readFile "resource/sample.fits"

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
    pokeElemOff pData i (42e99 :: CDouble)

  let nBig :: Int
      nBig = 4096
      sizeOfHeader :: Int
      sizeOfHeader = 8640

  forM_ [0..(nBig-1)] $ \iy -> do
    forM_ [0..(nBig-1)] $ \ix -> do
      let
        addr0 :: Int
        addr0 = sizeOfHeader + 4* (iy*nBig + ix)
        binVal :: BSL.ByteString
        binVal = BSL.pack
          [ BS.index origData (fromIntegral $ addr0+di) | di <- [0,1,2,3]]

        
        val' :: Int32
        val' = runGet get binVal


        tobas = 1000000

        val3, val :: CDouble
        val3 =  fromIntegral $ val'

        val
          | abs(val3) > tobas = -tobas
          | otherwise         = val3
                                
                                

        ix' = ix `div` (nBig `div`n)
        iy' = iy `div` (nBig `div`n)
      pokeElemOff pData (iy'*n+ix') val

  bitmapText <- fmap Text.concat $ forM [0..(n-1)] $ \iy -> do
    lineText <- fmap Text.concat $ forM [0..(n-1)] $ \ix -> do
      val <- peekElemOff pData (iy*n + ix)
      return $ Text.pack $ printf "%i %i %s\n" ix iy (show (val :: CDouble))
    return $ lineText <> "\n"

  Text.writeFile "test.txt" bitmapText

  flag <- c'gsl_wavelet_transform_forward pWavelet pData
          1 (fromIntegral $ n*n) pWork


  print flag
