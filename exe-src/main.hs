{-# LANGUAGE OverloadedStrings #-}
module Main where

import Bindings.Gsl.WaveletTransforms
import Control.Applicative
import Control.Monad
import qualified Data.ByteString as BS
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
import System.Endian (toBE32, toLE32)
import Text.Printf
import Unsafe.Coerce (unsafeCoerce)

main :: IO ()
main = do
  putStrLn "hi"
  origData <- BS.readFile "resource/sample.fits"
  BS.useAsCString origData analyze

analyze origDataStr = do


  let n :: Int
      n = 512
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
      pOrigData :: Ptr Int32
      pOrigData = castPtr $ origDataStr `plusPtr` sizeOfHeader

  forM_ [0..(nBig-1)] $ \iy -> do
    forM_ [0..(nBig-1)] $ \ix -> do
      val'' <- peekElemOff pOrigData (iy*nBig + ix)
      let
--         val' :: Int32
-- --        val' = unsafeCoerce . toBE32 . unsafeCoerce $ val''
--         val' = unsafeCoerce . toBE32 .unsafeCoerce $ val''
--

        val :: CDouble
        val
          | abs(val'') < 1000000 = read . show $ val''
          | otherwise   = 0
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
