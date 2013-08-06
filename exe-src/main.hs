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
import System.Process
import Text.Printf
import Unsafe.Coerce (unsafeCoerce)

sq x = x*x

gnuplot :: [String] -> IO ()
gnuplot cmds = do
  writeFile "tmp.gnuplot" $ unlines cmds
  system "gnuplot tmp.gnuplot"
  return ()

type Wavelet = Ptr (Ptr C'gsl_wavelet_type)

wavelets :: [(String, Wavelet, Int)]
wavelets = 
  [ ("bspl0", p'gsl_wavelet_bspline , 309)
  , ("bsplC", p'gsl_wavelet_bspline_centered , 309 ) 
  , ("daub0", p'gsl_wavelet_daubechies , 20)
  , ("daubC", p'gsl_wavelet_daubechies_centered , 20)
  , ("bspl0", p'gsl_wavelet_bspline , 103)
  , ("bsplC", p'gsl_wavelet_bspline_centered ,103 ) 
  , ("daub0", p'gsl_wavelet_daubechies , 4)
  , ("daubC", p'gsl_wavelet_daubechies_centered , 4)
  , ("haar0", p'gsl_wavelet_haar , 2)
  , ("haarC", p'gsl_wavelet_haar_centered , 2) ]


main :: IO ()
main = sequence_ $ testWavelet <$> [True,False] <*> wavelets <*> [2] -- ,1,2]

testWavelet :: Bool -> (String, Wavelet, Int)   -> Int   -> IO ()
testWavelet    isStd   (wlabel, wptr   , waveletK) dataId = do
  let fnBase :: String
      fnBase = printf "%s-%s-%d-DS%d" 
               (if isStd then "S" else "N" :: String) wlabel waveletK
               dataId 
               
      fnFitsBody :: String
      fnFitsBody = printf "dummyspot%d" dataId
      
      fnFwdTxt, fnBwdTxt, fnFwdPng, fnBwdPng :: String
      fnFwdTxt = printf "dist/fwd-%s.txt" fnBase 
      fnBwdTxt = printf "dist/bwd-%s.txt" fnBase 
      fnFwdPng = printf "dist/fwd-%s.png" fnBase 
      fnBwdPng = printf "dist/bwd-%s.png" fnBase 
  printf "%s, " fnBase       
  
  origData <- BS.readFile $ printf "resource/%s.fits" fnFitsBody

  -- prepare wavelet transformation
  let n :: Int
      n = 1024
  pData <- mallocBytes (sizeOf (0::CDouble) * n * n)
  pAbsCoeff <- mallocBytes (sizeOf (0::CDouble) * n * n)
  pWavDau <- peek wptr

  pWork <- c'gsl_wavelet_workspace_alloc (fromIntegral $ n*n)

  pWavelet <- c'gsl_wavelet_alloc pWavDau (fromIntegral waveletK)

  nam <- peekCString =<< c'gsl_wavelet_name pWavelet
  printf "%s\n" nam

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
        val3 = fromIntegral $ val'

        val
          | sq(ix-rOfInput) + sq(iy-rOfInput) > sq 1792 = 0
          | otherwise         =  val3



        ix' = ix `div` (nOfInput `div`n)
        iy' = iy `div` (nOfInput `div`n)
      pokeElemOff pData (iy'*n+ix') val


  let wavelet2d 
       | isStd     = c'gsl_wavelet2d_transform
       | otherwise = c'gsl_wavelet2d_nstransform

  -- perform wavelet transformation.
  ret <- wavelet2d pWavelet pData
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

  Text.writeFile fnFwdTxt bitmapText


  -- perform inverse transformation.
  ret <- wavelet2d pWavelet pData
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

  Text.writeFile fnBwdTxt bitmapText

  gnuplot
    [ "set term png size 1800,1600"
    , printf "set out '%s'" fnFwdPng
    , "set pm3d"
    , "set pm3d map"
    , "set xrange [0:256]"
    , "set yrange [0:256]"
    , "set cbrange [-5000:5000]"
    , "set palette define (-5000 'blue', 0 'white', 5000 'red')"
    , "set size ratio -1"
    , printf "splot '%s'" fnFwdTxt ]



