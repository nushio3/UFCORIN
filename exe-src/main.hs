{-# LANGUAGE OverloadedStrings #-}
module Main where

import Bindings.Gsl.WaveletTransforms
import Control.Applicative
import Control.Monad
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Repr.ForeignPtr as R
import qualified Data.Array.Repa.IO.BMP as R
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BSL
import Data.Binary (get)
import Data.Binary.Get (runGet)
import Data.Int(Int32)
import Data.List(isSuffixOf)
import Data.Monoid ((<>))
import Data.Word(Word8,Word32)
import qualified Data.Text as Text
import qualified Data.Text.IO as Text
import Data.String.Utils (replace)
import Foreign.Marshal.Alloc (mallocBytes, free)
import Foreign.Marshal.Utils (new)
import Foreign.Ptr (plusPtr, castPtr, Ptr(..))
import Foreign.ForeignPtr.Safe (newForeignPtr_)
import Foreign.Storable (peek, poke, sizeOf, peekElemOff, pokeElemOff)
import Foreign.C.String (peekCString)
import Foreign.C.Types (CDouble, CFloat, CChar)
import System.Endian (fromBE32, fromLE32)
import System.FilePath
import System.Posix.Process (getProcessID)
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
  [("haarC", p'gsl_wavelet_haar_centered , 2)]
  -- [
  --   ("haar0", p'gsl_wavelet_haar , 2)
  -- , ("haarC", p'gsl_wavelet_haar_centered , 2)
  -- , ("daub0", p'gsl_wavelet_daubechies , 4)
  -- , ("daubC", p'gsl_wavelet_daubechies_centered , 4)
  -- , ("bspl0", p'gsl_wavelet_bspline , 103)
  -- , ("bsplC", p'gsl_wavelet_bspline_centered ,103 )
  -- , ("daub0", p'gsl_wavelet_daubechies , 20)
  -- , ("daubC", p'gsl_wavelet_daubechies_centered , 20)
  -- , ("bspl0", p'gsl_wavelet_bspline , 309)
  -- , ("bsplC", p'gsl_wavelet_bspline_centered , 309 )
  -- ]


-- sourceFns :: [FilePath]
-- sourceFns = ["/user/shibayama/sdo/hmi/2011/02/28/20.fits", "/user/shibayama/sdo/hmi/2011/01/31/10.fits"]

main :: IO ()
main = do
  input <- getContents
  let sourceFns = filter (isSuffixOf ".fits") $ words input
  sequence_ $ testWavelet <$> [False,True] <*> wavelets <*> sourceFns


testWavelet :: Bool -> (String, Wavelet, Int)   -> FilePath   -> IO ()
testWavelet    isStd   (wlabel, wptr   , waveletK) sourcePath = do

  myUnixPid <- getProcessID

  let fnBase :: String
      fnBase = printf "%s-%s-%s-%d"
               sourceFnBody
               (if isStd then "S" else "N" :: String) wlabel waveletK
               
      (sourceDir, sourceFn) = splitFileName sourcePath
      (sourceFnBody, _) = splitExtension sourceFn
      
      localFitsFn :: String
      localBmpFn :: String
      localFitsFn = printf "/tmp/%s.fits" (show myUnixPid)
      localBmpFn = printf "/tmp/%s.bmp" (show myUnixPid)
      
      destinationFn = (replace "/shibayama/" "/nushio/" sourceDir) </> (fnBase++".bmp")
                      
      (destinationDir,_) = splitFileName destinationFn

  system $ printf "rm -f %s" localFitsFn
  system $ printf "hadoop fs -get %s %s" sourcePath localFitsFn

  origData <- BS.readFile localFitsFn

  -- prepare wavelet transformation
  let n :: Int
      n = 1024

  pWavDau <- peek wptr

  pData <- mallocBytes (sizeOf (0::CDouble) * n * n)
  pWavelet <- c'gsl_wavelet_alloc pWavDau (fromIntegral waveletK)
  pWork <- c'gsl_wavelet_workspace_alloc (fromIntegral $ n*n)

  fgnPData <- newForeignPtr_ pData

  let finalizer = do
        c'gsl_wavelet_free pWavelet
        c'gsl_wavelet_workspace_free pWork
        free pData


  nam <- peekCString =<< c'gsl_wavelet_name pWavelet

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
         1
         pWork

  let bmpShape = R.ix2 n n

  let toRGB :: Double -> (Word8,Word8,Word8)
      toRGB x = 
        let red   = rb (x/10)
            green = min red blue
            blue  = rb (negate $ x/10)
        in (red,green,blue)

      rb :: Double -> Word8
      rb = round . min 255 . max 0 . (255-)

  bmpData <- R.computeUnboxedP $ R.map (toRGB . realToFrac) $ R.fromForeignPtr bmpShape fgnPData

  R.writeImageToBMP localBmpFn  bmpData
  system $ printf "hadoop fs -mkdir -p %s" destinationDir
  system $ printf "hadoop fs -rm -f -skipTrash %s" destinationFn
  system $ printf "hadoop fs -put %s %s" localBmpFn destinationFn

  finalizer

  printf "%s:%s\t%s\n" sourcePath nam (show ret)
