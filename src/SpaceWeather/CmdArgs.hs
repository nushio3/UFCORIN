module SpaceWeather.CmdArgs where

import System.Process
import System.Posix.Process
import System.IO
import System.IO.Unsafe

{-# NOINLINE workDir #-}
workDir :: FilePath
workDir = unsafePerformIO $ do
  pid <- getProcessID
  let ret = "/tmp/spaceweather-" ++ show pid
  system $ "mkdir -p " ++ ret
  hPutStrLn stderr $ "using workdir: " ++ ret
  return ret
