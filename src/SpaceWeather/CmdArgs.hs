module SpaceWeather.CmdArgs where

import Control.Exception (bracket_)
import Data.Maybe
import Data.List
import System.Process
import System.Posix.Process
import System.IO
import System.IO.Unsafe
import System.Environment
import Safe (readMay)

{-# NOINLINE workDir #-}
workDir :: FilePath
workDir = unsafePerformIO $ do
  pid <- getProcessID
  let ret = "/tmp/spaceweather-" ++ show pid
  system $ "mkdir -p " ++ ret
  hPutStrLn stderr $ "using workdir: " ++ ret
  return ret

withWorkDir = bracket_
  (system $ "mkdir -p " ++ workDir)
  (system $ "rm -fr " ++ workDir)

withWorkDirOf wd = bracket_
  (system $ "mkdir -p " ++ wd)
  (system $ "rm -fr " ++ wd)

spacialNoise :: Double
spacialNoise = fromMaybe 0 $ listToMaybe $ catMaybes $ map mkCand unsafeArgv
  where
    mkCand str = case splitAt 4 str of
      ("--sn", rest) -> readMay rest
      _              -> Nothing

temporalNoise :: Double
temporalNoise = fromMaybe 0 $ listToMaybe $ catMaybes $ map mkCand unsafeArgv
  where
    mkCand str = case splitAt 4 str of
      ("--tn", rest) -> readMay rest
      _              -> Nothing

crossValidationNoise :: Bool
crossValidationNoise = fromMaybe False $ listToMaybe $ catMaybes $ map mkCand unsafeArgv
  where
    mkCand str = case str of
      "--cvn" -> Just True
      _       -> Nothing


{-# NOINLINE unsafeArgv #-}
unsafeArgv :: [String]
unsafeArgv = unsafePerformIO getArgs
