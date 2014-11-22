{-# LANGUAGE TemplateHaskell #-}
module System.IO.Hadoop where

import Control.Lens
import Data.List (isPrefixOf)
import qualified Data.Map as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import System.IO
import System.IO.Unsafe
import System.Process
import Text.Printf
import Data.IORef

data FileSystem = LocalFS | HadoopFS

data FSOption = FSOPtion
  { _targetFS :: FileSystem
  }
makeClassy ''FSOption

{-# NOINLINE cacheRef #-}
cacheRef :: IORef (M.Map FilePath T.Text)
cacheRef = unsafePerformIO $ newIORef M.empty


memcached :: (FilePath -> IO T.Text) -> FilePath -> IO T.Text 
memcached prog input = do
  cache <- readIORef cacheRef
  case M.lookup input cache of
    Nothing -> do
      resp <- prog input
      modifyIORef cacheRef $ M.insert input resp
      return resp
    Just resp -> return resp


readFile :: FilePath -> IO T.Text 
readFile = memcached readFile'

readFile' :: FilePath -> IO T.Text 
readFile' fp = do
  (_,Just hout,_,hproc) <- createProcess (shell $ printf "aws s3 cp --region us-west-2 %s -" (compatibleFilePath fp)){std_out = CreatePipe}
  ret <- T.hGetContents hout
  _ <- waitForProcess hproc
  return ret

writeFile :: FilePath -> T.Text -> IO ()
writeFile fp txt = do
  (Just hin,_,_,hproc) <- createProcess (shell $ printf "aws s3 cp --region us-west-2 - %s" (compatibleFilePath fp)){std_in = CreatePipe}
  T.hPutStr hin txt
  hClose hin
  _ <- waitForProcess hproc
  return ()


compatibleFilePath :: FilePath -> FilePath
compatibleFilePath fp
  | "s3://" `isPrefixOf` fp = fp
  | otherwise               = "s3://bbt.hdfs.backup" ++ fp
