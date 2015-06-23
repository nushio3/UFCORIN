{-# LANGUAGE TemplateHaskell #-}
module System.IO.Hadoop where

import Control.Lens
import Data.List (isPrefixOf)
import qualified Data.Map as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import System.Directory(doesFileExist)
import System.IO
import System.IO.Unsafe
import System.Process
import Text.Printf
import Data.IORef
import Network.HTTP.Base(urlEncode)
import System.Random

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
      let cacheFn = "/tmp/" ++ urlEncode input
      e <- doesFileExist cacheFn
      case e of
        True -> T.readFile cacheFn
        False -> do
          resp <- prog input
          modifyIORef cacheRef $ M.insert input resp
          rand <- randomRIO (0,(2::Integer)^256)
          let randomFn = cacheFn ++ show rand
          T.writeFile randomFn resp
          system $ printf "mv %s %s" randomFn cacheFn
          return resp
    Just resp -> return resp


readFile :: FilePath -> IO T.Text
readFile fp =
  case localProtocol `isPrefixOf` fp of
   True  -> T.readFile $ drop (length localProtocol) fp
   False -> memcached readFile' fp

readFile' :: FilePath -> IO T.Text
readFile' fp = do
  let cmd =  printf "aws s3 cp --region ap-northeast-1 %s -" (compatibleFilePath fp)
  putStrLn cmd
  (_,Just hout,_,hproc) <- createProcess (shell cmd){std_out = CreatePipe}
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
  | "s3://" `isPrefixOf` fp       = fp
  | localProtocol `isPrefixOf` fp = drop (length localProtocol) fp
  | otherwise                     = "s3://bbt.hdfs.backup" ++ fp


localProtocol :: FilePath
localProtocol = "file://"
