{-# LANGUAGE TemplateHaskell #-}
module System.IO.Hadoop where

import Control.Lens
import qualified Data.Text as T
import qualified Data.Text.IO as T
import System.Process
import Text.Printf

data FileSystem = LocalFS | HadoopFS

data FSOption = FSOPtion
  { _targetFS :: FileSystem
  }
makeClassy ''FSOption

readFile :: FilePath -> IO T.Text 
readFile fp = do
  (_,Just hout,_,hproc) <- createProcess (shell $ printf "hadoop fs -cat %s" fp){std_out = CreatePipe}
  ret <- T.hGetContents hout
  _ <- waitForProcess hproc
  return ret

writeFile :: FilePath -> T.Text -> IO ()
writeFile fp txt = do
  (Just hin,_,_,hproc) <- createProcess (shell $ printf "hadoop fs -put -f - %s" fp){std_in = CreatePipe}
  T.hPutStr hin txt
  _ <- waitForProcess hproc
  return ()
