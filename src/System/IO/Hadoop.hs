{-# LANGUAGE TemplateHaskell #-}
module System.IO.Hadoop where

import Control.Lens
import qualified Data.Text as T
import qualified Data.Text.IO as T

data FileSystem = LocalFS | HadoopFS

data FSOption = FSOPtion
  { _targetFS :: FileSystem
  }

makeClassy ''FSOption

writeHadoopFS :: FilePath -> T.Text -> IO ()
writeHadoopFS fp txt = do
  return ()
