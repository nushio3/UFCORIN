{-# LANGUAGE OverloadedStrings #-}

import Control.Monad
import Control.Spoon(spoon)
import Data.Maybe
import qualified Data.Text as T
import qualified Data.Text.IO as T

import SpaceWeather.Text

main :: IO ()
main = do
  con <- T.getContents
  mapM_ T.putStrLn $ mapMaybe processLine $ T.lines con


processLine :: T.Text -> Maybe T.Text
processLine str = spoon $ T.intercalate "\t" [newKeyStr,newValStr]
  where
    [keyStr,valStr] = T.splitOn "\t" str
    [coloned0, coordStr, funcStr] = T.splitOn ":" keyStr
    [ymdhStr,waveletStr,kStr,nsStr] = T.splitOn "-" coloned0
    [year,month,day,hour] = T.splitOn "/" ymdhStr

    px :: Int
    py :: Int
    (px,py) = readT coordStr

    newYmdStr = T.intercalate "-" [year,month,day] 

    newFnStr = T.intercalate "-" [waveletStr,kStr,nsStr,showT px,showT py]
    newKeyStr = T.unwords [newFnStr, newYmdStr, hour]
    newValStr = T.unwords [funcStr, valStr]