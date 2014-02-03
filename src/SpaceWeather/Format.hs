{-# LANGUAGE OverloadedStrings #-}
module SpaceWeather.Format where

import Control.Lens
import Data.Char
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Safe(readMay, atMay)

class Format a where
  encode :: a -> T.Text

  decode :: T.Text -> Either String a

  encodeFile :: FilePath -> a -> IO ()
  encodeFile fp x = T.writeFile fp $ encode x

  decodeFile :: FilePath -> IO (Either String a)
  decodeFile fp = do
    txt <- T.readFile fp
    return $ case decode txt of
      Left err -> Left $ fp ++ ":" ++ err
      Right x -> Right x

-- utility functions to parse & ppt text


showT :: Show a => a -> T.Text
showT = T.pack . show

readT :: Read a => T.Text -> a
readT = read . T.unpack

readMayT :: Read a => T.Text -> Maybe a
readMayT = readMay . T.unpack

readMaySubT :: Read a => Int -> Int -> T.Text -> Maybe a
readMaySubT start len = readMay .  T.unpack . T.take len . T.drop start 

readAt :: Read a => [T.Text] -> Int -> Maybe a
readAt xs n = atMay xs n >>= readMayT

linesWithComment :: T.Text -> [(Int, T.Text)]
linesWithComment = filter (not.isComment.(^._2)) . zip [1..] . T.lines 
  where
    isComment :: T.Text -> Bool
    isComment = T.isPrefixOf "#" . T.dropWhile isSpace 