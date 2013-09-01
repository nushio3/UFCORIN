module SpaceWeather.Text where

import qualified Data.Text as T
import Safe(readMay, atMay)

showT :: Show a => a -> T.Text
showT = T.pack . show

readT :: Read a => T.Text -> a
readT = read . T.unpack

readMayT :: Read a => T.Text -> Maybe a
readMayT = readMay . T.unpack

readAt :: Read a => [T.Text] -> Int -> Maybe a
readAt xs n = atMay xs n >>= readMayT
