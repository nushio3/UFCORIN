{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TypeSynonymInstances #-}
module SpaceWeather.FeaturePack where

import Control.Lens
import qualified Data.Aeson.TH as Aeson
import qualified Data.ByteString.Char8 as BS
import Data.Char
import qualified Data.Text as Text
import qualified Data.Yaml as Yaml

import SpaceWeather.Format

newtype FeaturePackFile 
  = FeaturePackFile  [(Double, FilePath)] 

makeWrapped ''FeaturePackFile
makeClassy  ''FeaturePackFile
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 4, Aeson.constructorTagModifier = map toLower} ''FeaturePackFile
instance Format FeaturePackFile where
  encode = Text.pack . BS.unpack . Yaml.encode
  decode = Yaml.decodeEither . BS.pack . Text.unpack