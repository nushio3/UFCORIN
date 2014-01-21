{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.FeaturePack where

import Control.Lens
import Control.Monad.Trans.Either
import Control.Monad.IO.Class
import qualified Data.Aeson.TH as Aeson
import qualified Data.ByteString.Char8 as BS
import Data.Char
import qualified Data.Text as Text
import qualified Data.Yaml as Yaml
import           Test.QuickCheck.Arbitrary

import SpaceWeather.Feature
import SpaceWeather.Format

newtype FeaturePack
  = FeaturePack  [(Double, Feature)] 
makeWrapped ''FeaturePack
makeClassy  ''FeaturePack

newtype FeaturePackFile 
  = FeaturePackFile  [(Double, FilePath)] 



makeWrapped ''FeaturePackFile
makeClassy  ''FeaturePackFile
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 4, Aeson.constructorTagModifier = map toLower} ''FeaturePackFile
instance Format FeaturePackFile where
  encode = Text.pack . BS.unpack . Yaml.encode
  decode = Yaml.decodeEither . BS.pack . Text.unpack

loadFeaturePackFile :: FeaturePackFile -> IO (Either String FeaturePack)
loadFeaturePackFile fpf = do 
  let map0 = view unwrapped fpf
  map1 <- runEitherT $ mapM go map0 
  return $ fmap FeaturePack map1  

  where
    go :: (Double, FilePath) -> EitherT String IO (Double,Feature)
    go (w,fp) = do
      econ <- liftIO $ decodeFile fp
      hoistEither $ fmap (w,) econ
