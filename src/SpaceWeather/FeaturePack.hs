{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeSynonymInstances #-}
module SpaceWeather.FeaturePack where

import Control.Lens
import Control.Monad.Trans.Either
import Control.Monad.IO.Class
import qualified Data.Aeson.TH as Aeson
import qualified Data.ByteString.Char8 as BS
import Data.Char
import qualified Data.Map as Map
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Yaml as Yaml
import           Test.QuickCheck.Arbitrary
import           Text.Printf

import SpaceWeather.Feature
import SpaceWeather.Format
import SpaceWeather.TimeLine

data FeatureSchema 
  = FeatureSchema 
  { _colX           :: Int
  , _colY           :: Int
  , _weight         :: Double
  , _isLog          :: Bool
  , _schemaFilename :: FilePath
  }
makeClassy ''FeatureSchema
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''FeatureSchema

newtype FeaturePack
  = FeaturePack [Feature] 
makeWrapped ''FeaturePack
makeClassy  ''FeaturePack

newtype FeatureSchemaPack
  = FeatureSchemaPack [FeatureSchema]

makeWrapped ''FeatureSchemaPack
makeClassy  ''FeatureSchemaPack
Aeson.deriveJSON Aeson.defaultOptions ''FeatureSchemaPack
instance Format FeatureSchemaPack where
  encode = T.pack . BS.unpack . Yaml.encode
  decode = Yaml.decodeEither . BS.pack . T.unpack

loadFeatureSchema :: FeatureSchema -> EitherT String IO Feature
loadFeatureSchema schema0 = do
  txt0 <- liftIO $ T.readFile $ schema0^.schemaFilename

  let 
      convert :: Double -> Double
      convert x = if schema0^.isLog then log x / log 10 else x


      parseLine :: (Int, T.Text) -> Either String (TimeBin, Double)
      parseLine (lineNum, txt) = 
        maybe (Left $ printf "parse error on line %d" lineNum) Right $ do
          -- maybe monad here
          let wtxt = T.words txt
          t <- readAt wtxt (schema0 ^. colX)
          a <- readAt wtxt (schema0 ^. colY)
          return (t,(schema0^.weight) * convert a)

      ret :: Either String Feature
      ret = fmap Map.fromList $ mapM parseLine $ linesWithComment txt0
  hoistEither ret


loadFeatureSchemaPack :: FeatureSchemaPack -> IO (Either String FeaturePack)
loadFeatureSchemaPack fsp = do 
  let list0 = view unwrapped fsp
  list1 <- runEitherT $ mapM loadFeatureSchema list0
  return $ fmap (view wrapped) list1


