{-# LANGUAGE FlexibleContexts, FlexibleInstances, MultiParamTypeClasses, TemplateHaskell, TupleSections, TypeFamilies, TypeSynonymInstances #-}
module SpaceWeather.FeaturePack where

import Control.Lens
import Control.Monad.Trans.Either
import Control.Monad.IO.Class
import qualified Data.Aeson.TH as Aeson
import qualified Data.ByteString.Char8 as BS
import Data.Char
import qualified Data.Map.Strict as Map
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Yaml as Yaml
import           System.IO
import qualified System.IO.Hadoop as HFS
import           Test.QuickCheck.Arbitrary
import           Text.Printf

import SpaceWeather.Feature
import SpaceWeather.Format
import SpaceWeather.TimeLine

data FeatureSchema 
  = FeatureSchema 
  { _colT           :: Int
  , _colX           :: Int
  , _scaling         :: Double
  , _isLog          :: Bool
  } deriving (Eq, Ord, Show, Read)
makeClassy ''FeatureSchema
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 1} ''FeatureSchema

defaultFeatureSchema :: FeatureSchema
defaultFeatureSchema = FeatureSchema
  { _colT = 1
  , _colX = 2
  , _scaling = 1
  , _isLog = False
  } 

newtype FeaturePack
  = FeaturePack [Feature] 
makeWrapped ''FeaturePack
makeClassy  ''FeaturePack

data FeatureSchemaPack
  = FeatureSchemaPack 
  { _fspSchemaDefinitions :: Map.Map String FeatureSchema
  , _fspFilenamePairs :: [(String, FilePath)]
  } deriving (Eq, Ord, Show, Read)
makeClassy  ''FeatureSchemaPack
Aeson.deriveJSON Aeson.defaultOptions{Aeson.fieldLabelModifier = drop 4} ''FeatureSchemaPack
instance Format FeatureSchemaPack where
  encode = T.pack . BS.unpack . Yaml.encode
  decode = Yaml.decodeEither . BS.pack . T.unpack

loadFeatureWithSchema :: FeatureSchema -> FilePath -> IO (Either String Feature)
loadFeatureWithSchema schema0 fp = runEitherT $ loadFeatureWithSchemaT schema0 fp

loadFeatureWithSchemaT :: FeatureSchema -> FilePath -> EitherT String IO Feature
loadFeatureWithSchemaT schema0 fp = do
  txt0 <- liftIO $ do
    hPutStrLn stderr $ "loading: " ++ fp
    HFS.readFile $ fp

  let 
      convert :: Double -> Double
      convert x = if schema0^.isLog then (if x <= 0 then 0 else log x / log 10) else x


      parseLine :: (Int, T.Text) -> Either String (TimeBin, Double)
      parseLine (lineNum, txt) = do
          let wtxt = T.words txt
              m2e :: Maybe a -> Either String a
              m2e Nothing =  Left $ printf "file %s line %d: parse error" fp lineNum
              m2e (Just x) = Right $ x
          t <- m2e $ readAt wtxt (schema0 ^. colT - 1)
          a <- m2e $ readAt wtxt (schema0 ^. colX - 1)
--           if (schema0^.isLog && a <= 0) 
--              then Left $ printf "file %s line %d:  logscale specified but non-positive colX value: %s" fp lineNum (show a)
--              else return (t,(schema0^.scaling) * convert a)
          return (t,(schema0^.scaling) * convert a)


      ret :: Either String Feature
      ret = fmap Map.fromList $ mapM parseLine $ linesWithComment txt0
  hoistEither ret


loadFeatureSchemaPack :: FeatureSchemaPack -> IO (Either String FeaturePack)
loadFeatureSchemaPack fsp = do 
  let 
      list0 = fsp ^. fspFilenamePairs
      scMap = fsp ^. fspSchemaDefinitions

      name2map :: String -> FeatureSchema
      name2map fmtName = maybe defaultFeatureSchema
        id (Map.lookup fmtName scMap)  
  list1 <- runEitherT $ mapM (uncurry loadFeatureWithSchemaT) $ 
    map (_1 %~ name2map) list0 
  return $ fmap FeaturePack list1

