{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
module Main where

import           Control.Arrow (Arrow(first))
import           Control.Monad (when)
import           Control.Monad.Cont (ContT(runContT))
import           Data.Char (ord)
import           Data.Csv (FromNamedRecord)
import qualified Data.HashMap.Strict as H
import qualified Data.Vector as V
import           GHC.Generics (Generic)
import           Pipes ((>->), enumerate, yield, ListT(Select))
import           Pipes (MonadIO(liftIO))
import qualified Pipes.Prelude as P
import           Pipes.Safe (runSafeT)
import           Torch
import           Torch.Data.CsvDataset
import           Torch.Data.StreamedPipeline
import System.Directory
import Control.Monad ((>=>))
import Data.Function ((&))
import qualified Pipes.Safe.Prelude as Safe
import Control.Monad (forM_)
import System.IO (IOMode(ReadMode))
import Pipes.Text 
import Data.Text (replace)
import Pipes.Prelude.Text (fromHandleLn)
import qualified Data.Text as Text

data Glove = Glove { label :: Text
                   , gloveEmbed :: V.Vector Float
                   } deriving (Eq, Show, Generic )

instance FromRecord Glove where
  parseRecord v = do
    mLabel <- parseField $ v V.! 0 
    mEmbed <- parseRecord $ V.tail v
    pure Glove { label = mLabel
               , gloveEmbed = mEmbed
               }

gloveToTensor :: V.Vector Glove -> V.Vector (Text, Tensor)
gloveToTensor gloves =  (\g -> (label g, asTensor . V.toList . gloveEmbed $ g))  <$> gloves 

pipeline dataset = [()] & (makeListT' dataset  >=> pmap 2 (first gloveToTensor))
 
buildVocab :: (MonadIO m) => ListT m ((V.Vector (Text, Tensor)), Int) -> m (H.HashMap Text Tensor)
buildVocab inputs = P.foldM step begin done $  enumerate inputs 
  where step h1 (inputs, iter) = do
          when (iter >= 39000) $ do
            liftIO $ print $ (fst <$> inputs , iter)
          pure $ V.foldl' (\ h (label, tensor) -> H.insert label tensor h) h1 inputs 

        begin = pure H.empty
        done = pure

main :: IO ()
main = runSafeT $ do
  let (gloveFile :: CsvDataset Glove) = (csvDataset "glove.6B.200d.txt") { batchSize = 50, delimiter = fromIntegral (ord ' ') }
  gloveEmbed <- runContT (pipeline gloveFile) buildVocab 
  liftIO $ print  $ H.lookup "the" gloveEmbed
  return ()

  -- maybe (lookup word gloveEmbed)
embedWord :: MonadIO m =>  Text -> H.HashMap Text Tensor -> Int -> m Tensor
embedWord word gloveEmbed seqLen = case H.lookup word gloveEmbed of
                         Nothing -> liftIO $ randnIO' [seqLen]
                         Just a -> pure a

  -- Torsten: use gru
  -- use embedding layer
newtype Imdb = Imdb String

-- | for preexisting datasets we could reexport a pipeline like this rather as well as the raw dataset
imdbPipeline dataset = makeListT' dataset
                       >=> pmap 1 (first $ Text.words .  Text.replace "<br>" " " . Text.toLower) 
                       >=> pmap 1 (first $ fmap $ Text.dropAround meaninglessTokens)

meaninglessTokens :: Char -> Bool 
meaninglessTokens char = char == '\"' || char == ')' || char == '('

readImdb :: IO ()
readImdb = runSafeT $ do
  let imdb = Imdb "../aclImdb/train"
  flip runContT (\x -> runEffect $ enumerate x >-> P.take 1 >-> P.print ) $ imdbPipeline imdb [()]

instance (MonadBase IO m, MonadSafe m) => Datastream m () Imdb Text where
  -- streamBatch (Imdb str) _ = liftIO $ (listDirectory dataDir) >>= fold
  streamBatch (Imdb dataDir) _ = Select $ do
    liftIO $ getCurrentDirectory >>= print
    -- TODO: use withCurrentDirectory
    liftIO $ setCurrentDirectory (dataDir <> "/pos")
    filePaths <- liftIO $ listDirectory "."
    liftIO $ print $ Prelude.head filePaths
  -- TODO: get rid of this forM_ and use a stream

    forM_ filePaths $ \fp -> Safe.withFile fp ReadMode $ \fh -> do

      fromHandleLn fh
        -- >-> P.map (replace "<br>" )
      -- undefined
      
-- ask adam paszke what his long term vision of dex is  
    
  
-- | JULY 25  DO THIS
--   - start from the ground up on RNN
--   - learn how RNNs work in good detail
--   - then LSTM
--   - then GRU
--   - then attention
--   - then transformers


-- ok how do we decode a variable length tensor without padding beforehand?
-- dynamic types????
