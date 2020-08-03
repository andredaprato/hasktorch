{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
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

import           Control.Arrow (second, Arrow(first))
import           Control.Concurrent.Async.Lifted
import           Control.Monad (when)
import           Control.Monad.Cont (ContT(runContT))
import qualified Control.Foldl as L
import           Data.Char (ord)

import qualified Data.HashMap.Strict as H
import qualified Data.Vector as V
import           GHC.Generics (Generic)
import           Pipes
import qualified Pipes.Prelude as P
import qualified Pipes.Safe.Prelude as Safe
import qualified Pipes.Group as Group
import qualified Pipes.Text as PT
import qualified Pipes.Prelude.Text as PT
import           Pipes.Safe (runSafeT)
import           Lens.Family (view)
import           System.Directory
import           System.FilePath
import           Control.Monad ((>=>))
import           System.IO (IOMode(ReadMode))
import           Data.Text (Text(..))
import qualified Data.Text as Text

import           Torch.Data.CsvDataset
import           Torch.Data.StreamedPipeline
import           Torch.Typed

import qualified Data.Set.Ordered as OSet
import qualified GHC.Exts as Exts
import           GHC.TypeLits

import           Data.Maybe (fromMaybe, fromJust)
import qualified Data.List as List
import           Data.Foldable (toList)
import           Model
  
data Glove = Glove { label :: Text.Text
                   , gloveEmbed :: [Float]
                   } deriving (Eq, Show, Generic)

instance FromRecord Glove where
  parseRecord v = do
    mLabel <- parseField $ v V.! 0 
    mEmbed <- parseRecord $ V.tail v
    pure Glove { label = mLabel , gloveEmbed = mEmbed}

getEmbeds :: forall device vocabSize embedDim m .
  ( MonadIO m
  ) =>
  ListT m (V.Vector Glove, Int)
  -> m _
getEmbeds = P.fold step begin done .  enumerate 
  where step h1 (inputs, iter) = do
          V.foldl' (\ h Glove{..} -> H.insert label gloveEmbed h) h1 inputs 
        begin = H.empty
        done = id 

pad = "[PAD]"
unk = "[UNK]"

ixSetWithEmbed :: Int -> (OSet.OSet Text, [[Float]])
ixSetWithEmbed embedDim = (ixSet, Prelude.replicate 2 embed)
  where ixSet =   "[PAD]" OSet.|< "[UNK]" OSet.|< OSet.empty
        embed = Prelude.replicate embedDim 0 

lookupIx ::  OSet.OSet Text -> Text -> Int
lookupIx set str = fromMaybe 1 (OSet.findIndex str set) 

type BatchSize = 64
type SeqLen = 128
type VocabSize = 10000
type EmbedDim = 100
-- type Directionality 

-- https://ai.stanford.edu/~amaas/data/sentiment/
-- https://nlp.stanford.edu/projects/glove/
main :: IO ()
main = runSafeT $ do
  -- we should probably avoid using a csvDataset here and just define our own dataset 
  -- since glove files aren't really valid csv (and this causes some issues)
  let (gloveFile :: CsvDataset Glove) = (csvDataset "glove.6B.100d.txt") { batchSize = 50, delimiter = fromIntegral (ord ' ') }
  let imdb = Imdb "aclImdb/train"

  (indexSet, embeds) <- relevantEmbeddings imdb gloveFile
  liftIO $ print $ OSet.size indexSet

  init <- liftIO $ imdbModel @128 @4 @Unidirectional $ toTensorUnsafe @'( CUDA, 0) @VocabSize @EmbedDim $ embeds
  let optim    = mkAdam 0 0.9 0.999 (flattenParameters init)
      trainer  = initTrainer init optim

  _ <- V.foldM (\trainer' epoch  -> runContT (dataPipeline @SeqLen @BatchSize indexSet imdb [Pos, Neg])
                                                        --  $ train model optim' (gruWithEmbedForward True)
                                                        $ train trainer' (gruWithEmbedForward True)
                            ) trainer (V.fromList [1..10])
  return ()

toTensorUnsafe :: forall device vocabSize embedDim .
  ( All KnownNat '[vocabSize, embedDim]
  , KnownDevice device
  )
  => [[Float]]
  -> Tensor device 'Float '[vocabSize, embedDim]
toTensorUnsafe stuff = fromJust $ Exts.fromList stuff -- bad

padSequence :: forall (seqLen :: Nat) . (KnownNat seqLen) =>  [Text] -> [Text]
padSequence tokens =  Prelude.take (natValI @seqLen) tokens <> (Prelude.take diffFrom $ repeat pad) 
  where diffFrom = (natValI @seqLen) - Prelude.length tokens

imdbToIndices :: forall seqLen batchSize device m .
  (Functor m, KnownDevice device, KnownNat seqLen, KnownNat batchSize)
  => OSet.OSet Text
  -> Pipe [(([Text], Sentiment), Int)] ((Tensor device 'Int64 '[batchSize, seqLen], Tensor device 'Int64 '[batchSize]), Int) m ()
imdbToIndices oset =
  for Pipes.cat $ \x -> case f x of
                                    Nothing -> return ()
                                    Just y -> yield ( y, (snd . head $  x) `mod` natValI @batchSize)
    where f batch = do
            labels <- (Exts.fromList $ (fromEnum . snd . fst) <$> batch) :: Maybe (Tensor '( 'CPU, 0) 'Int64 '[batchSize])
            let indices = (fmap) (fmap (lookupIx oset) . fst . fst) $ batch
            ixTensor <- Exts.fromList indices
            pure (toDevice @device @('( 'CPU, 0)) ixTensor, toDevice @device @('( 'CPU, 0)) labels)

relevantEmbeddings imdb gloveFile = do
  (imdbVocab, embeds) <- concurrently (buildImdbVocab imdb) (runContT (makeListT' gloveFile [()]) getEmbeds)
  let relevantTokens = H.intersectionWith (,) imdbVocab embeds
  liftIO $ print $ H.size relevantTokens
  --FIXME cleanup these constants
  pure $ takeNMostCommon 9998 100 relevantTokens

-- | for preexisting datasets we could reexport a pipeline like this as well as the raw dataset
tokenizerPipeline dataset = makeListT' dataset
                            -- FIXME
                       >=> pmap 1 (first $ first $ Text.words
                                   . Text.replace "/>" " "
                                   . Text.replace "<br" " "
                                   . Text.replace "<br>" " "
                                   . Text.replace "*" " "
                                   . Text.toLower
                                   ) 
                       >=> pmap 1 (first $ first  $ fmap $ Text.dropAround meaninglessTokens)

dataPipeline :: forall seqLen batchSize device r m f seed .  _ =>
  OSet.OSet Text -> Imdb -> f seed -> ContT r m (ListT m ((Tensor device 'Int64 '[batchSize, seqLen], Tensor device 'Int64 '[batchSize]), Int))
dataPipeline indexSet imdb = tokenizerPipeline imdb >=> pmapChunk batches >=> pmap' 1 (imdbToIndices indexSet)

  where batches = L.purely Group.folds (mapList pad) . view (Group.chunksOf $ natValI @batchSize) 
        pad = first . first $ padSequence @seqLen

-- | Fold all values into a list
mapList :: (a->b) -> L.Fold a [b]
mapList f = L.Fold (\x a -> x . (f a:)) id ($ [])

-- TODO: fix this instead actually process punctuation
meaninglessTokens :: Char -> Bool 
meaninglessTokens char = Prelude.any (== char)  [ '\"',  ')', '(', ',', '.']

buildImdbVocab :: _ => Imdb -> m (H.HashMap Text Int)
buildImdbVocab trainData = runContT (tokenizerPipeline trainData [Pos, Neg]) $ (P.fold step begin done . enumerate)
  where step h ((tokens, _), _) =  foldr (\token hAccum ->  H.insertWith (+) token 1 hAccum) h tokens
        begin = H.empty
        done = id

-- | Sort the given hashmap by most common occurrences,
-- | and return the index set of tokens and the corresponding embedding table, for the N most common tokens
takeNMostCommon :: Int -> Int -> H.HashMap Text (Int, [Float]) -> (OSet.OSet Text, [[Float]])
takeNMostCommon vocabSize embeddingDim = foldl accumIxSetAndEmbed (ixSetWithEmbed 100) . mostCommon
  where mostCommon = List.take vocabSize . fmap (\(x, b) -> (x, snd b)) .  sortDesc . H.toList 
        sortDesc = List.sortBy (\a b -> compare (snd b) (snd a) )
  
accumIxSetAndEmbed = \(os, embeds) (x, embed)  -> (os OSet.|> x, embeds <> pure embed)
  
newtype Imdb = Imdb String

data Sentiment = Pos | Neg deriving (Eq, Show, Enum)

instance (MonadBaseControl IO m, MonadSafe m) => Datastream m Sentiment Imdb (Text, Sentiment) where
  streamBatch (Imdb dataDir) sent = Select $ do
    liftIO $ getCurrentDirectory >>= print
    rawFilePaths <- zip (repeat sent) <$> (liftIO $ listDirectory (dataDir </> sentToPath sent))
    let filePaths = fmap (second $ mappend (dataDir </> sentToPath sent)) rawFilePaths
    for (each $ filePaths ) $ \(rev, fp) -> Safe.withFile fp ReadMode $ \fh -> do
      P.zip (PT.fromHandleLn fh) (yield rev)
   where sentToPath Pos = "pos" ++ pure pathSeparator 
         sentToPath Neg = "neg" ++ pure pathSeparator
  

