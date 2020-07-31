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
import           Data.Char (ord)
import           Data.Csv (FromNamedRecord)
import qualified Data.HashMap.Strict as H
import qualified Data.Vector as V
import           GHC.Generics (Generic)
import           Pipes (MonadIO(liftIO))
import           Pipes (cat, (>->), enumerate, yield, ListT(Select))
import qualified Pipes.Prelude as P
import qualified Pipes.Safe.Prelude as Safe
import           Pipes.Text
import           Pipes.Prelude.Text (fromHandleLn)
import           Pipes.Safe (runSafeT)
import           Lens.Family (view)
-- import           Torch
import           System.Directory
import           Control.Monad ((>=>))
import           Control.Monad (forM_)
import           System.IO (IOMode(ReadMode))
import           Data.Text (replace)
import qualified Data.Text as Text

import qualified Torch.Typed as Typed
import           Torch.Typed
import           Torch.Data.CsvDataset
import           Torch.Data.StreamedPipeline

import           Data.Reflection
import qualified Data.Set.Ordered as OSet
import qualified GHC.Exts as Exts
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           GHC.TypeNats

import           Data.Maybe (fromMaybe, fromJust)
import qualified Data.List as L
import           Control.Monad.Trans.Control (control)
import           Data.Foldable (toList)
import           Model
  
data Glove = Glove { label :: Text
                   , gloveEmbed :: [Float]
                   } deriving (Eq, Show, Generic)

instance FromRecord Glove where
  parseRecord v = do
    mLabel <- parseField $ v V.! 0 
    mEmbed <- parseRecord $ V.tail v
    pure Glove { label = mLabel
               , gloveEmbed = mEmbed
               }
pad = "[PAD]"
unk = "[UNK]"

ixSetWithEmbed :: Int -> (OSet.OSet Text, [[Float]])
ixSetWithEmbed embedDim = (ixSet, Prelude.replicate 2 embed)
  where ixSet =  "[UNK]" OSet.|< "[PAD]" OSet.|< OSet.empty
        embed = Prelude.replicate embedDim 0 

lookupIx ::  OSet.OSet Text -> Text -> Int
lookupIx set str = fromMaybe 1 (OSet.findIndex str set) 
  
gloveToTensor :: V.Vector Glove -> V.Vector (Text, [Float])
gloveToTensor gloves =  (\g -> (label g,  gloveEmbed g))  <$> gloves 

pipeline glove = makeListT' glove >=> pmap 2 (first gloveToTensor)
 
-- buildVocab :: (MonadIO m) => OSet.OSet Text -> ListT m ((V.Vector (Text, Tensor)), Int) -> m (H.HashMap Text Tensor)
getEmbeds :: forall device vocabSize embedDim m .
  ( MonadIO m
  ) =>
  ListT m ((V.Vector (Text, [Float])), Int)
  -> m _
getEmbeds inputs = P.foldM step begin done $  enumerate inputs 
  where step h1 (inputs, iter) = do
          pure $ V.foldl' (\ h (label, tensor) -> H.insert label tensor h) h1 inputs 
        begin = pure H.empty
        done = pure 

catIfMember ::
  Text ->
  OSet.OSet Text ->
  [Float] ->
  [[Float]] ->
  [[Float]]
catIfMember label oset embed embeds = if label `OSet.member` oset
                                      then embeds <> pure embed
                                      else embeds

main :: IO ()
main = runSafeT $ do
  -- we can probably avoid using a csvDataset here and just define our own since glove files aren't really valid csv (and this causes some issues)
  let (gloveFile :: CsvDataset Glove) = (csvDataset "glove.6B.100d.txt") { batchSize = 50, delimiter = fromIntegral (ord ' ') }
  let imdb = Imdb "../aclImdb/train"

  (indexSet, embeds) <- relevantEmbeddings imdb gloveFile
  liftIO $ print $ OSet.size indexSet

  init <- liftIO $ imdbModel @128 @1 @Unidirectional $ toTensoraa @'( CPU, 0) @5001 @100 $ embeds
  let optim = mkAdam 0 0.9 0.999 (Typed.flattenParameters init)
  -- TODO: batch tensors
  runContT ((tokenizerPipeline imdb >=> pmap 1 (first $ first $ padToken @256) >=> pmap' 1 (imdbToIndices @256 @'( 'CPU, 0)  indexSet) $ [()]))
    (train init optim)


  return ()

embedTensor embedding p = runEffect $ enumerate p >-> do
  (this, label) <- await  
  liftIO $ print $ Typed.embed embedding this
  
toTensoraa :: forall device vocabSize embedDim .
  ( Typed.All KnownNat '[vocabSize, embedDim]
  , Typed.KnownDevice device
  )
  => [[Float]]
  -> Typed.Tensor device 'Float '[vocabSize, embedDim]
toTensoraa stuff = fromJust $ Exts.fromList stuff -- bad

padToken :: forall (seqLen :: Nat) . (KnownNat seqLen) =>  [Text] -> [Text]
padToken tokens =  Prelude.take (Typed.natValI @seqLen) tokens <> (Prelude.take diffFrom $ repeat "[PAD]") 
  where diffFrom = (Typed.natValI @seqLen) - Prelude.length tokens

imdbToIndices :: forall seqLen device m a .
  (Functor m, Typed.KnownDevice device, KnownNat seqLen)
  => OSet.OSet Text
  -> Pipe (([Text], Sentiment), a) ((Typed.Tensor device 'Int64 '[seqLen], Typed.Tensor device 'Float '[]), a) m ()
imdbToIndices oset =
  for Pipes.cat $ \x -> case f $ fst x of
                                    Nothing -> return ()
                                    Just y -> yield (y, snd x)
    where f xs = do
            -- ugly because fromList doesn't support scalar tensors (yet?)
            labels <-  Typed.squeezeAll <$> ((Exts.fromList $ [fromEnum (snd xs)]) :: Maybe (Typed.Tensor '( 'CPU, 0) 'Int64 '[1]))
            let indices = lookupIx oset <$> fst xs 
            ixTensor <- Exts.fromList indices
            pure (toDevice @device @('( 'CPU, 0)) ixTensor, toDType @'Float @'Int64 $ toDevice @device @('( 'CPU, 0)) labels)

relevantEmbeddings imdb gloveFile = do
  (imdbVocab, embeds) <- concurrently (buildImdbVocab imdb) (runContT (pipeline gloveFile [()]) getEmbeds)
  let relevantTokens = H.intersectionWith (,) imdbVocab embeds
  liftIO $ print $ H.size relevantTokens
  --FIXME
  pure $ takeNMostCommon 4998 100 relevantTokens


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

-- dataPipeline :: forall batchSize device r m .
--   OSet.OSet Text -> Imdb -> ContT r m (ListT m ((Tensor device 'Int64 '[batchSize, 256], Tensor device 'Int64 '[batchSize]), Int))
-- dataPipeline indexSet imdb = tokenizerPipeline imdb >=> pmap 1 pad >=> pmap' 1 batches
--   where batches = (imdbToIndicesBatched @256 @'( 'CPU, 0)  indexSet) . view (chunksOf $ natValI @batchSize)
--         pad = first $ first $ padToken @256

-- TODO: fix this instead actually process punctuation
meaninglessTokens :: Char -> Bool 
meaninglessTokens char = Prelude.any (== char)  [ '\"',  ')', '(', ',', '.']

buildImdbVocab :: _ => Imdb -> m (H.HashMap Text Int)
buildImdbVocab trainData = runContT (tokenizerPipeline trainData [()]) $ (P.fold step begin done . enumerate)

  where step h ((tokens, _), _) =  foldr (\token hAccum ->  H.insertWith (+) token 1 hAccum) h tokens
        begin = H.empty
        done = id


-- | Sort the given hashmap by most common occurrences,
-- | and return the index set of tokens and the corresponding embedding table, for the N most common tokens
takeNMostCommon :: Int -> Int -> H.HashMap Text (Int, [Float]) -> (OSet.OSet Text, [[Float]])
takeNMostCommon vocabSize embeddingDim = foldl accumIxSetAndEmbed (ixSetWithEmbed 100) . mostCommon
  where mostCommon = L.take vocabSize . fmap (\(x, b) -> (x, snd b)) .  sortDesc . H.toList 
        sortDesc = L.sortBy (\a b -> compare (snd b) (snd a) )
  
accumIxSetAndEmbed = \(os, embeds) (x, embed)  -> (x OSet.|< os, embeds <> pure embed)
  
newtype Imdb = Imdb String

data Sentiment = Pos | Neg deriving (Eq, Show, Enum)

instance (MonadBaseControl IO m, MonadSafe m) => Datastream m () Imdb (Text, Sentiment) where
  -- streamBatch (Imdb str) _ = liftIO $ (listDirectory dataDir) >>= fold
  streamBatch (Imdb dataDir) _ = Select $ do
    liftIO $ getCurrentDirectory >>= print
    rawPosFilePaths <- zip (repeat Neg) <$> (liftIO $ listDirectory (dataDir <> "/pos"))
    rawNegFilePaths <- zip (repeat Pos) <$> (liftIO $ listDirectory (dataDir <> "/neg"))
    let posFilePaths = fmap (second $ mappend (dataDir <> "/pos/")) rawPosFilePaths
        negFilePaths = fmap (second $ mappend (dataDir <> "/neg/")) rawNegFilePaths

    for (each $ posFilePaths <> negFilePaths) $ \(rev, fp) -> Safe.withFile fp ReadMode $ \fh -> do
      P.zip (fromHandleLn fh) (yield rev)
  

