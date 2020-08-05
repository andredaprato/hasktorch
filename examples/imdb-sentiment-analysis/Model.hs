{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE NoStarIsType #-}
module Model where


import Torch.Typed
import Torch.Typed.NN.Sparse

import GHC.TypeLits
import GHC.Generics (Generic)
import Torch.Typed.NN.Recurrent.Aux
import qualified Pipes.Prelude as P
import Pipes ((>->), liftIO, ListT(enumerate))
import Control.Arrow (Arrow(first))
import Control.Monad (when)
import System.Mem (performGC)
import System.IO.Unsafe (unsafePerformIO)
import qualified Debug.Trace as Debug

data GRUWithEmbedSpec
  -- (inputSize :: Nat)
  (hiddenSize :: Nat)
  (numLayers :: Nat)
  (directionality :: RNNDirectionality)
  -- (paddingIdx :: Maybe Nat)
  (initialization :: RNNInitialization)
  (numEmbeds :: Nat)
  (embedSize :: Nat)
  (dtype :: DType)
  (device :: (DeviceType, Nat))
  = GRUWithEmbedSpec { embeddingSpec :: EmbeddingSpec 'Nothing numEmbeds embedSize 'Learned dtype device
                     , gruSpec :: GRUWithInitSpec embedSize hiddenSize numLayers directionality initialization dtype device
                     , fcSpec :: LinearSpec (hiddenSize * NumberOfDirections directionality) 1 dtype device
                     } deriving Generic

data GRUWithEmbed
   hiddenSize numLayers directionality initialization numEmbeds embedSize dtype device
  = GRUWithEmbed { gru :: GRUWithInit embedSize hiddenSize numLayers directionality initialization dtype device
                 , gruEmbed :: Embedding 'Nothing numEmbeds embedSize 'Learned dtype device
                 , fc :: Linear (hiddenSize * NumberOfDirections directionality) 1 dtype device
                 } deriving Generic
instance
  ( KnownDType dtype
  , KnownDevice device
  , KnownNat hiddenSize
  , KnownNat numLayers
  , KnownNat (NumberOfDirections directionality)
  , KnownNat numEmbeds
  , KnownNat embedSize
  , RandDTypeIsValid device dtype
  , Randomizable (GRUSpec embedSize hiddenSize numLayers directionality dtype device)
                 (GRU embedSize hiddenSize numLayers directionality dtype device)
   
  ) =>
  Randomizable
  (GRUWithEmbedSpec hiddenSize numLayers directionality 'ConstantInitialization numEmbeds embedSize dtype device)
  (GRUWithEmbed hiddenSize numLayers directionality  'ConstantInitialization numEmbeds embedSize dtype device) where
  sample GRUWithEmbedSpec{..} = GRUWithEmbed <$> sample gruSpec <*> sample embeddingSpec <*> sample fcSpec

gruWithEmbedForward :: 
  _ =>
     Bool
  -> GRUWithEmbed
       hiddenSize
       4 -- num Layers
       Unidirectional
       initialization
       numEmbeds
       embedSize
       'Float
       device
     -> Tensor device 'Int64 '[batchSize, 128]
     -> Tensor device 'Float '[batchSize]
gruWithEmbedForward dropoutOn GRUWithEmbed{..} =
   -- squeezeAll . forward fc . squeezeAll . snd . gruForward @BatchFirst dropoutOn gru . forward gruEmbed 
   squeezeAll . forward fc . f . chunk @5 @0 .  snd . gruForward @BatchFirst dropoutOn gru . forward gruEmbed 
   -- squeezeAll . forward fc . f . chunk @2 @0 .  snd . gruForward @BatchFirst dropoutOn gru . forward gruEmbed 
  where f g@(_ :. _ :. _ :. lastLayer) = cat @2 lastLayer
  -- where f l = cat @2 l
  
imdbModel :: forall hiddenSize numLayers directionality numEmbeds embedSize dtype device .
  _ =>
  Tensor device dtype '[numEmbeds, embedSize]
  -> IO (GRUWithEmbed 
          hiddenSize
          numLayers
          directionality
          'ConstantInitialization
          numEmbeds
          embedSize
          'Float
          device
        )
imdbModel tensor = sample (GRUWithEmbedSpec { gruSpec = GRUWithZerosInitSpec (GRUSpec (DropoutSpec 0.5))
                                            , embeddingSpec = LearnedEmbeddingWithCustomInitSpec tensor
                                            -- , embeddingSpec = ConstEmbeddingSpec tensor
                                            , fcSpec = LinearSpec
                                            }
                   ) 

train ::  forall batchSize device model optim m forward . _ =>
  Trainer model optim device -> _ -> ListT m ((Tensor device 'Int64 '[batchSize, 128], Tensor device 'Int64 '[batchSize]), Int) -> m _
train trainer forward  = P.foldM step begin done . enumerate 
  where step t@Trainer{..} ((input, target), _) = do
          let pred = sigmoid $ forward model input
          let loss = binaryCrossEntropy @ReduceMean ones pred (toDType @'Float @'Int64 target) 
              errorCount =  sumAll . ne (Torch.Typed.round pred) $ target
          -- liftIO $ print pred
          liftIO $ print target 
          newParams <- liftIO $ runStep model optim loss 1e-3
          pure $ updateTrainer newParams loss (toInt $ errorCount) $ t
        begin = pure trainer
        done t@Trainer{..} = do  
            liftIO $ print iter 
            liftIO $ putStrLn $ "Loss: " <> show netLoss  
            liftIO $ putStrLn $ "Accuracy: " <> show (1 - fromIntegral errorCount / (fromIntegral $ natValI @batchSize))
            pure $ t { Model.iter = 0}
          
-- An extensible record for this sort of datatype would be really nice
data Trainer model optim device = Trainer { model :: model 
                                          , optim :: optim
                                          -- right now we get a gpu memory leak if we accumulate these metrics
                                          -- which is strange since static-mnist accumulates metrics using toFloat. 
                                          -- , netLoss :: Float
                                          , errorCount  :: Int
                                          , netLoss :: Tensor device 'Float '[]
                                          -- , errorCount  :: Tensor device  'Int64 '[]
                                          , iter :: Int
                                   }
updateTrainer (model', optim') loss' errorCount' Trainer{..} =  Trainer { model = model'
                                                                        , optim = optim'
                                                                        -- , netLoss = loss' + netLoss
                                                                        -- , errorCount = errorCount' + errorCount
                                                                        , netLoss = loss' 
                                                                        , errorCount = errorCount' 
                                                                        , iter = iter + 1
                                                                        }
initTrainer model optim = Trainer model optim 0 0 0
