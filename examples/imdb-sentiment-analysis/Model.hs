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
module Model where


import Torch.Typed
import Torch.Typed.NN.Sparse

import GHC.TypeLits
-- import GHC.TypeLits.Extra
-- import GHC.TypeLits (KnownNat, Nat)
import GHC.Generics (Generic)
import Torch.Typed.NN.Recurrent.Aux
import qualified Pipes.Prelude as P
import Pipes ((>->), liftIO, ListT(enumerate))
import Control.Arrow (Arrow(first))

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
                     , fcSpec :: LinearSpec (hiddenSize GHC.TypeLits.* NumberOfDirections directionality) 1 dtype device
                     } deriving Generic

data GRUWithEmbed
  -- inputSize hiddenSize numLayers directionality initialization numEmbeds embedSize dtype device
   hiddenSize numLayers directionality initialization numEmbeds embedSize dtype device
  = GRUWithEmbed { gru :: GRUWithInit embedSize hiddenSize numLayers directionality initialization dtype device
                 -- , embed :: Embedding paddingIdx numEmbeds embedSize 'Learned dtype device
                 , gruEmbed :: Embedding 'Nothing numEmbeds embedSize 'Learned dtype device
                 , fc :: Linear (hiddenSize GHC.TypeLits.* NumberOfDirections directionality) 1 dtype device
                 } deriving Generic
instance
  ( KnownDType dtype
  , KnownDevice device
  -- , KnownNat inputSize
  , KnownNat hiddenSize
  , KnownNat numLayers
  , KnownNat (NumberOfDirections directionality)
  , KnownNat numEmbeds
  , KnownNat embedSize
  , RandDTypeIsValid device dtype
  -- , (2 GHC.TypeLits.<=? numLayers) ~ False
  , Randomizable (GRUSpec embedSize hiddenSize numLayers directionality dtype device)
                 (GRU embedSize hiddenSize numLayers directionality dtype device)
   
  ) =>
  Randomizable
  (GRUWithEmbedSpec hiddenSize numLayers directionality 'ConstantInitialization numEmbeds embedSize dtype device)
  (GRUWithEmbed hiddenSize numLayers directionality  'ConstantInitialization numEmbeds embedSize dtype device) where
  sample GRUWithEmbedSpec{..} = GRUWithEmbed <$> sample gruSpec <*> sample embeddingSpec <*> sample fcSpec

-- instance HasForward (GRUWithEmbedSpec inputSize hiddenSize numLayers directionality numEmbeds embedSize dtype device)
--          (Tensor device 'Int64 shape) (Tensor device dtype shape') where
--   forward = embed gruEmbed 
-- gruWithEmbedForward :: forall inputSize
--                              hiddenSize
--                              numLayers
--                              directionality
--                              initialization
--                              numEmbeds
--                              embedSize
--                              dtype
--                              device
--                              batchSize
--                              shape
--                              shapeOrder
--                              seqLen
--                              hShape
--                              hcShape
--                              .
--                              ( hShape ~ (hiddenSize * NumberOfDirections directionality)
--                              , hcShape ~ '[(numLayers * NumberOfDirections directionality), batchSize, hiddenSize]
--                              ,  _)
--                     => (GRUWithEmbed
--                              inputSize
--                              hiddenSize
--                              numLayers
--                              directionality
--                              initialization
--                              numEmbeds
--                              embedSize
--                              dtype
--                              device)
--                     -> (Tensor device 'Int64 shape)
--                     -> ((Tensor device dtype
--                          (RNNShape
--                           shapeOrder
--                           seqLen
--                           batchSize
--                           -- (hiddenSize * NumberOfDirections directionality)),
--                           hShape),
--                           Tensor
--                           device dtype hcShape)
--                        )
gruWithEmbedForward :: 
  (_) => GRUWithEmbed
       hiddenSize
       -- 256
       numLayers
       -- directionality
       -- 'Bidirectional
       'Unidirectional
       initialization
       numEmbeds
       embedSize
       -- 100
       'Float
       device
     -> Bool
     -> Tensor device 'Int64 '[batchSize, 128]
     -> Tensor device 'Float '[batchSize]
gruWithEmbedForward GRUWithEmbed{..} dropoutOn =
  --FIXME need to take the final 2 layers of the rnn and concat them together then feed this to the FC layer and softmax
  -- with dropout on final layer 
  -- we should be able to do this with Torch.Typed.Functional.chunk
   squeezeAll . forward fc . squeezeAll . snd . gruForward @BatchFirst dropoutOn gru . forward gruEmbed 
  
    
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
                                            , embeddingSpec = LearnedEmbeddingWithCustomInitSpec  tensor
                                            , fcSpec = LinearSpec
                                            }
                   ) 

train :: _ => _ -> _ -> ListT m ((Tensor device 'Int64 '[batchSize, 128], Tensor device 'Int64 '[batchSize]), Int) -> _
train model optim = P.foldM step begin done . enumerate 
  where step (model, optim) ((input, target), iter) = do
          let pred = sigmoid $ gruWithEmbedForward  model True input
          let loss =  binaryCrossEntropy @ReduceMean ones pred (toDType @'Float @'Int64 target) 
              errorCount = toDType @'Float @'Int64  . sumAll . ne (Torch.Typed.round pred)
          liftIO $ putStrLn $ "Loss: " <> show loss
          liftIO $ putStrLn $ "Error count: " <> show (errorCount target)
          liftIO $ runStep model optim loss 1e-3
        begin = pure (model, optim)
        done = pure 
          
data Trainer model optim = Trainer { model :: model 
                                   , optim :: optim
                                   , netLoss :: Int
                                   , acc  :: Acc
                                   }
data Acc
