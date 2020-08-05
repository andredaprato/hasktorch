{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}
module CNN where


import Torch.Typed
import GHC.TypeLits 
import GHC.Generics (Generic)

-- https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
-- TODO: you dont need these constructors, make them in the randomizable instance
data CNNSpec embedDim numEmbeds nFilters dtype device
  = CNNSpec { embeddingSpec :: EmbeddingSpec 'Nothing numEmbeds embedDim 'Constant dtype device
            , cnnSpec1      :: Conv2dSpec 1 nFilters 3 embedDim dtype device
            , cnnSpec2      :: Conv2dSpec 1 nFilters 4 embedDim dtype device
            , cnnSpec3      :: Conv2dSpec 1 nFilters 5 embedDim dtype device
            , fcSpec        :: LinearSpec (3 * nFilters) 1 dtype device
            , dropoutSpec   :: DropoutSpec
            } deriving Generic

data CNN  embedDim numEmbeds nFilters dtype device
  = CNN { embedding :: Embedding 'Nothing numEmbeds embedDim 'Constant dtype device
        , cnn1      :: Conv2d 1 nFilters 3 embedDim dtype device
        , cnn2      :: Conv2d 1 nFilters 4 embedDim dtype device
        , cnn3      :: Conv2d 1 nFilters 5 embedDim dtype device
        , fc        :: Linear (3 * nFilters) 1 dtype device
        , dropout   :: Dropout
        } deriving Generic
instance
  ( KnownDType dtype
  , KnownDevice device
  , KnownNat nFilters
  , KnownNat embedDim
  , RandDTypeIsValid device dtype
  ) =>
  Randomizable
  (CNNSpec embedDim numEmbeds nFilters dtype device)
  (CNN embedDim numEmbeds nFilters dtype device) where
  sample CNNSpec{..} = CNN <$> sample embeddingSpec
                       <*> sample cnnSpec1
                       <*> sample cnnSpec2
                       <*> sample cnnSpec3 
                       <*> sample fcSpec
                       <*> sample dropoutSpec

cnnForward :: forall seqLen batchSize embedDim numEmbeds nFilters device.
  _ => CNN embedDim numEmbeds nFilters 'Float device -> Tensor  device 'Int64 '[batchSize, seqLen] -> Tensor device 'Float _
cnnForward c@CNN{..} input = squeezeAll . forward fc $ cat  @1 ((first  embednew ) :. (send  embednew  ) :. (third  embednew ) :. HNil)  
  where
    first  = squeezeAll . maxPool1d @(seqLen - 3 + 1) @1 @0 .  squeezeAll . relu . conv2dForward @'( 1,1) @'( 0, 0) cnn1 
    send  = squeezeAll . maxPool1d @(seqLen - 4 + 1) @1 @0 . squeezeAll  . relu . conv2dForward @'( 1,1) @'( 0, 0) cnn2
    third  =  squeezeAll . maxPool1d @(seqLen - 5 + 1) @1 @0 .  squeezeAll . relu . conv2dForward @'( 1,1) @'( 0, 0) cnn3  
    embednew = unsqueeze @1 . embed embedding $ input
