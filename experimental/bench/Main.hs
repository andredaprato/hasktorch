{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
module Main where

import Torch
import Torch.Data.StreamedPipeline 
import Torch.Data.Dataset
import Control.Monad.Cont (ContT(runContT))
import qualified Pipes as Pipes
import Control.Exception (evaluate)
import Control.Monad (forever)
import System.Clock 
import qualified Pipes.Prelude as P
import Criterion.Main
import Criterion.Types (Config(reportFile))

newtype SyntheticDataset = SyntheticDataset { iters :: Int } 

instance Datastream IO (Int, Int) SyntheticDataset Tensor where
  streamBatch s (id, numSeeds)  = Select $ for (each [start .. end]) $ \_ -> do yield $ ones' [ 10 ]
    where start = id * (iters s `Prelude.div` numSeeds)
          end = (id + 1) * (iters s `Prelude.div` numSeeds)

seeds :: Int -> [(Int, Int)]
seeds n = zip (take n [0..]) (repeat n)

readEpoch :: ((Datastream IO (Int, Int) dataset Tensor)) => dataset -> [(Int, Int)] ->  IO () 
readEpoch dataset seeds = do
  val <- runContT (makeListT' dataset seeds) $ (runEffect . consume)
  pure ()

consume x = enumerate x >-> (for Pipes.cat $  (\x -> do void . pure . (+ 1) . fst $ x))
  
batch :: Pipe [Tensor] Tensor IO ()
batch = for Pipes.cat $ \x -> yield $ Torch.cat (Dim 0) x

main :: IO ()
main = do
  defaultMainWith
    (defaultConfig { reportFile = Just "dataloader.html"})
    [env (pure ())
         (\ ~_ ->
            let syntheticDataset = SyntheticDataset 500_000_0
                batchedDataset = CollatedDataset { set = syntheticDataset, chunkSize = 64, collateFn = batch}
            in bgroup
              "dataloader with threads/5000000"
              [
                bench "read batch size (64) tensors with 1 seed(s)"
                $ nfIO $ readEpoch batchedDataset (seeds 1)
              , bench "read batch size (64) tensors with 2 seed(s)"
                $ nfIO $ readEpoch batchedDataset (seeds 2)
              ])
    ]

