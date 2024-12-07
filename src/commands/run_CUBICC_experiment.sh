#!/bin/bash

OUTPUTDIR='../outputs'
EXPERIMENT="CUBICC_1"
DATADIR='../data'
EPOCHS=300
SEED=2
SHARED_LAT_DIM=64
MS_LAT_DIM=32
BETA=1.0

# Train CMVAE
python train_CMVAE_CUBICC.py --experiment $EXPERIMENT --obj "dreg" --K 10 --batch-size 32 --epochs $EPOCHS \
      --latent-dim-c 35 --latent-dim-z $SHARED_LAT_DIM --latent-dim-w $MS_LAT_DIM --seed $SEED --beta $BETA \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --priorposterior 'Normal'
