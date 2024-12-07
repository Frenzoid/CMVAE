#!/bin/bash

OUTPUTDIR='../outputs'
EXPERIMENT="CUBICC_1"
DATADIR='../data'
EPOCHS=300
SEED=2
SHARED_LAT_DIM=64
MS_LAT_DIM=32
# [NEW] Added beta parameter and adapted train call
BETA=1.0
# [NEW] Added cluster latent dimension and adapted train and prune calls
CLUSTER_LAT_DIM=35
# [NEW] Added number of samples parameters K and adapted train and prune calls
K_SAMPLES=10

# Train CMVAE
python train_CMVAE_CUBICC.py --experiment $EXPERIMENT --obj "dreg" --K $K_SAMPLES --batch-size 32 --epochs $EPOCHS \
      --latent-dim-c $CLUSTER_LAT_DIM --latent-dim-z $SHARED_LAT_DIM --latent-dim-w $MS_LAT_DIM --seed $SEED --beta $BETA \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --priorposterior 'Normal'

# Entropy-based latent cluster selection
# [NEW] Added prune call
python prune_CUBICC.py --save-dir "${OUTPUTDIR}/${EXPERIMENT}/checkpoints/${MS_LAT_DIM}_${SHARED_LAT_DIM}_${CLUSTER_LAT_DIM}_${BETA}_${K_SAMPLES}_${SEED}" \
                          --epoch $EPOCHS --seed $SEED \
                          --datadir $DATADIR
