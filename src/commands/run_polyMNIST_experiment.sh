#!/bin/bash

OUTPUTDIR='../outputs'
EXPERIMENT="PolyMNIST_1"
DATADIR='../data'
EPOCHS=250
SEED=2
SHARED_LAT_DIM=32
MS_LAT_DIM=32
BETA=2.5

# Train CMVAE
python train_CMVAE_polyMNIST.py --experiment $EXPERIMENT --obj "iwae" --K 1 --batch-size 128 --epochs $EPOCHS \
      --latent-dim-c 40 --latent-dim-z $SHARED_LAT_DIM --latent-dim-w $MS_LAT_DIM --seed $SEED --beta $BETA \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --inception_path "${DATADIR}/pt_inception-2015-12-05-6726825d.pth" \
      --pretrained-clfs-dir-path "${DATADIR}/trained_clfs_polyMNIST" \
      --priorposterior 'Laplace'

# Entropy-based latent cluster selection
python prune_polyMNIST.py --save-dir "${OUTPUTDIR}/${EXPERIMENT}/checkpoints/${MS_LAT_DIM}_${SHARED_LAT_DIM}_${BETA}_${SEED}" \
                          --epoch $EPOCHS --seed $SEED \
                          --datadir DATADIR
