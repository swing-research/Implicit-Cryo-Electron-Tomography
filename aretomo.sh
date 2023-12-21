#!/bin/bash

if [ "$#" -eq 0 ]; then
    path_aretomo=/scicore/home/dokman0000/debarn0000/Softwares/AreTomo_1.3.4_Cuda101_Feb22_2023
    path_exp=./results/model_0_SNR_10_size_512_Nangles_61/
    VolZ=180
    AlignZ=180
    npatch=0
else
    IFS='|' read -r path_aretomo path_exp VolZ AlignZ npatch
fi


path_save="${path_exp}AreTomo"
mkdir -p "$path_save"

$path_aretomo -InMrc "${path_exp}projections.mrc" -OutMrc "${path_save}projections_rec_aretomo_${npatch}by$npatch.mrc" -VolZ $VolZ -AlignZ $AlignZ -OutBin 1 -AngFile ../angles.txt -Wbp 1 -Patch 0 0 -OutImod 1 -DarkTol 0.001 -FlipVol 1 -TiltAxis 0.00000001 1 -TiltCor -1


