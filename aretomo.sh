#!/bin/bash

if [ "$#" -eq 0 ]; then
    path_aretomo=/scicore/home/dokman0000/debarn0000/Softwares/AreTomo_1.3.4_Cuda101_Feb22_2023
    path_exp=./results/model_0_SNR_10_size_512_Nangles_61/
    VolZ=0
    AlignZ=180
    npatch=0
    echo $npatch
else
    IFS='|' read -r path_aretomo path_exp VolZ AlignZ npatch

    echo $AlignZ
fi


path_save="${path_exp}AreTomo/"
mkdir -p "$path_save"

# # return aligned tilt-series
# $path_aretomo -InMrc "${path_exp}projections.mrc" -OutMrc "${path_save}projections_aligned_aretomo_${npatch}by$npatch.mrc" -VolZ 0 -AlignZ $AlignZ -OutBin 1 -AngFile "${path_exp}angles.txt" -Patch $npatch $npatch -OutImod 1 -DarkTol 0.000001 -FlipVol 1 -TiltAxis 0.00000001 1 -TiltCor -1
# mv "${path_save}projections.aln" "${path_save}projections_${npatch}by$npatch.aln"
# # FBP with AreTomo
# $path_aretomo -InMrc "${path_exp}projections.mrc" -OutMrc "${path_save}projections_rec_aretomo_${npatch}by$npatch.mrc"  -VolZ $VolZ -AlignZ 0 -OutBin 1 -AngFile "${path_exp}angles.txt" -AlnFile "${path_save}projections_${npatch}by$npatch.aln" -Wbp 1 -Patch $npatch $npatch -OutImod 1 -DarkTol 0.000001 -FlipVol 1 -TiltAxis 0.00000001 1 -TiltCor -1


echo $npatch