#!/bin/bash

path_exp=./results/


path_save="${path_exp}Etomo/"
mkdir -p "$path_save"

cd results
for folder in */; do
    path_save="./${folder}Etomo/"
    mkdir -p $path_save
    
    start_time=$(date +%s.%N)
    memory_usage=$( /usr/bin/time -f "Time: %e seconds\nMemory: %M KB" xfalign  -InputImageFile "./${folder}projections.mrc" -OutputTransformFile "${path_save}projxfalign.xf" -FilterParameters 0.0,0.05,0,0.35 -ParametersToSearch 3 -PreCrossCorrelation 1 > ${path_save}xf.log; xftoxg -in "${path_save}projxfalign.xf" > ${path_save}xf.log; newstack -inp "./${folder}projections.mrc" -ou "${path_save}projxfalign.mrc" -bin 1 -xform "${path_save}projxfalign.xg" > ${path_save}xf.log 2>&1 )
    #memory_usage=$( /usr/bin/time -f "Time: %e seconds\nMemory: %M KB" xfalign  -InputImageFile ./projections.mrc -OutputTransformFile ./projxfalign.xf -FilterParameters 0.0,0.05,0,0.35 -ParametersToSearch 3 -PreCrossCorrelation 1  2>&1 )
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo $path_save
    echo "Time taken: $elapsed_time seconds"
    echo "Memory used: $memory_usage kilobytes"
done
cd ..






start_time=$(date +%s.%N)
memory_usage=$( /usr/bin/time -f "Time: %e seconds\nMemory: %M KB" xfalign  -InputImageFile ./projections.mrc -OutputTransformFile ./projxfalign.xf -FilterParameters 0.0,0.05,0,0.35 -ParametersToSearch 3 -PreCrossCorrelation 1 >xf.log; xftoxg -in projxfalign.xf >xf.log; newstack -inp ./projections.mrc -ou ./projxfalign.mrc -bin 1 -xform projxfalign.xg >xf.log 2>&1 )
#memory_usage=$( /usr/bin/time -f "Time: %e seconds\nMemory: %M KB" xfalign  -InputImageFile ./projections.mrc -OutputTransformFile ./projxfalign.xf -FilterParameters 0.0,0.05,0,0.35 -ParametersToSearch 3 -PreCrossCorrelation 1  2>&1 )
end_time=$(date +%s.%N)
elapsed_time=$(echo "$end_time - $start_time" | bc)
echo "Time taken: $elapsed_time seconds"
echo "Memory used: $memory_usage kilobytes"
