#!/bin/bash

path_exp=./results/

cd results
for folder in */; do
    path_save="./${folder}projections_etomo/"
    
    rm -rf $path_save
    mkdir ${path_save}
    cp "./${folder}projections.mrc" "./${folder}projections_etomo.mrc"
    start_time=$(date +%s.%N)
    /usr/bin/time -f "%M" -o "${path_save}memory.log" batchruntomo -directive ../experiment_scripts/etomo_options.adoc -gpu 1 -cpus 8 -root projections_etomo -deliver "./${folder}" -current "./${folder}" > ${path_save}etomo_batch.log 
    memory_usage=$(tail -n 1 ${path_save}memory.log)
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken: $elapsed_time seconds"
    echo "Memory used: $memory_usage kilobytes"
done
cd ..




