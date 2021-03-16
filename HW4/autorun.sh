#!/bin/bash

outputpath=$1
sizes=(small large)
models=(1 2)

for size in "${sizes[@]}";
do
    mkdir -p $outputpath/${size}output/
    
    if [ $size = "small" ]; then
        n_epoch=30
    else
        n_epoch=60
    fi
    
    for m in "${models[@]}";
    do
        echo Running $size with model $m and n_epochs = $n_epoch
        ./build/Feature handout/${size}data/train_data.tsv handout/${size}data/valid_data.tsv handout/${size}data/test_data.tsv handout/dict.txt $outputpath/${size}output/model${m}_formatted_train.tsv $outputpath/${size}output/model${m}_formatted_valid.tsv $outputpath/${size}output/model${m}_formatted_test.tsv $m
        ./build/Lr $outputpath/${size}output/model${m}_formatted_train.tsv $outputpath/${size}output/model${m}_formatted_valid.tsv $outputpath/${size}output/model${m}_formatted_test.tsv handout/dict.txt $outputpath/${size}output/model${m}_train_out.labels $outputpath/${size}output/model${m}_test_out.labels $outputpath/${size}output/model${m}_metrics_out.txt $n_epoch
        echo "Comparing results : "
        output=$(diff -c $outputpath/${size}output/model${m}_metrics_out.txt handout/${size}output/model${m}_metrics_out.txt)
        if [ -z "$output" ]; then
            echo "The result matches expectation!"
        else
            echo $output
        fi
    done
done
