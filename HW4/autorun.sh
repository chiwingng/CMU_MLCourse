#!/bin/bash

sizes=(small large)
models=(1 2)
for size in "${sizes[@]}";
do
    if [ $size = "small" ]; then
        n_epoch=30
    else
        n_epoch=60
    fi
    
    for m in "${models[@]}";
    do
        echo Running $size with model $m and n_epochs = $n_epoch
        ./feature handout/${size}data/train_data.tsv handout/${size}data/valid_data.tsv handout/${size}data/test_data.tsv handout/dict.txt myresults/${size}output/model${m}_formatted_train.tsv myresults/${size}output/model${m}_formatted_valid.tsv myresults/${size}output/model${m}_formatted_test.tsv $m
        ./lr myresults/${size}output/model${m}_formatted_train.tsv myresults/${size}output/model${m}_formatted_valid.tsv myresults/${size}output/model${m}_formatted_test.tsv handout/dict.txt myresults/${size}output/model${m}_train_out.labels myresults/${size}output/model${m}_test_out.labels myresults/${size}output/model${m}_metrics_out.txt $n_epoch
        echo "My results : "
        cat myresults/${size}output/model${m}_metrics_out.txt
        echo "Should match : "
        cat handout/${size}output/model${m}_metrics_out.txt
    done
done
