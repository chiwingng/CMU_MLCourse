#!/bin/bash

datasets=(politicians
          #education
         )
depths=($(seq 0 1 7))

for dataset in "${datasets[@]}"
do
    for depth in "${depths[@]}"
    do
        echo Running on ${dataset} with max_depth = ${depth}
        ./decisionTree handout/${dataset}_train.tsv handout/${dataset}_test.tsv $depth ${dataset}_${depth}_train.label ${dataset}_${depth}_test.label ${dataset}_${depth}_metrics.txt
    done
done
