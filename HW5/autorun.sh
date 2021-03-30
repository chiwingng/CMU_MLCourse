#!/bin/bash

output=$1
nlines=$2
xaxistitle=$3
yaxistitle=$4
plottitle=$5
#units=(5 20 50 100 200)
unit=50
dataset=large
n_epoch=100
init_flag=1
#lr=0.01
lrs=(0.1 0.01 0.001)
outputDir=output/${dataset}/
mkdir -p $outputDir
for lr in "${lrs[@]}";
do
    echo running $n_epoch $unit $init_flag $lr
    SECONDS=0
    outfile=${output}_${lr}
    echo "Epoch\tTrain\tValidation" > ${outfile}.txt
    ./build/neuralnet handout/${dataset}Train.csv handout/${dataset}Validation.csv $outputDir/${dataset}Train_lr${lr}_out.labels $outputDir/${dataset}Validation_lr${lr}_out.labels $outputDir/${dataset}Metrics_lr${lr}_out.txt $n_epoch $unit $init_flag $lr
    rm $outputDir/${dataset}Train_lr${lr}_out.labels $outputDir/${dataset}Validation_lr${lr}_out.labels
    echo Training done. Time used = $((SECONDS/60))m $((SECONDS%60))s.

    # format the data for plotting
    echo Formatting data...
    for i in $(seq $n_epoch);
    do
        trainerror=$(grep "epoch=$i " $outputDir/${dataset}Metrics_lr${lr}_out.txt | grep train | cut -d: -f2)
        validerror=$(grep "epoch=$i " $outputDir/${dataset}Metrics_lr${lr}_out.txt | grep valid | cut -d: -f2)
        echo "$i\t$trainerror\t$validerror" >> ${outfile}.txt
    done

    # gnuplot
    echo Gnuplot...
    echo set terminal png > plot.txt
    echo "set output \"${outfile}.png\"" >> plot.txt
    echo set key autotitle columnhead >> plot.txt
    echo "set xlabel \"${xaxistitle}\"" >> plot.txt
    echo "set ylabel \"${yaxistitle}\"" >> plot.txt
    echo "set title \"${plottitle} at Learning Rate ${lr}\"" >> plot.txt
    echo "plot for [i=2:${nlines}] \"${outfile}.txt\" using 1:i with line lw 3" >> plot.txt
    gnuplot plot.txt

    #open the final plot
    mv ${outfile}.png $outputDir/${outfile}.png
    rm plot.txt
    open $outputDir/${outfile}.png
done
