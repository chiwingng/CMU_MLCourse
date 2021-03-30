# HW5
This is the folder holding the dataset and implementation of my HW5, a neural network characterizing images of characters.

# Compilation
The source code for the implementation is `src/neuralnet.cpp`, it can be compiled using CMake. From the `HW5` directory do the following:
- mkdir -p build
- cd build && cmake ..
- cmake --build ./
- cd ../

# Running
The code can be run directly through the executable `build/neuralnet`. The running command is:
- `build/neuralnet <input training> <input validation> <training output> <validation output> <metrics output> <n_epochs> <n_units> <init_flag> <learning_rate>`

A Bash script `autorun.sh` is also developed to help complete the assignment. The current version loops over 3 values of learning rate 
and plot the cross entropy value after the update for each epoch, and produce the plot via gnuplot. To use this feature one needs to 
install gnuplot before running this script.
