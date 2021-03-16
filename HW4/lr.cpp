//--------------------------------------------//
//                  lr.cpp                    //
// Author: Chi Wing Ng                        //
// Date:   March 14, 2021                     //
// C++ program to perform logistic regression //
// on a set of formatted input files          //
//--------------------------------------------//

// IO includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

// data structures
#include <vector>
#include <string>
#include <unordered_map>

// input conversion
#include <cstdlib> // atoi

// computation
#include <numeric> // std::inner_product
#include <cmath>   // log, exp

int printUsage(char** argv) {
  std::cout << "Usage of this program:" << std::endl;
  std::cout << argv[0] << " <formatted train input> <formatted validation input> <formatted test input> <dict input> <train out> <test out> <metrics out> <num epoch>" << std::endl;
  return 1;
}

class LogisticRegression {
public:
  // constructor
  LogisticRegression(std::vector<std::ifstream> & inputs, std::ifstream & dict_input, double alpha, unsigned int n_epochs);

  // method to train model on train_input and validation_input
  int trainModel(std::ostream & out);

  // method to predict upon the current parameters
  int predict();

  // method to write output labels
  int writeOutput(std::vector<std::ofstream> & outputs);

  // method to write train/test errors
  int writeError(std::ofstream & output);

private:
  // Data members
  // model parameters (n_words + 1 dimension)
  std::vector<double> theta;
  // learning rate
  const double alpha;
  // number of epochs
  const unsigned int n_epochs;
  // dictionary size
  unsigned int n_words;
  // train, validation and test data (0 = train, 1 = validation, 2 = test)
  std::vector< std::vector<unsigned int> > labels, predictions;
  std::vector< std::vector<std::vector<unsigned int> > > features;

  // Private member functions
  // method to read data from file
  int readData(std::vector<std::ifstream> & inputs);

  // method to compute negative log likelihoods of the current state for each of the datasets
  std::vector<double> computeLogLikelihoods();
  
  // method to write the negative log likehood to an output file
  void writeLogLikelihood(std::ostream & output, double index);

  // method to compute train and test error rate
  std::vector<double> computeError();
  
  // extract feature dimension
  void getNwords(std::ifstream & dict_input);
};

LogisticRegression::LogisticRegression(std::vector<std::ifstream> & inputs, std::ifstream & dict_input, double alpha, unsigned int n_epochs)
  : n_epochs(n_epochs), alpha(alpha)
{
  getNwords(dict_input);
  theta = std::vector<double>(n_words + 1);
  int errorCode = readData(inputs);
  if(errorCode) {
    std::cout << "Error occured when reading in data!" << std::endl;
  }
}

int LogisticRegression::readData(std::vector<std::ifstream> & inputs) {
  for(unsigned int i = 0; i < inputs.size(); i++) {
    if(!inputs[i]) return i+1;
    std::vector<unsigned int> this_labels;
    std::vector< std::vector<unsigned int> > this_features;
    std::string line, pair;
    int label;
    while(getline(inputs[i], line)) {
      std::istringstream iss(line);
      iss >> label;
      this_labels.push_back(label);
      std::vector<unsigned int> example_features(n_words + 1, 0);
      while(iss >> pair) {
        size_t pos = pair.find(':');
        example_features[stoi(pair.substr(0, pos))] = stoi(pair.substr(pos+1));
      }// loop for features in a line
      // bias term folded into the last entry of each features
      example_features.back() = 1;
      this_features.push_back(example_features);
    }// loop for lines in input file
    labels.push_back(this_labels);
    features.push_back(this_features);
  }// loop for input files
  return 0;
}

int LogisticRegression::trainModel(std::ostream & out) {
  unsigned int N_examples = features[0].size();
  for(unsigned int i_epoch = 0; i_epoch < n_epochs; i_epoch++) {
    for(unsigned int i_example = 0; i_example < N_examples; i_example++) {
      // theta^T * x
      double dot_product = std::inner_product(theta.begin(), theta.end(), features[0][i_example].begin(), 0.);
      // sigmoid function
      double sigmoid = 1./(1.+exp(-dot_product));
      // prefactor in the gradient expression
      double prefactor = (labels[0][i_example] - sigmoid) / N_examples;
      for(unsigned int i_param = 0; i_param < theta.size(); i_param++) {
        theta[i_param] += alpha * prefactor * features[0][i_example][i_param];
      }// loop of parameters to update theta
    }// loop of examples for SGD
    writeLogLikelihood(out, i_epoch);
  }// loop of epochs
  return 0;
}

int LogisticRegression::predict() {
  predictions.clear();
  for(unsigned int i_dataset = 0; i_dataset < features.size(); i_dataset++) {
    std::vector<unsigned int> this_predictions;
    unsigned int N_examples = features[i_dataset].size();
    for(unsigned int i_example = 0; i_example < N_examples; i_example++) {
      // theta^T * x
      double product = std::inner_product(theta.begin(), theta.end(), features[i_dataset][i_example].begin(), 0.);
      // the sigmoid function predicts 1 if u >= 0 and 0 if u < 0
      this_predictions.push_back(product >= 0);
    }// loop of examples
    predictions.push_back(this_predictions);
  }// loop of datasets
  return 0;
}

int LogisticRegression::writeOutput(std::vector<std::ofstream> & outputs) {
  for(unsigned int i_output = 0; i_output < outputs.size(); i_output++) {
    // skip the validation dataset
    unsigned int i_dataset = 2*i_output;
    unsigned int N_examples = features[i_dataset].size();
    for(unsigned int i_example = 0; i_example < N_examples; i_example++) {
      outputs[i_output] << predictions[i_dataset][i_example] << std::endl;
    }// loop of examples
  }// loop of datasets/output streams
  return 0;
}

int LogisticRegression::writeError(std::ofstream & output) {
  std::vector<double> errors = computeError();
  output << "error(train): " << std::setprecision(6) << std::fixed << errors[0] << std::endl;
  output << "error(test): " <<	std::setprecision(6) <<	std::fixed << errors[2] << std::endl;
  return 0;
}

std::vector<double> LogisticRegression::computeLogLikelihoods() {
  std::vector<double> negLogLikelihoods;
  for(unsigned int i_dataset = 0; i_dataset < features.size(); i_dataset++) {
    unsigned int N_examples = features[i_dataset].size();
    double negLogLikelihood = 0;
    for(unsigned int i_example = 0; i_example < N_examples; i_example++) {
      // theta^T * x
      double dot_product = std::inner_product(theta.begin(), theta.end(), features[i_dataset][i_example].begin(), 0.);
      negLogLikelihood += (log(1 + exp(dot_product)) - labels[i_dataset][i_example] * dot_product);
    }// loop of examples
    negLogLikelihoods.push_back(negLogLikelihood/N_examples);
  }// loop of datasets
  return negLogLikelihoods;
}

void LogisticRegression::writeLogLikelihood(std::ostream & output, double index) {
  std::vector<double> negLogLikelihoods = computeLogLikelihoods();
  output << index << "\t" << std::setprecision(6) << std::fixed << negLogLikelihoods[0] << "\t" << negLogLikelihoods[1] << std::endl;
}

std::vector<double> LogisticRegression::computeError() {
  std::vector<double> errors;
  for(unsigned int i_dataset = 0; i_dataset < labels.size(); i_dataset++) {
    unsigned int N_examples = labels[i_dataset].size();
    double error_count = 0;
    for(unsigned int i_example = 0; i_example < N_examples; i_example++) {
      error_count += (labels[i_dataset][i_example] != predictions[i_dataset][i_example]);
    }//loop for examples
    errors.push_back(error_count/N_examples);
  }// loop for dataset
  return errors;
}

void LogisticRegression::getNwords(std::ifstream & dict_input) {
  std::string temp;
  n_words = 0;
  while(getline(dict_input, temp)) n_words++;
}

int main(int argc, char** argv) {
  if(argc != 9) return printUsage(argv);
  // read in data
  std::vector<std::ifstream> inputs;
  for(unsigned int i = 1; i < 4; i++) inputs.push_back(std::ifstream(argv[i], std::ifstream::in));
  std::ifstream dict_input(argv[4], std::ifstream::in);
  std::vector<std::ofstream> outputs;
  for(unsigned int i = 5; i < 7; i++) outputs.push_back(std::ofstream(argv[i], std::ofstream::out));
  std::ofstream metrics_output(argv[7], std::ofstream::out);
  unsigned int n_epochs = atoi(argv[8]);
  double alpha = 0.1;

  LogisticRegression myModel(inputs, dict_input, alpha, n_epochs);
  std::ofstream out("loglikelihood.txt");
  myModel.trainModel(out);
  myModel.predict();
  myModel.writeOutput(outputs);
  myModel.writeError(metrics_output);

  return 0;
}
