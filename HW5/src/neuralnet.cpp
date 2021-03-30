//--------------------------------------------//
//               neuralnet.cpp                //
// Author: Chi Wing Ng                        //
// Date:   March 24, 2021                     //
// C++ program to implement a neural network  //
// on a set of input pixel values of images   //
// for character recognition                  //
//--------------------------------------------//

// IO includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

// STL inlcudes
#include <vector>
#include <stack>
#include <utility> // std::pair

// Linear algebra includes
#include <Eigen/Dense>

constexpr bool DEBUG = false;

class Layer {
public:
  Layer() {};
  virtual Eigen::VectorXd forwardPass(Eigen::VectorXd & input_data) = 0;
  virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> backwardPass(Eigen::VectorXd input_data, Eigen::VectorXd output_data, Eigen::VectorXd output_gradient) = 0;
  virtual bool isLinearLayer() = 0;
  void setdebug(bool debug) {debug = debug;}
protected:
  bool debug;
};

class LinearLayer : public Layer {
public:
  LinearLayer(unsigned int input_dimension, unsigned int output_dimension, unsigned int init_flag) {
    if(init_flag == 2) weights = Eigen::MatrixXd::Zero(output_dimension, input_dimension+1); // extra input dimension for the bias parameter
    else if(init_flag == 1) {
      weights = Eigen::MatrixXd::Random(output_dimension, input_dimension+1)/10.0; // scale range of random to [-0.1, 0.1]
      weights.col(0) = Eigen::VectorXd::Zero(output_dimension); // first column (bias weight) is always initialized to 0
    }
  }
  
  inline Eigen::VectorXd forwardPass(Eigen::VectorXd & input_data) {
    Eigen::VectorXd output = weights * foldBiasTerm(input_data);
    if(debug) std::cout << "a/b:" << std::endl << output.transpose() << std::endl << std::endl;
    return weights * foldBiasTerm(input_data);
  }

  // gradients w.r.t. input/weights (input = vector, weights = matrix
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> backwardPass(Eigen::VectorXd input_data, Eigen::VectorXd output_data, Eigen::VectorXd output_gradient) {
    Eigen::VectorXd input_gradient = unfoldBiasTerm((weights.transpose()) * output_gradient);
    if(debug) std::cout << "d(loss)/d(input) = " << std::endl << input_gradient.transpose() << std::endl << std::endl;
    Eigen::MatrixXd weight_gradient = output_gradient * foldBiasTerm(input_data).transpose();
    if(debug) std::cout << "d(loss)/d(weight) = " << std::endl << weight_gradient << std::endl << std::endl;
    return std::make_pair(input_gradient, weight_gradient);
  }
  
  void updateWeights(Eigen::MatrixXd weights_change) {
    weights -= weights_change;
  }

  void printWeights() {
    std::cout << "Weights = " << std::endl << weights << std::endl << std::endl;
  }
  
  bool isLinearLayer() {return true;}

  // accessor
  inline Eigen::MatrixXd getWeights() { return weights; }
  
private:
  Eigen::VectorXd unfoldBiasTerm(Eigen::VectorXd input_gradient) {
    return Eigen::Map<Eigen::VectorXd>(input_gradient.data()+1, input_gradient.size()-1);
  }

  Eigen::VectorXd foldBiasTerm(Eigen::VectorXd & input_data) {
    std::vector<double> data_vect(input_data.data(), input_data.data()+input_data.size());
    data_vect.insert(data_vect.begin(), 1.);
    return Eigen::Map<Eigen::VectorXd>(data_vect.data(), data_vect.size());
  }
  Eigen::MatrixXd weights;
};

class SigmoidLayer : public Layer {
public:
  SigmoidLayer() {};
  
  Eigen::VectorXd forwardPass(Eigen::VectorXd & input_data) {
    Eigen::VectorXd output = input_data;
    for(unsigned int i = 0; i < output.size(); i++) output[i] = 1./(1. + std::exp(-output[i]));
    if(debug) std::cout << "z = " << std::endl << output.transpose() << std::endl << std::endl;
    return output;
  }
  
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> backwardPass(Eigen::VectorXd input_data, Eigen::VectorXd output_data, Eigen::VectorXd output_gradient) {
    Eigen::VectorXd gradient = output_data;
    if(debug) std::cout << "d(loss)/d(z) = " << std::endl << output_gradient.transpose() << std::endl << std::endl;
    for(unsigned int i = 0; i < gradient.size(); i++) gradient[i] *= (1. - gradient[i]);
    gradient = gradient.cwiseProduct(output_gradient);
    if(debug) std::cout << "d(loss)/d(a) = " << std::endl << gradient.transpose() << std::endl << std::endl;
    return std::make_pair(gradient, Eigen::MatrixXd());
  }

  bool isLinearLayer() {return false;}
};

class SoftMaxLayer : public Layer {
public:
  SoftMaxLayer() {};
  
  Eigen::VectorXd forwardPass(Eigen::VectorXd & input_data) {
    Eigen::VectorXd output = input_data.array().exp().matrix();
    double sum = output.sum();
    output /= sum;
    if(debug) std::cout << "y^hat = " << std::endl << output.transpose() << std::endl << std::endl;
    return output;
  }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> backwardPass(Eigen::VectorXd input_data, Eigen::VectorXd output_data, Eigen::VectorXd output_gradient) {
    Eigen::MatrixXd sftmax_gradient(output_data.size(), input_data.size());
    sftmax_gradient.colwise() = Eigen::Map<Eigen::VectorXd>(output_data.data(), output_data.size());
    sftmax_gradient = Eigen::MatrixXd::Identity(output_data.size(), input_data.size()) - sftmax_gradient;
    output_gradient = output_gradient.cwiseProduct(output_data);
    Eigen::VectorXd result = sftmax_gradient * output_gradient;
    if(debug) std::cout << "d(loss)/d(b) = " << std::endl << result.transpose() << std::endl << std::endl;
    return std::make_pair(result, Eigen::MatrixXd());
  }

  bool isLinearLayer() {return false;}
};

class ObjectiveLayer {
public:
  ObjectiveLayer() {};

  double forwardPass(Eigen::VectorXd & predicted_labels, Eigen::VectorXd & correct_predictions) {
    return -correct_predictions.dot(predicted_labels.array().log().matrix());
  }

  Eigen::VectorXd backwardPass(Eigen::VectorXd & predicted_labels, Eigen::VectorXd & correct_predictions) {
    Eigen::VectorXd result = -predicted_labels.cwiseInverse().cwiseProduct(correct_predictions);
    if(debug) std::cout << "d(loss)/d(y^hat) = " << std::endl << result.transpose() << std::endl << std::endl;
    return result;
  }

  void setdebug(bool debug) {debug = debug;}

private:
  bool debug;
};

class NeuralNetwork {
public:
  NeuralNetwork(std::vector<std::ifstream> & inputs, const char input_delim, unsigned int num_epoch, unsigned int hidden_units, unsigned int init_flag, double learning_rate, unsigned int n_labels)
    : num_epoch(num_epoch), hidden_units(hidden_units), init_flag(init_flag), learning_rate(learning_rate), n_labels(n_labels)
  {
    input_dimension = 0;
    readData(inputs, input_delim);
    L1 = new LinearLayer(input_dimension, hidden_units, init_flag);
    L2 = new LinearLayer(hidden_units, n_labels, init_flag);
    S1 = new SigmoidLayer();
    SM1 = new SoftMaxLayer();
    processLayers = {L1, S1, L2, SM1};
    for(auto layer : processLayers) layer->setdebug(DEBUG);
    O1 = ObjectiveLayer();
    O1.setdebug(DEBUG);
    dataset_tags = {"(train): ", "(validation): "};
  }

  void train(std::vector<std::ofstream> & outputs, std::ofstream & metric_output) {
    for(unsigned int i_epoch = 1; i_epoch <= num_epoch; i_epoch++) {
      if(i_epoch % 10 == 0) std::cout << "Epoch " << i_epoch << std::endl;
      for(unsigned int i_example = 0; i_example < input_data[0].size(); i_example++) {
        if(DEBUG) std::cout << "Example " << i_example << " \t";
        std::vector<Eigen::VectorXd> forwardPassResults = forwardPass(input_data[0][i_example], 0);
        if(DEBUG) std::cout << "Cross Entropy = " << O1.forwardPass(forwardPassResults.back(), correct_predictions[0][i_example]) << std::endl;
        std::vector<Eigen::MatrixXd> weights_gradient = backwardPass(forwardPassResults, correct_predictions[0][i_example]);
        L1->updateWeights(weights_gradient[0] * learning_rate);
        L2->updateWeights(weights_gradient[1] * learning_rate);
      }// example loop for SGD
      //std::cout << "Alpha = " << std::endl;
      //L1->printWeights();
      //std::cout	<< "Beta = " << std::endl;
      //L2->printWeights();
      for(unsigned int dataset = 0; dataset < dataset_tags.size(); dataset++) {
        metric_output << "epoch=" << i_epoch << " crossentropy" << dataset_tags[dataset] << std::setprecision(12) << std::fixed << totalCrossEntropy(dataset) << std::endl;;
      }// dataset loop for cross entropy
    }// epoch loop
    
    std::vector<std::vector<unsigned int> > all_labels;
    for(unsigned int dataset = 0; dataset < dataset_tags.size(); dataset++) {
      std::vector<unsigned int> labels;
      for(unsigned int example = 0; example < input_data[dataset].size(); example++) {
        Eigen::VectorXd prediction = forwardPass(input_data[dataset][example]).back();
        unsigned int label = convertLabel(prediction);
        outputs[dataset] << label << std::endl;
        labels.push_back(label);
      }// example loop for predictions
      all_labels.push_back(labels);
    }// dataset loop for predictions
    printErrors(metric_output, all_labels);
  }

  void printErrors(std::ofstream & metric_output, const std::vector<std::vector<unsigned int> > & all_labels) {
    for(unsigned int dataset = 0; dataset < all_labels.size(); dataset++) {
      double error = 0.;
      unsigned int Nexamples = all_labels[dataset].size();
      for(unsigned int example = 0; example < Nexamples; example++) error += (all_labels[dataset][example] != correct_labels[dataset][example]);
      error /= Nexamples;
      metric_output << "error" << dataset_tags[dataset] << std::setprecision(2) << std::fixed << error << std::endl;
    }// dataset loop
  }

  unsigned int convertLabel(const Eigen::VectorXd & prediction) {
    unsigned int index;
    prediction.maxCoeff(&index);
    return index;
  }
  
  std::vector<unsigned int> convertLabels(const std::vector<Eigen::VectorXd> & predictions) {
    std::vector<unsigned int> result;
    for(unsigned int i = 0; i < predictions.size(); ++i) {
      result.push_back(convertLabel(predictions[i]));
    }
    return result;
  }

  inline double getNExamples(unsigned int dataset_id) {return input_data[dataset_id].size();}
  
private:
  unsigned int num_epoch, hidden_units, init_flag, input_dimension, n_labels;
  double learning_rate;
  // Layers: L1 -> S1 -> L2 -> SM1 -> O1
  LinearLayer *L1, *L2;
  SigmoidLayer *S1;
  SoftMaxLayer *SM1;
  std::vector<Layer*> processLayers;
  ObjectiveLayer O1;
  std::vector<std::string> dataset_tags;
  std::vector<std::vector<Eigen::VectorXd> > input_data, correct_predictions;
  std::vector<std::vector<unsigned int> > correct_labels;
  void readData(std::vector<std::ifstream> & inputs, const char input_delim) {
    for(std::ifstream & input : inputs) {
      std::vector<Eigen::VectorXd> mydata;
      std::vector<Eigen::VectorXd> mypredictions;
      std::vector<unsigned int> mylabels;
      std::string line;
      while(std::getline(input, line)) {
        std::istringstream iss(line);
        std::string temp;
        std::vector<double> line_data;
        // get label and its corresponding one-hot encoding
        std::getline(iss, temp, input_delim);
        unsigned int label = stoi(temp);
        if(label >= n_labels) throw std::runtime_error("label " + std::to_string(label) + " is out of bound!");
        mylabels.push_back(label);
        Eigen::VectorXd prediction = Eigen::VectorXd::Zero(n_labels);
        prediction[label] = 1.0;
        mypredictions.push_back(prediction);

        // get pixel map
        while(std::getline(iss, temp, input_delim)) {
          line_data.push_back(std::stod(temp));
        }// read pixel loop
        if(input_dimension == 0) input_dimension = line_data.size();
        else if(input_dimension != line_data.size()) {
          if(DEBUG) std::cout << line << std::endl;
          throw std::runtime_error("Inconsistent input data! " + std::to_string(input_dimension) + " is not equal to the dimension of " + std::to_string(mydata.size()) + "th line (" + std::to_string(line_data.size()) + ")!");
        }
        Eigen::Map<Eigen::VectorXd> input_example(line_data.data(), input_dimension);
        mydata.push_back(input_example);
        
      }// read line loop
      input_data.push_back(mydata);
      correct_predictions.push_back(mypredictions);
      correct_labels.push_back(mylabels);
    }// read file loop
  }

  double totalCrossEntropy(unsigned int dataset) {
    double result = 0;
    for(unsigned int example = 0; example < getNExamples(dataset); example++) result += crossEntropy(input_data[dataset][example], dataset, example);
    return result/getNExamples(dataset);
  }

  double crossEntropy(Eigen::VectorXd data, unsigned int dataset, unsigned int example, unsigned int start_layer = 0) {
    return O1.forwardPass(forwardPass(data, start_layer).back(), correct_predictions[dataset][example]);
  }

  std::vector<Eigen::VectorXd> forwardPass(Eigen::VectorXd & input_data, const unsigned int & start_layer = 0) {
    std::vector<Eigen::VectorXd> result{input_data};
    for(unsigned int layer = start_layer; layer < processLayers.size(); layer++) {
      result.push_back(processLayers[layer]->forwardPass(result.back()));
    }
    return result;
  }

  std::vector<Eigen::MatrixXd> backwardPass(std::vector<Eigen::VectorXd> forwardPassResult, Eigen::VectorXd correct_predictions) {
    if(forwardPassResult.size() != 5) throw std::runtime_error("Not enough forwardPassResult information passed to backwardPass!");
    std::stack<Eigen::MatrixXd> result_stack;
    Eigen::VectorXd gradient = O1.backwardPass(forwardPassResult.back(), correct_predictions);
    for(unsigned int layer = processLayers.size() - 1; layer >= 0; layer--) {
      std::pair<Eigen::VectorXd, Eigen::MatrixXd> layer_gradient = processLayers[layer]->backwardPass(forwardPassResult.end()[-2], forwardPassResult.back(), gradient);
      //gradient.resize(layer_gradient.first);
      gradient = layer_gradient.first;
      if(processLayers[layer]->isLinearLayer()) result_stack.push(layer_gradient.second);
      if(layer == 0) break;
      forwardPassResult.pop_back();
    }// layer loop
    // transfer gradients from stack to the result vector
    std::vector<Eigen::MatrixXd> result;
    while(!result_stack.empty()) {
      result.push_back(result_stack.top());
      result_stack.pop();
    }
    return result;
  }
};

int printUsage(char** argv) {
  std::cout << "Usage of this program: " << std::endl;
  std::cout << argv[0] << " <train input> <validation input> <train out> <validation out> <metrics out> <num epoch> <hidden units> <init flag> <learning rate>" << std::endl;
  std::cout << "init flag = 1 : Random initialization of weights; 2: Zero initialization of weights (bias term is always zero at initial stage)" << std::endl << std::endl;
  std::cout << "Example command from the run directory: " << std::endl;
  std::cout << argv[0] << " ../handout/smallTrain.csv ../handout/smallValidation.csv smallTrain_out.labels smallValidation_out.labels smallMetrics_out.txt 2 4 2 0.1" << std::endl;
  return 1;
}

int main(int argc, char** argv) {
  if(argc != 10) return printUsage(argv);

  std::vector<std::ifstream> inputs;
  std::vector<std::ofstream> outputs;
  for(unsigned int i = 1; i < 3; i++) inputs.push_back(std::ifstream(argv[i], std::ifstream::in));
  for(unsigned int i = 3; i < 5; i++) outputs.push_back(std::ofstream(argv[i], std::ofstream::out));
  std::ofstream metrics_out = std::ofstream(argv[5], std::ofstream::out);
  unsigned int num_epoch = std::stoi(argv[6]);
  unsigned int hidden_units = std::stoi(argv[7]);
  unsigned int init_flag = std::stoi(argv[8]);
  double learning_rate = std::stod(argv[9]);

  NeuralNetwork myNet(inputs, ',', num_epoch, hidden_units, init_flag, learning_rate, 10);
  myNet.train(outputs, metrics_out);
}
