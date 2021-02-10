#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <unordered_map>

int print_help(char** argv) {
  std::cout << "Usage of this program:" << std::endl;
  std::cout << argv[0] << " <train input> <test input> <split index> <train out> <test out> <metrics out>" << std::endl;
  std::cout << "Example: " << std::endl;
  std::cout << argv[0] << " politicians_train.tsv politicians_test.tsv 0 pol_0_train.labels pol_0_test.labels pol_0_metrics.txt" << std::endl;
  return 1;
}

struct node {
  int split_index;        // since only one feature is logged, the split index is always 0
  int category;              // if category is non-negative, a category is assigned to this node (it's a leaf now)
  struct node *left, *right; // left subtree (feature = false) and right subtree (feature = true)
};

node* newNode(int split_index, int category = -1) {
  node* node = new struct node();
  node->split_index = split_index;
  node->category = category;
  node->left = NULL;
  node->right = NULL;
  return node;
}

class MyClassifier{

public:

  MyClassifier(std::string train_input, std::string test_input, size_t split_index, std::string train_output, std::string test_output, std::string metrics_output):
    train_input(train_input), test_input(test_input), train_output(train_output), test_output(test_output), metrics_output(metrics_output), split_index(split_index) {
    rootNode = newNode(0);
    result_indices = std::unordered_map<std::string, unsigned int>();
    feature_indices = std::unordered_map<std::string, unsigned int>();
    readData(train_input, train_features, train_result);
    readData(test_input, test_features, test_result);
  };

  void train() {
    std::vector< std::vector<unsigned int> > counts(2, std::vector<unsigned int>(result_categories.size(), 0));
    for(size_t i = 0; i < train_features.size(); i++) counts[train_features[i]][train_result[i]]++;
    rootNode->left = newNode(0, max_element(counts[0].begin(), counts[0].end()) - counts[0].begin());
    rootNode->right = newNode(0, max_element(counts[1].begin(), counts[1].end()) - counts[1].begin());
    std::cout << "Training done!" << std::endl;
  }

  void predictAll() {
    train_predictions = predict(train_features);
    test_predictions = predict(test_features);
    std::cout << "Prediction done!" << std::endl;
  }

  void printPredictions() {
    std::ofstream f_train(train_output.c_str());
    std::ofstream f_test(test_output.c_str());
    for(size_t i = 0; i < train_predictions.size(); i++) {
      f_train << result_categories[train_predictions[i]] << std::endl;
    }
    std::cout << "Train predictions written to " << train_output << std::endl;
    for(size_t i = 0; i < test_predictions.size(); i++) {
      f_test << result_categories[test_predictions[i]] << std::endl;
    }
    std::cout << "Test predictions written to " << test_output << std::endl;
    f_train.close();
    f_test.close();
  }

  void printError() {
    std::ofstream f(metrics_output.c_str());
    f << "error(train): " << std::setprecision(6) << std::fixed << computeError(train_predictions, train_result) << std::endl;
    f << "error(test): " << std::setprecision(6) << std::fixed << computeError(test_predictions, test_result) << std::endl;
    f.close();
  }

  void readData(std::string input_file, std::vector<unsigned int> & feature, std::vector<unsigned int> & result) {
    std::ifstream file(input_file.c_str());
    std::string line, temp;
    std::istringstream iss;
    size_t index = 0;
    
    getline(file, line);
    iss.str(line);

    // find the feature name corresponding to split_index
    if(input_file == train_input) {
      while(iss >> temp) {
        if(index++ == split_index) {
          feature_name = temp;
          std::cout << "Spliting the dataset using " << feature_name << "!" << std::endl;
        }
      }
    }
    // record the features and the corresponding result
    while(getline(file, line)) {
      std::istringstream iss1;
      index = 0;
      iss1.str(line);
      while(iss1 >> temp) {
        if(index++ == split_index) {
          if(feature_indices.find(temp) == feature_indices.end()) {
            feature_indices[temp] = feature_categories.size();
            feature_categories.push_back(temp);
          }
          feature.push_back(feature_indices[temp]);
        }
      }
      // register the category if not seen before
      if(result_indices.find(temp) == result_indices.end()) {
        result_indices[temp] = result_categories.size();
        result_categories.push_back(temp);
      }
      result.push_back(result_indices[temp]);
    }
    
    std::cout << "Data from " << input_file << " is read successfully!" << std::endl;
  }

  std::vector<unsigned int> predict(std::vector<unsigned int> & features) {
    std::vector<unsigned int> predictions;
    for(size_t i = 0; i < features.size(); i++) {
      predictions.push_back(features[i] ? rootNode->right->category : rootNode->left->category);
    }
    return predictions;
  }

  double computeError(std::vector<unsigned int> & predictions, std::vector<unsigned int> & result) {
    double error = 0;
    size_t n_examples = predictions.size();
    for(size_t i = 0; i < n_examples; i++) error += (predictions[i] != result[i]);
    return error/n_examples;
  }

private:

  size_t split_index;                                                             // index to split on the dataset to perform predictions
  node* rootNode;                                                                 // Tree to be trained
  std::string train_input, test_input, train_output, test_output, metrics_output; // paths to input/output files
  std::string feature_name;                                                       // name of the feature used to split the dataset
  std::vector<std::string> feature_categories, result_categories;                 // categories for feature and result
  std::unordered_map<std::string, unsigned int> feature_indices, result_indices;  // map of categories to indices
  std::vector<unsigned int> train_features, train_predictions, train_result;      // feature values, predicted results and true results for train dataset
  std::vector<unsigned int> test_features, test_predictions, test_result;         // feature values, predicted results and true results for test dataset

};

int main(int argc, char** argv) {
  if(argc != 7) return print_help(argv);
  std::string train_input = std::string(argv[1]);
  std::string test_input = std::string(argv[2]);
  size_t split_index = atoi(argv[3]);
  std::string train_output = std::string(argv[4]);
  std::string test_output = std::string(argv[5]);
  std::string metrics_output = std::string(argv[6]);

  MyClassifier c = MyClassifier(train_input, test_input, split_index, train_output, test_output, metrics_output);
  c.train();
  c.predictAll();
  c.printPredictions();
  c.printError();
  return 0;
}
