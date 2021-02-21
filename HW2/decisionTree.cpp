#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#define DEBUG 0

//-----------------------------------
// help function
//-----------------------------------
int print_help(char** argv) {
  std::cout << "Usage of this program:" << std::endl;
  std::cout << argv[0] << " <train input> <test input> <max_depth> <train out> <test out> <metrics out>" << std::endl;
  std::cout << "Example: " << std::endl;
  std::cout << argv[0] << " politicians_train.tsv politicians_test.tsv 0 pol_0_train.labels pol_0_test.labels pol_0_metrics.txt" << std::endl;
  return 1;
}

//-----------------------------------
// tree structure
//-----------------------------------
struct node {
  int split_index;           // index of feature to split on this node, -1 indicates a leaf
  int category;              // if category is non-negative, a category is assigned to this node (it's a leaf now)
  size_t current_depth;      // current depth of this node from the root
  std::vector< std::vector<size_t> > features; // training data to be classified by this node, including labels (last index of vector)
  std::unordered_set<int> features_considered; // features that are already considered before the split at this node
  std::vector<node*> subtrees; // subtrees corresponding to each of the categories of split_index
};

// node constructor
node* newNode(size_t current_depth, std::vector< std::vector<size_t> > features, std::unordered_set<int> features_considered, int category = -1) {
  node* node = new struct node();
  node->split_index = -1;
  node->category = category;
  node->current_depth = current_depth;
  node->features = features;
  node->features_considered = features_considered;
  node->subtrees = std::vector<struct node*>();
  return node;
}

//-----------------------------------
// Error calculation
//-----------------------------------
double computeError(std::vector<size_t> & predictions, std::vector<size_t> & labels) {
  double error = 0;
  size_t n_examples = predictions.size();
  for(size_t i = 0; i < n_examples; i++) error += (predictions[i] != labels[i]);
  return error/n_examples;
}

//-----------------------------------
// Entropy calculation
//-----------------------------------
double computeEntropy(std::vector< double > & label_counts) {
  double entropy = 0;
  double total_counts = std::accumulate(label_counts.begin(), label_counts.end(), 0.);
  size_t n_labels = label_counts.size();
  for(size_t i = 0; i < n_labels; i++) {
    if(label_counts[i] != 0) {
      double p = label_counts[i]/total_counts;
      entropy -= p * log2(p);
    }
  }
  return entropy;
}

double conditionalEntropy(std::vector<size_t> & feature, const size_t & n_feature_categories, std::vector< size_t > & labels, const size_t & n_labels) {
  std::vector< std::vector<double> > label_counts(n_feature_categories, std::vector<double>(n_labels, 0.));
  double total_counts =	feature.size();
  // split label counts into feature categories
  for(size_t i = 0; i < total_counts; i++) {
    label_counts[feature[i]][labels[i]]++;
  }
  // compute and sum the entropies
  double entropy = 0.;
  for(size_t i = 0; i < n_feature_categories; i++) {
    double count = std::accumulate(label_counts[i].begin(), label_counts[i].end(), 0.);
    if(count == 0) continue;
    entropy += (count/total_counts) * computeEntropy(label_counts[i]);
  }
  return entropy;
}

class MyClassifier{

public:

  // constructor
  MyClassifier(std::string train_input, std::string test_input, size_t max_depth): train_input(train_input), test_input(test_input), max_depth(max_depth) {
    readData(train_input, train_features);
    readData(test_input, test_features);
    std::unordered_set<int> features_considered;
    rootNode = newNode(0, train_features, features_considered);
    if(DEBUG) std::cout << "MyClassifier initlaized!" << std::endl;
  };

  void train() {
    rootNode = trainNode(rootNode);
    if(DEBUG) std::cout << "Training done!" << std::endl;
  }

  void predictAll() {
    train_predictions = predict(train_features);
    test_predictions = predict(test_features);
    if(DEBUG) std::cout << "Prediction done!" << std::endl;
  }

  void printPredictions(std::ostream & out_train, std::ostream & out_test) {
    printPredictions(out_train, train_predictions, feature_categories.back());
    printPredictions(out_test, test_predictions, feature_categories.back());
  }

  void printError(std::ostream & out) {
    out << "error(train): " << std::setprecision(6) << std::fixed << computeError(train_predictions, train_features.back()) << std::endl;
    out << "error(test): " << std::setprecision(6) << std::fixed << computeError(test_predictions, test_features.back()) << std::endl;
  }
  
  void printTree(std::ostream & out) { printTree(out, rootNode); }

private:

  size_t max_depth;                                                                      // index to split on the dataset to perform predictions
  size_t n_features, n_labels;                                                           // number of features (including label) and number of label categories in dataset
  node* rootNode;                                                                        // Tree to be trained
  std::string train_input, test_input;                                                   // paths to input/output files
  std::vector<std::string> feature_names;                                                // name of the features/label
  std::vector< std::vector<std::string> > feature_categories;                            // categories for each feature/label
  std::vector< std::unordered_map<std::string, size_t> > feature_indices;                // map of feature/label categories to indices
  std::vector< std::vector<size_t> > train_features, test_features;                      // feature values for train and test dataset [1st index: feature id, 2nd index: example id]
  std::vector<size_t> train_predictions, test_predictions;                               // predicted results for train and test dataset

  // read in data from input_file into feature vectors
  void readData(std::string input_file, std::vector< std::vector<size_t> > & feature) {
    std::ifstream file(input_file.c_str());
    std::string line, temp;
    std::istringstream iss;
    
    getline(file, line);
    iss.str(line);

    // first line should be feature names
    while(iss >> temp) {
      feature_names.push_back(temp);
    }
    // read in the feature information from train data
    if(input_file == train_input) {
      n_features = feature_names.size();
      feature_categories = std::vector< std::vector<std::string> >(n_features);
      feature_indices = std::vector< std::unordered_map<std::string, size_t> >(n_features);
    }
    feature = std::vector< std::vector<size_t> >(n_features);
    // record the features and the corresponding result
    while(getline(file, line)) {
      std::istringstream iss1;
      iss1.str(line);
      size_t feature_id = 0;
      while(iss1 >> temp) {
        if(feature_id < n_features) {
          if(feature_indices[feature_id].find(temp) == feature_indices[feature_id].end()) {
            feature_indices[feature_id][temp] = feature_categories[feature_id].size();
            feature_categories[feature_id].push_back(temp);
          }
          feature[feature_id].push_back(feature_indices[feature_id][temp]);
          feature_id++;
        }
        else {
          std::cerr << "more columns than the first row!" << std::endl;
          throw std::runtime_error("readData error!");
        }
      }
    }
    n_labels = feature_categories.back().size();
    if(DEBUG) std::cout << "Data from " << input_file << " is read successfully!" << std::endl;
  }

  // perform prediction using the trained tree
  std::vector<size_t> predict(std::vector< std::vector<size_t> > & features) {
    std::vector<size_t> predictions;
    for(size_t i = 0; i < features[0].size(); i++) {
      node* my_node = rootNode;
      // search for a leaf node using the features of the current example
      while(my_node->category < 0) {
        my_node = my_node->subtrees[features[my_node->split_index][i]];
      }
      predictions.push_back(my_node->category);
    }
    return predictions;
  }

  // Feature selection
  int selectBestFeatureToSplit(std::vector< std::vector<size_t> > & features, std::unordered_set<int> & features_considered) {
    // compute original entropy for all data in this node
    std::vector<size_t> labels = features.back();
    std::vector<double> label_counts(n_labels, 0.);
    for(size_t i = 0; i < labels.size(); i++) label_counts[labels[i]]++;
    double original_entropy = computeEntropy(label_counts);
    // skip loop if the data is perfectly classified
    if(original_entropy == 0.) return -1;
    // loop over unconsidered features to find the best feature
    // to split on that provides largest positive mutual information
    int feature_id = -1;
    double mutual_information_max = 0.;
    for(size_t i = 0; i < n_features - 1; i++) {
      if(features_considered.find(i) == features_considered.end()) {
        double mutual_info = original_entropy - conditionalEntropy(features[i], feature_categories[i].size(), labels, n_labels);
        if(mutual_info > mutual_information_max) {
          feature_id = i;
          mutual_information_max = mutual_info;
        }
      }
    }
    return feature_id;
  }

  // search for index of label with the highest count
  size_t get_max_index(std::vector<double> & label_counts) {
    double max_count = label_counts[0];
    size_t index = 0;
    for(size_t i = 1; i < n_labels; i++) {
      // tie breaker by lexicographical order of labels
      if(label_counts[i] > max_count || (label_counts[i] == max_count && feature_categories.back().at(i) > feature_categories.back().at(index))) {
        index = i;
        max_count = label_counts[i];
      }
    }
    return index;
  }
  
  // Node training
  node* trainNode(node* my_node) {
    bool isleaf = (my_node->current_depth == max_depth); // check if max_depth of tree is reached
    // select best feature to split on
    // if no feature gives positive information gain, the default value -1 will be returned
    // if the above happens then this node is turned to a leaf
    size_t n_examples = my_node->features[0].size();
    int feature_id = (isleaf ? -1 : selectBestFeatureToSplit(my_node->features, my_node->features_considered));
    isleaf = (feature_id < 0);
    if (isleaf)  {
      // this node is a leaf
      // no further training allowed, perform majority vote here
      std::vector<double> label_counts(n_labels, 0.);
      std::vector<size_t> labels = my_node->features.back();
      for(size_t i = 0; i < n_examples; i++) label_counts[labels[i]]++;
      my_node->category = get_max_index(label_counts);
    }
    else {
      // a feature is selected, split the data according to this feature
      my_node->split_index = feature_id;
      size_t n_feature_cat = feature_categories[feature_id].size();
      std::vector<std::vector<std::vector<size_t> > > features_splitted(n_feature_cat, std::vector<std::vector<size_t> >(n_features));
      for(size_t i_ex = 0; i_ex < n_examples; i_ex++) {
        size_t feature_cat = my_node->features[feature_id][i_ex];
        for(size_t i_feature = 0; i_feature < n_features; i_feature++) {
          features_splitted[feature_cat][i_feature].push_back(my_node->features[i_feature][i_ex]);
        }
      }
      // create and train the vector of subtrees
      for(size_t i_cat = 0; i_cat < n_feature_cat; i_cat++) {
        std::unordered_set<int> features_considered_subtrees = my_node->features_considered;
        features_considered_subtrees.insert(feature_id);
        // recursively train the child node
        node* child = trainNode(newNode(my_node->current_depth + 1, features_splitted[i_cat], features_considered_subtrees));
        // insert the child node into subtree vector
        my_node->subtrees.push_back(child);
      }
    }
    // training finished!
    return my_node;
  }

  void printTree(std::ostream & out, node* my_node) {
    // print the trained tree starting at my_node
    std::vector<size_t> label_count(n_labels, 0);
    std::vector<size_t> labels = my_node->features.back();
    for(size_t i = 0; i < labels.size(); i++) label_count[labels[i]]++;
    out << "[ ";
    for(size_t i = 0; i < n_labels; i++) out << label_count[i] << " " << feature_categories.back().at(i) << (i + 1 < n_labels ? "/" : "");
    out << "]" << std::endl;
    if(my_node->category < 0) {
      int index = my_node->split_index;
      for(size_t i_cat = 0; i_cat < my_node->subtrees.size(); i_cat++) {
        for(size_t i = 0; i <= my_node->current_depth; i++) out << "| ";
        out << feature_names[index] << " = " << feature_categories[index][i_cat] << ": ";
        printTree(out, my_node->subtrees[i_cat]);
      }
    }
  }

  void printPredictions(std::ostream & out, std::vector<size_t> & predictions, std::vector<std::string> & label_categories) {
    for(size_t i = 0; i < predictions.size(); i++) {
      out << label_categories[predictions[i]] << std::endl;
    }
  }

};

int main(int argc, char** argv) {
  if(argc != 7) return print_help(argv);
  std::string train_input = std::string(argv[1]);
  std::string test_input = std::string(argv[2]);
  size_t max_depth = atoi(argv[3]);
  std::string train_output = std::string(argv[4]);
  std::string test_output = std::string(argv[5]);
  std::string metrics_output = std::string(argv[6]);

  MyClassifier c = MyClassifier(train_input, test_input, max_depth);
  c.train();
  c.predictAll();
  std::ofstream f_train(train_output.c_str());
  std::ofstream	f_test(test_output.c_str());
  std::ofstream f_metrics(metrics_output.c_str());
  c.printPredictions(f_train, f_test);
  c.printError(f_metrics);
  f_train.close();
  f_test.close();
  f_metrics.close();
  c.printError(std::cout);
  //c.printTree(std::cout);
  return 0;
}
