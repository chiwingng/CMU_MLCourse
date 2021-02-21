#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <unordered_map>

int print_help(char** argv) {
  std::cout << "Usage of program : " << std::endl;
  std::cout << argv[0] << " <input file> <output_file>" << std::endl;
  std::cout << argv[0] << " small_train.tsv small_inspect.txt" << std::endl;
  return 1;
}

void output(std::ostream & out, const double & entropy, const double & error) {
  out << "entropy: " << std::setprecision(12) << std::fixed << entropy << std::endl;
  out << "error: " << std::setprecision(12) << std::fixed << error << std::endl;
}

int main(int argc, char** argv) {
  if(argc != 3) return print_help(argv);
  const char* input_file = argv[1];
  const char* output_file = argv[2];

  std::ifstream inFile(input_file);
  std::istringstream iss;
  std::string line, temp;

  // process first line to retrieve feature and label names (label is treated as one of the features, located at the back of vectors)
  std::getline(inFile, line);
  iss.str(line);
  std::vector<std::string> feature_names;
  while(iss >> temp) {
    feature_names.push_back(temp);
  }
  size_t n_features = feature_names.size();
  
  // read in data
  std::vector< std::vector<size_t> > feature_values(n_features);
  std::vector< std::unordered_map<std::string, size_t> > feature_indices(n_features);
  while(std::getline(inFile, line)) {
    std::istringstream iss1;
    iss1.str(line);
    size_t feature_index = 0;
    while(iss1 >> temp) {
      if (feature_indices[feature_index].find(temp) == feature_indices[feature_index].end()) {
        size_t n_feature_values = feature_indices[feature_index].size();
        feature_indices[feature_index][temp] = n_feature_values;
      }
      feature_values[feature_index].push_back(feature_indices[feature_index][temp]);
      feature_index++;
    }
  }

  // count label categories
  std::vector<size_t> count_labels(feature_indices.back().size(), 0);
  std::vector<size_t> label_values = feature_values.back();
  for(size_t i = 0; i < label_values.size(); i++) {
    count_labels[label_values[i]]++;
  }

  // compute entropy and error rate of majority vote
  size_t total_examples = std::accumulate(count_labels.begin(), count_labels.end(), 0);
  size_t max_label = std::max_element(count_labels.begin(), count_labels.end()) - count_labels.begin();
  double entropy = 0, error = 1. - double(*(count_labels.begin() + max_label))/double(total_examples);
  for(size_t i = 0; i < count_labels.size(); i++) {
    double p_i = double(count_labels[i])/double(total_examples);
    entropy -= p_i * log2(p_i);
  }

  // print output
  output(std::cout, entropy, error);
  std::ofstream outfile(output_file);
  if(outfile) output(outfile, entropy, error);
  outfile.close();
  
  return 0;
}
