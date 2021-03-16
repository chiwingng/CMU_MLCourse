//--------------------------------------//
//             feature.cpp              //
// Author: Chi Wing Ng                  //
// Date:   March 14, 2021               //
// C++ program to extract features from //
// a set of input files                 //
//--------------------------------------//

// IO includes
#include <iostream>
#include <fstream>
#include <sstream>

// data structures
#include <vector>
#include <string>
#include <unordered_map>
#include <climits>

// input conversion
#include <cstdlib> // atoi

int printUsage(char** argv) {
  std::cout << "Usage of this program:" << std::endl;
  std::cout << argv[0] << " <train input> <validation input> <test input> <dict input> <formatted train out> <formatted validation out> <formatted test out> <feature flag>" << std::endl;
  std::cout << "feature flag 1 - bag of words; 2 - trimmed bag of words" << std::endl;
  return 1;
}

class FeatureExtraction {
public:
  // constructor
  FeatureExtraction(std::ifstream & dictfile, int model_id);

  // function to extract features using the dictionary provided
  void extractFeatures(std::ifstream & input_file, std::ofstream & output_file);

  // accessors
  unsigned int getWordsInDict();
  int getThreshold();
  
private:
  // dictionary for the features
  std::unordered_map<std::string, int> dict;
  // threshold for the trimmed bag of words model (4 if model_id == 2, INT_MAX if model_id == 1)
  int threshold;
};

FeatureExtraction::FeatureExtraction(std::ifstream & dictfile, int model_id) {
  threshold = (model_id == 1 ? INT_MAX : 4);
  if(!dictfile) {
    std::cout << "Please specify a valid input dictionary file!" << std::endl;
  }
  std::string line, word;
  int index;
  while(std::getline(dictfile, line)) {
    std::istringstream iss(line);
    iss >> word >> index;
    if(dict.find(word) == dict.end()) dict[word] = index;
    else {
      std::cout << "Warning! " << word << "appears more than once in the dictionary input!" << std::endl;
      std::cout << "Registered as index " << dict[word] << " but new entry at index " << index << std::endl;
    }// end if dict find word
  }// getline loop
}

void FeatureExtraction::extractFeatures(std::ifstream & input_file, std::ofstream & output_file) {
  std::string line, temp;
  while(std::getline(input_file, line)) {
    std::istringstream iss(line);
    // pass in the label in the front of line and write in output file
    if(!(iss >> temp)) break;
    output_file << temp << "\t";
    // pass in the rest of the line into a vector of strings
    std::vector<std::string> words;
    std::vector<int> count(dict.size());
    while(iss >> temp) words.push_back(temp);
    std::vector<bool> wordinDict(words.size());
    for(unsigned int i = 0; i < words.size(); i++) {
      if(dict.find(words[i]) != dict.end()) {
        count[dict[words[i]]]++;
        // record the existence of word in dict without double counting
        wordinDict[i] = (count[dict[words[i]]] == 1);
      } // check if word in dictionary
    } // count words loop
    for(unsigned int i = 0; i < words.size(); i++) {
      if(wordinDict[i] && count[dict[words[i]]] < threshold) {
        output_file << dict[words[i]] << ":1\t";
      } // end if word in set
    } // print formatted output loop
    output_file << std::endl;
  } // getline loop
}

inline unsigned int FeatureExtraction::getWordsInDict() {return dict.size();}

inline int FeatureExtraction::getThreshold() {return threshold;}

int main(int argc, char** argv) {
  if(argc != 9) return printUsage(argv);
  std::vector<std::ifstream> inputs;
  std::vector<std::ofstream> outputs;
  for(unsigned int i = 1; i <= 3; i++) inputs.push_back(std::ifstream(argv[i], std::ifstream::in));
  for(unsigned int i = 5; i <= 7; i++) outputs.push_back(std::ofstream(argv[i], std::ofstream::out));
  std::ifstream dict_input(argv[4], std::ifstream::in);
  int model_id = atoi(argv[8]);
  std::vector<std::string> ionames{"train", "valid", "test"};
  
  FeatureExtraction myFeatureExtraction(dict_input, model_id);
  // check for valid input/output files
  if(myFeatureExtraction.getWordsInDict() == 0) return printUsage(argv);
  for(unsigned int i = 0; i < ionames.size(); i++) {
    if(!inputs[i] || !outputs[i]) {
      std::cout << "Please provide valid " << ionames[i] << " paths" << std::endl;
      return printUsage(argv);
    }
    myFeatureExtraction.extractFeatures(inputs[i], outputs[i]);
  }// dataset loops

  return 0;
}
