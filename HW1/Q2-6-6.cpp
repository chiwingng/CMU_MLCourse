#include <iostream>

void print_help(char** argv) {
  std::cout << "usage of this program: " << argv[0] << " [int n to evaluate Ln]" << std::endl;
}

long L(int n) {
  if(n == 0) return 2;
  if(n == 1) return 1;
  return L(n-1) + L(n-2);
}

int main(int argc, char** argv) {
  if(argc != 2) {
    print_help(argv);
    return 1;
  }
  int n = atoi(argv[1]);
  std::cout << "L_" << n << " = " << L(n) << std::endl;
  return 0;
}
