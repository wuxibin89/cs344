#include <algorithm>
#include <iostream>
#include <random>

void radix_sort_test(const unsigned *h_inputVals, const unsigned *h_inputPos,
                     unsigned *h_outputVals, unsigned *h_outputPos,
                     int numElems);

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "usage: " << argv[0] << " numElems" << std::endl;
    return -1;
  }

  int numElems = std::stoi(argv[1]);
  unsigned *h_inputVals = new unsigned[numElems];
  unsigned *h_inputPos = new unsigned[numElems];
  unsigned *h_outputVals = new unsigned[numElems];
  unsigned *h_outputPos = new unsigned[numElems];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned> dis;
  for (int i = 0; i < numElems; ++i) {
    h_inputVals[i] = dis(gen);
    h_inputPos[i] = i;
  }

  radix_sort_test(h_inputVals, h_inputPos, h_outputVals, h_outputPos, numElems);

  std::sort(h_inputVals, h_inputVals + numElems);
  for (int i = 0; i < numElems; ++i) {
    if (h_inputVals[i] != h_outputVals[i]) {
      std::cout << "unexpected value, idx: " << i
                << ", truth: " << h_inputVals[i]
                << ", output: " << h_outputVals[i] << std::endl;
      return -1;
    }
  }

  return 0;
}