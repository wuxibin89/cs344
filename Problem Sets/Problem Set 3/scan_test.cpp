#include <iostream>
#include <random>

using namespace std;

void HS_scan(unsigned int *const h_cdf, int numBins);

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " nelems" << endl;
    return -1;
  }

  int nelems = std::stoi(argv[1]);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 100);

  uint32_t *data = new uint32_t[nelems];
  uint32_t *in_scan = new uint32_t[nelems]; // inclusive_scan
  uint32_t *ex_scan = new uint32_t[nelems]; // exclusive_scan
  uint32_t in_init = 0, ex_init = 0;
  for (int i = 0; i < nelems; ++i) {
    data[i] = dis(gen);

    in_init += data[i];
    in_scan[i] = in_init;

    ex_scan[i] = ex_init;
    ex_init += data[i];
  }

  HS_scan(data, nelems);
  for (int i = 0; i < nelems; ++i) {
    if (data[i] != in_scan[i]) {
      std::cerr << "[" << i << "] truth: " << in_scan[i]
                << ", data: " << data[i] << endl;
      return -1;
    }
  }

  return 0;
}