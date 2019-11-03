#include <iostream>
#include <random>

using namespace std;

void HS_scan(unsigned int *const h_cdf, int numBins, int block_size);
void Blelloch_scan(unsigned int *const h_cdf, int numBins, int block_size);

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " type nelems block_size" << endl;
    return -1;
  }

  int type = std::stoi(argv[1]);
  int nelems = std::stoi(argv[2]);
  int block_size = std::stoi(argv[3]);

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

  uint32_t *truth;
  switch (type) {
  case 0:
    HS_scan(data, nelems, block_size);
    truth = in_scan;
    break;
  case 1:
    Blelloch_scan(data, nelems, block_size);
    truth = ex_scan;
  default:
    break;
  }

  for (int i = 0; i < nelems; ++i) {
    if (data[i] != truth[i]) {
      std::cerr << "[" << i << "] truth: " << in_scan[i]
                << ", data: " << data[i] << endl;
      return -1;
    }
  }

  return 0;
}