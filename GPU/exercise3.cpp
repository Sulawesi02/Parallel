%%writefile lab/usm_lab.cpp
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <cmath>
using namespace sycl;

static const int N = 1024;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  // Initialize 2 arrays on host
  int *data1 = static_cast<int *>(malloc(N * sizeof(int)));
  int *data2 = static_cast<int *>(malloc(N * sizeof(int)));
  for (int i = 0; i < N; i++) {
    data1[i] = 25;
    data2[i] = 49;
  }
    
  int *data1_device = malloc_device<int>(N, q);
  int *data2_device = malloc_device<int>(N, q);

  q.memcpy(data1_device, data1, N * sizeof(int)).wait();
  q.memcpy(data2_device, data2, N * sizeof(int)).wait();

  q.parallel_for(N, [=](auto i) { 
    data1_device[i] = std::sqrt(data1_device[i]);
  }).wait();

  q.parallel_for(N, [=](auto i) { 
    data2_device[i] = std::sqrt(data2_device[i]);
  }).wait();

  q.parallel_for(N, [=](auto i) { 
    data1_device[i] += data2_device[i];
  }).wait();

  q.memcpy(data1, data1_device, N * sizeof(int)).wait();

  // Verify results
  int fail = 0;
  for (int i = 0; i < N; i++) {
    if(data1[i] != 12) {
      fail = 1;
      break;
    }
  }
  if(fail == 1) std::cout << "FAIL"; else std::cout << "PASS";
  std::cout << "\n";

  free(data1_device, q);
  free(data2_device, q);

  free(data1);
  free(data2);

  return 0;
}