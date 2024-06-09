%%writefile lab/vector_add.cpp
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
    const int N = 256;
    
    // Initialize a vector and print values
    std::vector<int> vector1(N, 10);
    std::cout << "\nInput Vector1: ";    
    for (int i = 0; i < N; i++) std::cout << vector1[i] << " ";

    // Create second vector, initialize to 20 and print values
    std::vector<int> vector2(N, 20);
    std::cout << "\nInput Vector2: ";    
    for (int i = 0; i < N; i++) std::cout << vector2[i] << " ";

    // Create Buffer
    buffer vector1_buffer(vector1);
    buffer vector2_buffer(vector2);

    // Submit task to add vector
    queue q;
    q.submit([&](handler &h) {
        // Create accessor for vector1_buffer
        accessor vector1_accessor(vector1_buffer, h);
        
        // Add second accessor for second buffer
        accessor vector2_accessor(vector2_buffer, h);

        h.parallel_for(range<1>(N), [=](id<1> index) {
            // Add the second vector to first one
            vector1_accessor[index] += vector2_accessor[index];
        });
    });

    // Create a host accessor to copy data from device to host
    host_accessor h_a(vector1_buffer, read_only);

    // Print Output values 
    std::cout << "\nOutput Values: ";
    for (int i = 0; i < N; i++) std::cout << vector1[i] << " ";
    std::cout << "\n";

    return 0;
}