#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace std::chrono;

int N = 5000; // 定义最大矩阵大小
float** A;

void init() {
    A = new float* [N];
    srand(time(nullptr)); // 初始化随机数种子
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
        A[i][i] = 1.0;
        for (int j = i; j < N; j++)
            A[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i][j] += A[k][j];
}

// 普通高斯消去法求解线性方程组
void normal() {    
    for (int k = 0; k < N; k++) {
        //除法操作
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        //消去操作
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

int main() {
    
    int N_values[] = { 500, 1000, 2000, 5000 };
    for (int i = 0; i < sizeof(N_values) / sizeof(N_values[0]); ++i) {
        N = N_values[i];
        cout << "数据规模: " << N << endl;

        init();
        auto start_time = high_resolution_clock::now();// 开始时间
        normal(); 
        auto end_time = high_resolution_clock::now();// 结束时间
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "执行时间: " << duration << " ms" << endl;
    }

    return 0;
}
