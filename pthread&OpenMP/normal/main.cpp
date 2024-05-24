#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

float** A;
float* b;
int n; // 定义矩阵大小

void reset(float**& A, float*& b, int n) {
    A = new float* [n];
    b = new float[n];
    srand(time(nullptr)); // 初始化随机数种子
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        A[i][i] = 1.0;
        for (int j = i; j < n; j++)
            A[i][j] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int k = 0; k < n; k++)
        for (int i = k + 1; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i][j] += A[k][j];
    // 生成随机向量b
    for (int i = 0; i < n; i++) {
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}
// 平凡算法
void normal(float** A, float* b, int n) {
    for (int k = 0; k < n; k++) {
        //除法操作
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        //消去操作
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
int main() {
    // 对不同的数据规模进行测试
    for (int sizes : {500, 1000, 2000, 5000}) {
        n = sizes;
        reset(A, b, sizes);

        auto start_time = high_resolution_clock::now();// 开始时间
        normal(A, b, sizes); // 执行平凡高斯消去法
        auto end_time = high_resolution_clock::now();// 结束时间
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "Sizes: " << sizes << ", 平凡算法耗时: " << duration << " ms" << endl;

    }
    return 0;
}
