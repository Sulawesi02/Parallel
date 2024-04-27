#include <iostream>
#include <ctime>
#include <cstdlib>
#include <immintrin.h> //AVX、AVX2

using namespace std;

const int N = 5000; // 定义最大矩阵大小

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

// AVX未对齐
void avx_unalign(float** A, float* b, int n) {
    // 消元过程
    for (int k = 0; k < n; k++) {
        __m256 div; // 用于存储除数
        float factor = A[k][k]; // 保存对角线元素的值
        div = _mm256_set1_ps(1.0f / factor); // 计算除数的倒数
        // 使用AVX指令集进行向量化操作
        for (int i = k + 1; i < n; i++) {
            // 计算factor，这里假设A[k][k]不为0
            factor = A[i][k] * factor; // 使用预先计算的倒数
            // 向量化减法操作
            int j = n - 8;
            for (; j >= k; j -= 8) {
                __m256 t1 = _mm256_loadu_ps(&A[k][j]);
                __m256 t2 = _mm256_loadu_ps(&A[i][j]);
                __m256 t3 = _mm256_mul_ps(t1, div);
                t2 = _mm256_sub_ps(t2, t3);
                _mm256_storeu_ps(&A[i][j], t2);
            }
            // 处理剩余的标量操作
            for (; j > k; j--) {
                A[i][j] -= factor * A[k][j];
            }
            // 更新b向量
            b[i] -= factor * b[k];
        }
        // 将A[k][k]设置为1.0f，因为已经用到了它的倒数
        A[k][k] = 1.0f;
    }
    // 回代过程
    float* x = new float[n];
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]; // 计算最后一个元素
    // 使用标量操作进行回代
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i]; // 回代得到x[i]
    }
}

int main() {
    float** A;
    float* b;

    // 对不同的数据规模进行测试
    for (int sizes : {500, 1000, 2000, 5000}) {
        reset(A, b, sizes);

        clock_t start = clock(); // 开始时间
        avx_unalign(A, b, sizes); // 执行高斯消去法
        clock_t end = clock(); // 结束时间

        float time_taken = float(end - start) / CLOCKS_PER_SEC; // 计算时间差
        cout << "Sizes: " << sizes << ", Time taken: " << time_taken << " seconds" << endl;

    }

    return 0;
}
