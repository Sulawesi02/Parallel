#include <iostream>
#include <ctime>
#include <cstdlib>
#include<nmmintrin.h>  // SSE 4,2

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

// SSE对齐
void sse_align(float** A, float* b, int n) {
    // 消元过程
    for (int k = 0; k < n; k++) {
        __m128 diagonal = _mm_set1_ps(A[k][k]);
        A[k][k] = 1.0f; // 消元后设置对角线元素为1.0f
        // 处理除法部分的对齐
        int j = k + 1;
        while (j % 4 != 0) { // 确保j是4的倍数
            A[k][j] /= A[k][k];
            j++;
        }
        // 使用SSE进行向量化除法
        for (; j + 4 <= n; j += 4) {
            __m128 row_k = _mm_load_ps(&A[k][j]);
            __m128 row_k_div = _mm_div_ps(row_k, diagonal);
            _mm_store_ps(&A[k][j], row_k_div);
        }
        // 处理剩余的标量除法
        for (; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        // 更新b向量
        for (int i = k + 1; i < n; i++) {
            __m128 factor = _mm_set1_ps(A[i][k]);
            A[i][k] = 0.0f; // 消元
            // 处理减法部分的对齐
            j = k + 1;
            while (j % 4 != 0) {
                A[i][j] -= A[i][k] * A[k][j];
                j++;
            }
            // 使用SSE进行向量化减法
            for (; j + 4 <= n; j += 4) {
                __m128 row_i = _mm_load_ps(&A[i][j]);
                __m128 row_k = _mm_load_ps(&A[k][j]);
                __m128 row_i_sub = _mm_sub_ps(row_i, _mm_mul_ps(row_k, factor));
                _mm_store_ps(&A[i][j], row_i_sub);
            }
            // 处理剩余的标量减法
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            // 更新b向量
            b[i] -= A[i][k] * b[k];
        }
    }
    // 回代过程（标量操作）
    float* x = new float[n];
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
}

int main() {
    float** A;
    float* b;

    // 对不同的数据规模进行测试
    for (int sizes : {500, 1000, 2000, 5000}) {
        reset(A, b, sizes);

        clock_t start = clock(); // 开始时间
        sse_align(A, b, sizes); // 执行高斯消去法
        clock_t end = clock(); // 结束时间

        float time_taken = float(end - start) / CLOCKS_PER_SEC; // 计算时间差
        cout << "Sizes: " << sizes << ", Time taken: " << time_taken << " seconds" << endl;

    }

    return 0;
}
