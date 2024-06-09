#include <iostream>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

int N = 5000; // 定义最大矩阵大小
float** A;
#define NUM_THREADS 7 // 定义线程数量

void reset() {
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
// 平凡算法
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
// sse优化
void sse() {
    for (int k = 0; k < N; k++) {
        __m128 div = _mm_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= N; j += 4) {
            __m128 row_k = _mm_loadu_ps(&A[k][j]);
            row_k = _mm_div_ps(row_k, div);
            _mm_storeu_ps(&A[k][j], row_k);
        }
        for (; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
// AVX优化
void avx() {
    for (int k = 0; k < N; k++) {
        __m256 div = _mm256_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[k][j]);
            va = _mm256_div_ps(va, div);
            _mm256_storeu_ps(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t1 = _mm256_loadu_ps(&A[k][j]);
                __m256 t2 = _mm256_loadu_ps(&A[i][j]);
                __m256 t3 = _mm256_mul_ps(t1, vaik);
                t2 = _mm256_sub_ps(t2, t3);
                _mm256_storeu_ps(&A[i][j], t2);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//==================================================================
//平凡算法静态调度
void normal_omp_static() {
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < N; j++) {
                A[k][j] /= tmp;
            }
            A[k][k] = 1.0;
        }

#pragma omp for
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//sse优化静态调度
void sse_omp_static() {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m128 div = _mm_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 4 <= N; j += 4) {
                __m128 row_k = _mm_loadu_ps(&A[k][j]);
                row_k = _mm_div_ps(row_k, div);
                _mm_storeu_ps(&A[k][j], row_k);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(static)
        for (i = k + 1; i < N; i++) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//avx优化静态调度
void avx_omp_static() {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(static)
        for (i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//==================================================================
//平凡算法动态调度
void normal_omp_dynamic() {
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < N; j++) {
                A[k][j] /= tmp;
            }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//sse优化动态调度
void sse_omp_dynamic() {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m128 div = _mm_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 4 <= N; j += 4) {
                __m128 row_k = _mm_loadu_ps(&A[k][j]);
                row_k = _mm_div_ps(row_k, div);
                _mm_storeu_ps(&A[k][j], row_k);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//avx优化动态调度
void avx_omp_dynamic() {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//==================================================================
//平凡算法向导调度
void normal_omp_guided() {
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < N; j++) {
                A[k][j] /= tmp;
            }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//sse优化向导调度
void sse_omp_guided() {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m128 div = _mm_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 4 <= N; j += 4) {
                __m128 row_k = _mm_loadu_ps(&A[k][j]);
                row_k = _mm_div_ps(row_k, div);
                _mm_storeu_ps(&A[k][j], row_k);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//avx优化向导调度
void avx_omp_guided() {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

int main() {
    // 对不同的数据规模进行测试
    for (int sizes : {500, 1000, 1500, 2000}) {
        N = sizes;

        reset();
        auto start_time = high_resolution_clock::now();// 开始时间
        normal(); // 执行平凡高斯消去法
        auto end_time = high_resolution_clock::now();// 结束时间
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "平凡算法耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        sse(); // 执行sse优化高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "sse优化耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        avx(); // 执行avx优化高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "avx优化耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        normal_omp_static(); // 平凡算法静态调度
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "平凡算法静态调度耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        sse_omp_static(); // 执行sse优化静态调度
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "sse优化静态调度耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        avx_omp_static(); // 执行avx优化静态调度
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "avx优化静态调度耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        normal_omp_dynamic(); // 平凡算法动态调度
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "平凡算法动态调度耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        sse_omp_dynamic(); // sse优化动态调度
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "sse优化动态调度耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        avx_omp_dynamic(); // avx优化动态调度
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "avx优化动态调度耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        normal_omp_guided(); // 平凡算法向导调度
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "平凡算法向导调度耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        sse_omp_guided(); // sse优化向导调度
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "sse优化向导调度耗时: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // 开始时间
        avx_omp_guided(); // avx优化向导调度
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "avx优化向导调度耗时: " << duration << " ms" << endl;
    }
    return 0;
}
