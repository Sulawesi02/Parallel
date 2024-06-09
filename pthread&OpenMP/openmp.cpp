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

int N = 5000; // �����������С
float** A;
#define NUM_THREADS 7 // �����߳�����

void reset() {
    A = new float* [N];
    srand(time(nullptr)); // ��ʼ�����������
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
// ƽ���㷨
void normal() {
    for (int k = 0; k < N; k++) {
        //��������
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        //��ȥ����
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
// sse�Ż�
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
// AVX�Ż�
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
//ƽ���㷨��̬����
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
//sse�Ż���̬����
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
//avx�Ż���̬����
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
//ƽ���㷨��̬����
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
//sse�Ż���̬����
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
//avx�Ż���̬����
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
//ƽ���㷨�򵼵���
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
//sse�Ż��򵼵���
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
//avx�Ż��򵼵���
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
    // �Բ�ͬ�����ݹ�ģ���в���
    for (int sizes : {500, 1000, 1500, 2000}) {
        N = sizes;

        reset();
        auto start_time = high_resolution_clock::now();// ��ʼʱ��
        normal(); // ִ��ƽ����˹��ȥ��
        auto end_time = high_resolution_clock::now();// ����ʱ��
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "ƽ���㷨��ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse(); // ִ��sse�Ż���˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "sse�Ż���ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx(); // ִ��avx�Ż���˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "avx�Ż���ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_omp_static(); // ƽ���㷨��̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "ƽ���㷨��̬���Ⱥ�ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_omp_static(); // ִ��sse�Ż���̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "sse�Ż���̬���Ⱥ�ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_omp_static(); // ִ��avx�Ż���̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "avx�Ż���̬���Ⱥ�ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_omp_dynamic(); // ƽ���㷨��̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "ƽ���㷨��̬���Ⱥ�ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_omp_dynamic(); // sse�Ż���̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "sse�Ż���̬���Ⱥ�ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_omp_dynamic(); // avx�Ż���̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "avx�Ż���̬���Ⱥ�ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_omp_guided(); // ƽ���㷨�򵼵���
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "ƽ���㷨�򵼵��Ⱥ�ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_omp_guided(); // sse�Ż��򵼵���
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "sse�Ż��򵼵��Ⱥ�ʱ: " << duration << " ms" << endl;

        reset();
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_omp_guided(); // avx�Ż��򵼵���
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "avx�Ż��򵼵��Ⱥ�ʱ: " << duration << " ms" << endl;
    }
    return 0;
}
