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

float** A;
int n; // ��������С
#define NUM_THREADS 7 // �����߳�����

void reset(float**& A, int n) {
    A = new float* [n];
    srand(time(nullptr)); // ��ʼ�����������
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        A[i][i] = 1.0;
        for (int j = i; j < n; j++)
            A[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (int k = 0; k < n; k++)
        for (int i = k + 1; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i][j] += A[k][j];
}
void clear(int n) {
    for (int i = 0; i < n; i++) {
        delete[] A[i];
    }
    delete A;
}
// ƽ���㷨
void normal(float**& A, int n) {
    for (int k = 0; k < n; k++) {
        //��������
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        //��ȥ����
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
// sse�Ż�
void sse(float**& A, int n) {
    for (int k = 0; k < n; k++) {
        __m128 div = _mm_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= n; j += 4) {
            __m128 row_k = _mm_loadu_ps(&A[k][j]);
            row_k = _mm_div_ps(row_k, div);
            _mm_storeu_ps(&A[k][j], row_k);
        }
        for (; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
// AVX�Ż�
void avx(float**& A, int n) {
    for (int k = 0; k < n; k++) {
        __m256 div = _mm256_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= n; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[k][j]);
            va = _mm256_div_ps(va, div);
            _mm256_storeu_ps(&A[k][j], va);
        }
        for (; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 t1 = _mm256_loadu_ps(&A[k][j]);
                __m256 t2 = _mm256_loadu_ps(&A[i][j]);
                __m256 t3 = _mm256_mul_ps(t1, vaik);
                t2 = _mm256_sub_ps(t2, t3);
                _mm256_storeu_ps(&A[i][j], t2);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//==================================================================
//ƽ���㷨��̬����
void normal_omp_static(float**& A, int n) {
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < n; j++) {
                A[k][j] /= tmp;
            }
            A[k][k] = 1.0;
        }

#pragma omp for
        for (i = k + 1; i < n; i++) {
            for (j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//sse�Ż���̬����
void sse_omp_static(float**& A, int n) {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            __m128 div = _mm_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 row_k = _mm_loadu_ps(&A[k][j]);
                row_k = _mm_div_ps(row_k, div);
                _mm_storeu_ps(&A[k][j], row_k);
            }
            for (; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(static)
        for (i = k + 1; i < n; i++) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//avx�Ż���̬����
void avx_omp_static(float**& A, int n) {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(static)
        for (i = k + 1; i < n; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//==================================================================
//ƽ���㷨��̬����
void normal_omp_dynamic(float**& A, int n) {
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < n; j++) {
                A[k][j] /= tmp;
            }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < n; i++) {
            for (j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//sse�Ż���̬����
void sse_omp_dynamic(float**& A, int n) {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            __m128 div = _mm_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 row_k = _mm_loadu_ps(&A[k][j]);
                row_k = _mm_div_ps(row_k, div);
                _mm_storeu_ps(&A[k][j], row_k);
            }
            for (; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < n; i++) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//avx�Ż���̬����
void avx_omp_dynamic(float**& A, int n) {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < n; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//==================================================================
//ƽ���㷨�򵼵���
void normal_omp_guided(float**& A, int n) {
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < n; j++) {
                A[k][j] /= tmp;
            }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(guided)
        for (i = k + 1; i < n; i++) {
            for (j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//sse�Ż��򵼵���
void sse_omp_guided(float**& A, int n) {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            __m128 div = _mm_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 row_k = _mm_loadu_ps(&A[k][j]);
                row_k = _mm_div_ps(row_k, div);
                _mm_storeu_ps(&A[k][j], row_k);
            }
            for (; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < n; i++) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//avx�Ż��򵼵���
void avx_omp_guided(float**& A, int n) {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < n; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

int main() {
    // �Բ�ͬ�����ݹ�ģ���в���
    for (int sizes : {500, 1000, 1500, 2000, 2500, 3000}) {
        n = sizes;

        reset(A, sizes);
        auto start_time = high_resolution_clock::now();// ��ʼʱ��
        normal(A, sizes); // ִ��ƽ����˹��ȥ��
        auto end_time = high_resolution_clock::now();// ����ʱ��
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse(A, sizes); // ִ��sse�Ż���˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", sse�Ż���ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx(A, sizes); // ִ��avx�Ż���˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", avx�Ż���ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_omp_static(A, sizes); // ƽ���㷨��̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬���Ⱥ�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_omp_static(A, sizes); // ִ��sse�Ż���̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", sse�Ż���̬���Ⱥ�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_omp_static(A, sizes); // ִ��avx�Ż���̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", avx�Ż���̬���Ⱥ�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_omp_dynamic(A, sizes); // ƽ���㷨��̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬���Ⱥ�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_omp_dynamic(A, sizes); // sse�Ż���̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", sse�Ż���̬���Ⱥ�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_omp_dynamic(A, sizes); // avx�Ż���̬����
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", avx�Ż���̬���Ⱥ�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_omp_guided(A, sizes); // ƽ���㷨�򵼵���
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨�򵼵��Ⱥ�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_omp_guided(A, sizes); // sse�Ż��򵼵���
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", sse�Ż��򵼵��Ⱥ�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_omp_guided(A, sizes); // avx�Ż��򵼵���
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", avx�Ż��򵼵��Ⱥ�ʱ: " << duration << " ms" << endl;
        clear(n);
    }
    return 0;
}
