#include <iostream>
#include <ctime>
#include <cstdlib>
#include <immintrin.h> // AVX-512

using namespace std;

const int N = 5000; // �����������С

void reset(float**& A, float*& b, int n) {
    A = new float* [n];
    b = new float[n];
    srand(time(nullptr)); // ��ʼ�����������
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
    // �����������b
    for (int i = 0; i < n; i++) {
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// AVX-512����
void avx512_align(float** A, float* b, int n) {
    // ��Ԫ����
    for (int k = 0; k < n; k++) {
        __m512 diagonal = _mm512_set1_ps(A[k][k]);
        A[k][k] = 1.0f; // ��Ԫ�����öԽ���Ԫ��Ϊ1.0f
        // ����������ֵĶ���
        int j = k + 1;
        while (j % 16 != 0) { // ȷ��j��16�ı�������ӦAVX-512��512λ�Ĵ���
            A[k][j] /= A[k][k];
            j++;
        }
        // ʹ��AVX-512��������������
        for (; j + 16 <= n; j += 16) {
            __m512 row_k = _mm512_load_ps(&A[k][j]);
            __m512 row_k_div = _mm512_div_ps(row_k, diagonal);
            _mm512_store_ps(&A[k][j], row_k_div);
        }
        // ����ʣ��ı�������
        for (; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        // ����b����
        for (int i = k + 1; i < n; i++) {
            __m512 factor = _mm512_set1_ps(A[i][k]);
            A[i][k] = 0.0f; // ��Ԫ
            // ����������ֵĶ���
            j = k + 1;
            while (j % 16 != 0) {
                A[i][j] -= A[i][k] * A[k][j];
                j++;
            }
            // ʹ��AVX-512��������������
            for (; j + 16 <= n; j += 16) {
                __m512 row_i = _mm512_load_ps(&A[i][j]);
                __m512 row_k = _mm512_load_ps(&A[k][j]);
                __m512 row_i_sub = _mm512_sub_ps(row_i, _mm512_mul_ps(row_k, factor));
                _mm512_store_ps(&A[i][j], row_i_sub);
            }
            // ����ʣ��ı�������
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            // ����b����
            b[i] -= A[i][k] * b[k];
        }
    }
    // �ش����̣�����������AVX-512�����ã�
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

    // �Բ�ͬ�����ݹ�ģ���в���
    for (int sizes : {500, 1000, 2000, 5000}) {
        reset(A, b, sizes);

        clock_t start = clock(); // ��ʼʱ��
        avx512_align(A, b, sizes); // ִ�и�˹��ȥ��
        clock_t end = clock(); // ����ʱ��

        float time_taken = float(end - start) / CLOCKS_PER_SEC; // ����ʱ���
        cout << "Sizes: " << sizes << ", Time taken: " << time_taken << " seconds" << endl;

    }

    return 0;
}
