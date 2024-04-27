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

// AVX-512δ����
void avx512_unalign(float** A, float* b, int n) {
    // ��Ԫ����
    for (int k = 0; k < n; k++) {
        __m512 div; // ���ڴ洢����
        float factor = A[k][k]; // ����Խ���Ԫ�ص�ֵ
        div = _mm512_set1_ps(1.0f / factor); // ��������ĵ���
        // ʹ��AVX-512ָ���������������
        for (int i = k + 1; i < n; i++) {
            // ����factor���������A[k][k]��Ϊ0
            factor = A[i][k] * factor; // ʹ��Ԥ�ȼ���ĵ���
            // ��������������
            int j = n - 16;
            for (; j >= k; j -= 16) {
                __m512 t1 = _mm512_loadu_ps(&A[k][j]);
                __m512 t2 = _mm512_loadu_ps(&A[i][j]);
                __m512 t3 = _mm512_mul_ps(t1, div);
                t2 = _mm512_sub_ps(t2, t3);
                _mm512_storeu_ps(&A[i][j], t2);
            }
            // ����ʣ��ı�������
            for (; j > k; j--) {
                A[i][j] -= factor * A[k][j];
            }
            // ����b����
            b[i] -= factor * b[k];
        }
        // ��A[k][k]����Ϊ1.0f����Ϊ�Ѿ��õ������ĵ���
        A[k][k] = 1.0f;
    }
    // �ش�����
    float* x = new float[n];
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]; // �������һ��Ԫ��
    // ʹ�ñ����������лش�
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i]; // �ش��õ�x[i]
    }
}

int main() {
    float** A;
    float* b;

    // �Բ�ͬ�����ݹ�ģ���в���
    for (int sizes : {500, 1000, 2000, 5000}) {
        reset(A, b, sizes);

        clock_t start = clock(); // ��ʼʱ��
        avx512_unalign(A, b, sizes); // ִ�и�˹��ȥ��
        clock_t end = clock(); // ����ʱ��

        float time_taken = float(end - start) / CLOCKS_PER_SEC; // ����ʱ���
        cout << "Sizes: " << sizes << ", Time taken: " << time_taken << " seconds" << endl;

    }

    return 0;
}
