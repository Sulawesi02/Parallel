#include <iostream>
#include <ctime>
#include <cstdlib>

const int N = 5000; // �����������С

using namespace std;

// ������������������ȷ���Խ���Ԫ��Ϊ1.0
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

// ��ͨ��˹��ȥ��������Է�����
void normal(float** A, float* b, int n) {
    //��Ԫ����
    for (int k = 0; k < n; k++) {
        float factor;
        for (int i = k + 1; i < n; i++) {
            factor = A[i][k] / A[k][k];
            for (int j = k; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    // �ش�����
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
        normal(A, b, sizes); // ִ�и�˹��ȥ��
        clock_t end = clock(); // ����ʱ��

        float time_taken = float(end - start) / CLOCKS_PER_SEC; // ����ʱ���
        cout << "Sizes: " << sizes << ", Time taken: " << time_taken << " seconds" << endl;

    }

    return 0;
}
