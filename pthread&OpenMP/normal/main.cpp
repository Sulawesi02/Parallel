#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

float** A;
float* b;
int n; // ��������С

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
// ƽ���㷨
void normal(float** A, float* b, int n) {
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
int main() {
    // �Բ�ͬ�����ݹ�ģ���в���
    for (int sizes : {500, 1000, 2000, 5000}) {
        n = sizes;
        reset(A, b, sizes);

        auto start_time = high_resolution_clock::now();// ��ʼʱ��
        normal(A, b, sizes); // ִ��ƽ����˹��ȥ��
        auto end_time = high_resolution_clock::now();// ����ʱ��
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��ʱ: " << duration << " ms" << endl;

    }
    return 0;
}
