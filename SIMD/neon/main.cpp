#include <iostream>
#include <ctime>
#include <cstdlib>
#include <arm_neon.h> //NEON

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


void neon(float** A, float* b, int n) {
    // ��Ԫ����
    for (int k = 0; k < n; k++) {
        float32x4_t vk = vdupq_n_f32(A[k][k]);
        float32x4_t vOneOverK = vdupq_n_f32(1.0f / A[k][k]);

        // ��������������
        for (int i = k + 1; i < n; i += 4) {
            // Load 4 elements from A[k] and A[i]
            float32x4_t vAik = vld1q_f32(&A[i][k]);
            float32x4_t vAkk = vld1q_f32(&A[k][k]);

            // Calculate A[i][k] * A[k][k] and store in A[i][k]
            vAik = vmulq_f32(vAik, vOneOverK);

            // Update A[i][j] for j in [k, n)
            for (int j = k; j < n; j += 4) {
                float32x4_t vAkj = vld1q_f32(&A[k][j]);
                float32x4_t vAij = vld1q_f32(&A[i][j]);
                vAij = vsubq_f32(vAij, vmulq_f32(vAik, vAkj));
                vst1q_f32(&A[i][j], vAij);
            }

            // Update b[i]
            b[i] = b[i] - A[i][k] * b[k];
        }

        // Set A[k][k] to 1.0f after using its reciprocal
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
        neon(A, b, sizes); // ִ�и�˹��ȥ��
        clock_t end = clock(); // ����ʱ��

        float time_taken = float(end - start) / CLOCKS_PER_SEC; // ����ʱ���
        cout << "Sizes: " << sizes << ", Time taken: " << time_taken << " seconds" << endl;

    }

    return 0;
}