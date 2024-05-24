#include <iostream>
#include <ctime>
#include <cstdlib>
#include <arm_neon.h> //NEON
#include <pthread.h>
#include <semaphore.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

float** A;
int n;                // ��������С
#define NUM_THREADS 7 // �����߳�����

sem_t sem_main; // �ź���
sem_t sem_workstart[NUM_THREADS];
sem_t sem_workend[NUM_THREADS];

sem_t sem_leader;
sem_t sem_Division[NUM_THREADS];
sem_t sem_Elimination[NUM_THREADS];

pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

typedef struct {
     int k; //��ȥ���ִ�
     int t_id; // �߳� id
}threadParam_t;

void reset(float **&A, int n)
{
    A = new float *[n];
    srand(time(nullptr)); // ��ʼ�����������
    for (int i = 0; i < n; i++)
    {
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
//==================================================================
// ƽ���㷨
void normal(float**& A, int n)
{
    for (int k = 0; k < n; k++)
    {
        // ��������
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        // ��ȥ����
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//==================================================================
// neon�Ż�
void neon(float**& A, int n)
{
    for (int k = 0; k < n; k++)
    {
        float32x4_t vt = vdupq_n_f32(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= n; j += 4)
        {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
//==================================================================
void *normal_threadFunc(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int k = p->k;         // ��ȥ���ִ�
    int t_id = p->t_id;   // �̱߳��
    int i = k + t_id + 1; // ��ȡ����

    for (int j = k + 1; j < n; j++)
    {
        A[i][j] -= A[i][k] * A[k][j];
    }

    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}
// ƽ���㷨��̬�߳�
void normal_dynamic(float **&A, int n)
{
    for (int k = 0; k < n; k++)
    {
        // ���߳�����������
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // ���������̣߳�ִ����ȥ����
        int work_count = n - 1 - k; // �����߳�����
        pthread_t *handles = (pthread_t *)malloc(work_count * sizeof(pthread_t));
        threadParam_t *param = (threadParam_t *)malloc(work_count * sizeof(threadParam_t));
        // ��������
        for (int t_id = 0; t_id < work_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        // �����߳�
        for (int t_id = 0; t_id < work_count; t_id++)
        {
            pthread_create(&handles[t_id], NULL, normal_threadFunc, &param[t_id]);
        }
        for (int t_id = 0; t_id < work_count; t_id++)
        {
            pthread_join(handles[t_id], NULL);
        }
        free(handles);
        free(param);
    }
}
//==================================================================
void *neon_threadFunc(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int k = p->k;         // ��ȥ���ִ�
    int t_id = p->t_id;   // �߳�
    int i = k + t_id + 1; // ��ȡ����

    float32x4_t vaik = vdupq_n_f32(A[i][k]);
    int j;
    for (j = k + 1; j + 4 <= n; j += 4)
    {
        float32x4_t vakj = vld1q_f32(&A[k][j]);
        float32x4_t vaij = vld1q_f32(&A[i][j]);
        float32x4_t vx = vmulq_f32(vakj, vaik);
        vaij = vsubq_f32(vaij, vx);
        vst1q_f32(&A[i][j], vaij);
    }
    for (; j < n; j++)
    {
        A[i][j] -= A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}
// neon�Ż��㷨��̬�߳�
void neon_dynamic(float **&A, int n)
{
    for (int k = 0; k < n; k++)
    {
        // ���߳�����������
        float32x4_t vt = vdupq_n_f32(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= n; j += 8)
        {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // ���������̣߳�ִ����ȥ����
        int work_count = n - 1 - k; // �����߳�����
        pthread_t *handles = (pthread_t *)malloc(work_count * sizeof(pthread_t));
        threadParam_t *param = (threadParam_t *)malloc(work_count * sizeof(threadParam_t));
        // ��������
        for (int t_id = 0; t_id < work_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        // �����߳�
        for (int t_id = 0; t_id < work_count; t_id++)
        {
            pthread_create(&handles[t_id], NULL, neon_threadFunc, &param[t_id]);
        }
        for (int t_id = 0; t_id < work_count; t_id++)
        {
            pthread_join(handles[t_id], NULL);
        }
        free(handles);
        free(param);
    }
}
//==================================================================
void *normal_sem_threadFunc(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;

    for (int k = 0; k < n; k++)
    {
        sem_wait(&sem_workstart[t_id]); // �������ȴ����̳߳������

        // ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // ִ����ȥ����
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        sem_post(&sem_main);          // �������߳�
        sem_wait(&sem_workend[t_id]); // �������ȴ����̻߳��ѽ�����һ��
    }
    pthread_exit(NULL);
    return NULL;
}
// ƽ���㷨��̬�߳�+�ź���ͬ��
void normal_sem_static(float **&A, int n)
{
    sem_init(&sem_main, 0, 0); // ��ʼ���ź���
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    // �����߳�
    pthread_t *handles = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t *param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, normal_sem_threadFunc, &param[t_id]);
    }
    for (int k = 0; k < n; k++)
    {
        // ���߳�����������
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // ��ʼ���ѹ����߳�
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workstart[t_id]);
        }
        // ���߳�˯��
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_wait(&sem_main);
        }
        // ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // �����߳�
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void *sem_threadFunc(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;

    for (int k = 0; k < n; k++)
    {
        sem_wait(&sem_workstart[t_id]); // �������ȴ����̳߳������

        // ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // ִ����ȥ����
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8)
            {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0.0;
        }
        sem_post(&sem_main);          // �������߳�
        sem_wait(&sem_workend[t_id]); // �������ȴ����̻߳��ѽ�����һ��
    }
    pthread_exit(NULL);
    return NULL;
}
// neon�Ż���̬�߳�+�ź���ͬ��
void neon_sem_static(float **&A, int n)
{
    sem_init(&sem_main, 0, 0); // ��ʼ���ź���
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    // �����߳�
    pthread_t *handles = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t *param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, sem_threadFunc, &param[t_id]);
    }
    for (int k = 0; k < n; k++)
    {
        // ���߳�����������
        float32x4_t vt = vdupq_n_f32(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= n; j += 8)
        {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // ��ʼ���ѹ����߳�
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workstart[t_id]);
        }
        // ���߳�˯��
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_wait(&sem_main);
        }
        // ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // �����߳�
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void *normal_sem_tri_thread(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;

    for (int k = 0; k < n; k++)
    {
        // 0���߳�����������������ȴ�
        if (t_id == 0)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]); // �������ȴ��������

        // 0���̻߳������������̣߳�ִ����ȥ����
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_post(&sem_Division[i]);
            }
        }

        // ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // ִ����ȥ����
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_post(&sem_Elimination[i]);
            }
        }
        else
        {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }
    pthread_exit(NULL);
    return NULL;
}
// ƽ���㷨��̬�߳� + �ź���ͬ�� + ����ѭ��ȫ�������̺߳���
void normal_sem_tri(float **&A, int n)
{
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t *handles = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t *param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, normal_sem_tri_thread, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // �����߳�
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void *sem_triplecircle_thread(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;

    for (int k = 0; k < n; k++)
    {
        // 0���߳�����������������ȴ�
        if (t_id == 0)
        {
            float32x4_t vt = vdupq_n_f32(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= n; j += 8)
            {
                float32x4_t va = vld1q_f32(&A[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&A[k][j], va);
            }
            for (; j < n; j++)
            {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]); // �������ȴ��������

        // 0���̻߳������������̣߳�ִ����ȥ����
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_post(&sem_Division[i]);
            }
        }

        // ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // ִ����ȥ����
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8)
            {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_post(&sem_Elimination[i]);
            }
        }
        else
        {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }
    pthread_exit(NULL);
    return NULL;
}
// neon�Ż���̬�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳���
void neon_sem_triplecircle(float **&A, int n)
{
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t *handles = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t *param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, sem_triplecircle_thread, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // �����߳�
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void *normal_barrier_thread(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; k++)
    {
        // 0���߳�����������������ȴ�
        if (t_id == 0)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }

        // ��һ��ͬ����
        pthread_barrier_wait(&barrier_Division);

        // ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // ִ����ȥ����
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        // �ڶ���ͬ����
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
// ƽ���㷨��̬�߳� + barrierͬ��
void normal_barrier(float **&A, int n)
{
    // ��ʼ��barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    // �����߳�
    pthread_t *handles = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t *param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, normal_barrier_thread, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // �������е�barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
    free(handles);
    free(param);
}
//==================================================================
void *neon_barrier_threadFunc(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; k++)
    {
        // 0���߳�����������������ȴ�
        if (t_id == 0)
        {
            float32x4_t vt = vdupq_n_f32(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= n; j += 8)
            {
                float32x4_t va = vld1q_f32(&A[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&A[k][j], va);
            }
            for (; j < n; j++)
            {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }

        // ��һ��ͬ����
        pthread_barrier_wait(&barrier_Division);

        // ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8)
            {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0.0;
        }

        // �ڶ���ͬ����
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
// neon��̬�߳� + barrierͬ��
void neon_barrier(float **&A, int n)
{
    // ��ʼ��barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    // �����߳�
    pthread_t *handles = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t *param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id],NULL, neon_barrier_threadFunc, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // �������е�barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
    free(handles);
    free(param);
}
//==================================================================

int main()
{
    // �Բ�ͬ�����ݹ�ģ���в���
    for (int sizes : {500, 1000, 1500, 2000})
    {
        n = sizes;

        reset(A, sizes);
        auto start_time = high_resolution_clock::now();                             // ��ʼʱ��
        normal(A, sizes);                                                        // ִ��ƽ����˹��ȥ��
        auto end_time = high_resolution_clock::now();                               // ����ʱ��
        auto duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // ��ʼʱ��
        neon(A, sizes);                                                     // ִ��neon�Ż���˹��ȥ��
        end_time = high_resolution_clock::now();                               // ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", neon�Ż���ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // ��ʼʱ��
        normal_dynamic(A, sizes);                                           // ִ��ƽ���㷨��̬�̸߳�˹��ȥ��
        end_time = high_resolution_clock::now();                               // ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬�̺߳�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // ��ʼʱ��
        neon_dynamic(A, sizes);                                             // ִ��neon�Ż���̬�̸߳�˹��ȥ��
        end_time = high_resolution_clock::now();                               // ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", neon�Ż���̬�̺߳�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // ��ʼʱ��
        normal_sem_static(A, sizes);                                        // ִ��ƽ���㷨��̬�߳�+�ź���ͬ����˹��ȥ��
        end_time = high_resolution_clock::now();                               // ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬�߳�+�ź���ͬ����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // ��ʼʱ��
        neon_sem_static(A, sizes);                                          // ִ��neon��̬�߳�+�ź���ͬ����˹��ȥ��
        end_time = high_resolution_clock::now();                               // ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", neon�Ż���̬�߳�+�ź���ͬ����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // ��ʼʱ��
        normal_sem_tri(A, sizes);                                           // ִ��ƽ���㷨��̬�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳�����˹��ȥ��
        end_time = high_resolution_clock::now();                               // ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳�����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // ��ʼʱ��
        neon_sem_triplecircle(A, sizes);                                    // ִ��neon��̬�߳�+�ź���+����ѭ��ȫ�������̺߳�����˹��ȥ��
        end_time = high_resolution_clock::now();                               // ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", neon�Ż���̬�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳�����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // ��ʼʱ��
        normal_barrier(A, sizes);                                           // ִ��ƽ���㷨��̬�߳�+barrierͬ����˹��ȥ��
        end_time = high_resolution_clock::now();                               // ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬+barrierͬ����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // ��ʼʱ��
        neon_barrier(A, sizes);                                             // ִ��neon��̬�߳�+barrierͬ����˹��ȥ��
        end_time = high_resolution_clock::now();                               // ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // ����ʱ���
        cout << "Sizes: " << sizes << ", neon�Ż���̬barrier��ʱ: " << duration << " ms" << endl;
        clear(n);
    }
    return 0;
}
