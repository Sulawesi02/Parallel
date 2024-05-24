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
int n;                // 定义矩阵大小
#define NUM_THREADS 7 // 定义线程数量

sem_t sem_main; // 信号量
sem_t sem_workstart[NUM_THREADS];
sem_t sem_workend[NUM_THREADS];

sem_t sem_leader;
sem_t sem_Division[NUM_THREADS];
sem_t sem_Elimination[NUM_THREADS];

pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

typedef struct {
     int k; //消去的轮次
     int t_id; // 线程 id
}threadParam_t;

void reset(float **&A, int n)
{
    A = new float *[n];
    srand(time(nullptr)); // 初始化随机数种子
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
// 平凡算法
void normal(float**& A, int n)
{
    for (int k = 0; k < n; k++)
    {
        // 除法操作
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        // 消去操作
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
// neon优化
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
    int k = p->k;         // 消去的轮次
    int t_id = p->t_id;   // 线程编号
    int i = k + t_id + 1; // 获取任务

    for (int j = k + 1; j < n; j++)
    {
        A[i][j] -= A[i][k] * A[k][j];
    }

    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}
// 平凡算法动态线程
void normal_dynamic(float **&A, int n)
{
    for (int k = 0; k < n; k++)
    {
        // 主线程做除法操作
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // 创建工作线程，执行消去操作
        int work_count = n - 1 - k; // 工作线程数量
        pthread_t *handles = (pthread_t *)malloc(work_count * sizeof(pthread_t));
        threadParam_t *param = (threadParam_t *)malloc(work_count * sizeof(threadParam_t));
        // 分配任务
        for (int t_id = 0; t_id < work_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        // 创建线程
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
    int k = p->k;         // 消去的轮次
    int t_id = p->t_id;   // 线程
    int i = k + t_id + 1; // 获取任务

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
// neon优化算法动态线程
void neon_dynamic(float **&A, int n)
{
    for (int k = 0; k < n; k++)
    {
        // 主线程做除法操作
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

        // 创建工作线程，执行消去操作
        int work_count = n - 1 - k; // 工作线程数量
        pthread_t *handles = (pthread_t *)malloc(work_count * sizeof(pthread_t));
        threadParam_t *param = (threadParam_t *)malloc(work_count * sizeof(threadParam_t));
        // 分配任务
        for (int t_id = 0; t_id < work_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        // 创建线程
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
        sem_wait(&sem_workstart[t_id]); // 阻塞，等待主线程除法完成

        // 循环划分任务
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // 执行消去操作
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        sem_post(&sem_main);          // 唤醒主线程
        sem_wait(&sem_workend[t_id]); // 阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return NULL;
}
// 平凡算法静态线程+信号量同步
void normal_sem_static(float **&A, int n)
{
    sem_init(&sem_main, 0, 0); // 初始化信号量
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    // 创建线程
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
        // 主线程做除法操作
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // 开始唤醒工作线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workstart[t_id]);
        }
        // 主线程睡眠
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_wait(&sem_main);
        }
        // 主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // 销毁线程
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
        sem_wait(&sem_workstart[t_id]); // 阻塞，等待主线程除法完成

        // 循环划分任务
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // 执行消去操作
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
        sem_post(&sem_main);          // 唤醒主线程
        sem_wait(&sem_workend[t_id]); // 阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return NULL;
}
// neon优化静态线程+信号量同步
void neon_sem_static(float **&A, int n)
{
    sem_init(&sem_main, 0, 0); // 初始化信号量
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    // 创建线程
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
        // 主线程做除法操作
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

        // 开始唤醒工作线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workstart[t_id]);
        }
        // 主线程睡眠
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_wait(&sem_main);
        }
        // 主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // 销毁线程
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
        // 0号线程做除法操作，其余等待
        if (t_id == 0)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]); // 阻塞，等待除法完成

        // 0号线程唤醒其他工作线程，执行消去操作
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_post(&sem_Division[i]);
            }
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // 执行消去操作
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
// 平凡算法静态线程 + 信号量同步 + 三重循环全部纳入线程函数
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
    // 销毁线程
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
        // 0号线程做除法操作，其余等待
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
            sem_wait(&sem_Division[t_id - 1]); // 阻塞，等待除法完成

        // 0号线程唤醒其他工作线程，执行消去操作
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_post(&sem_Division[i]);
            }
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // 执行消去操作
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
// neon优化静态线程+信号量同步+三重循环全部纳入线程函数
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
    // 销毁线程
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
        // 0号线程做除法操作，其余等待
        if (t_id == 0)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }

        // 第一个同步点
        pthread_barrier_wait(&barrier_Division);

        // 循环划分任务
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            // 执行消去操作
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
// 平凡算法静态线程 + barrier同步
void normal_barrier(float **&A, int n)
{
    // 初始化barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    // 创建线程
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
    // 销毁所有的barrier
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
        // 0号线程做除法操作，其余等待
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

        // 第一个同步点
        pthread_barrier_wait(&barrier_Division);

        // 循环划分任务
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

        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
// neon静态线程 + barrier同步
void neon_barrier(float **&A, int n)
{
    // 初始化barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    // 创建线程
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
    // 销毁所有的barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
    free(handles);
    free(param);
}
//==================================================================

int main()
{
    // 对不同的数据规模进行测试
    for (int sizes : {500, 1000, 1500, 2000})
    {
        n = sizes;

        reset(A, sizes);
        auto start_time = high_resolution_clock::now();                             // 开始时间
        normal(A, sizes);                                                        // 执行平凡高斯消去法
        auto end_time = high_resolution_clock::now();                               // 结束时间
        auto duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", 平凡算法耗时: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // 开始时间
        neon(A, sizes);                                                     // 执行neon优化高斯消去法
        end_time = high_resolution_clock::now();                               // 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", neon优化耗时: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // 开始时间
        normal_dynamic(A, sizes);                                           // 执行平凡算法动态线程高斯消去法
        end_time = high_resolution_clock::now();                               // 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", 平凡算法动态线程耗时: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // 开始时间
        neon_dynamic(A, sizes);                                             // 执行neon优化动态线程高斯消去法
        end_time = high_resolution_clock::now();                               // 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", neon优化动态线程耗时: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // 开始时间
        normal_sem_static(A, sizes);                                        // 执行平凡算法静态线程+信号量同步高斯消去法
        end_time = high_resolution_clock::now();                               // 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", 平凡算法静态线程+信号量同步耗时: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // 开始时间
        neon_sem_static(A, sizes);                                          // 执行neon静态线程+信号量同步高斯消去法
        end_time = high_resolution_clock::now();                               // 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", neon优化静态线程+信号量同步耗时: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // 开始时间
        normal_sem_tri(A, sizes);                                           // 执行平凡算法静态线程+信号量同步+三重循环全部纳入线程函数高斯消去法
        end_time = high_resolution_clock::now();                               // 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", 平凡算法静态线程+信号量同步+三重循环全部纳入线程函数耗时: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // 开始时间
        neon_sem_triplecircle(A, sizes);                                    // 执行neon静态线程+信号量+三重循环全部纳入线程函数高斯消去法
        end_time = high_resolution_clock::now();                               // 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", neon优化静态线程+信号量同步+三重循环全部纳入线程函数耗时: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // 开始时间
        normal_barrier(A, sizes);                                           // 执行平凡算法静态线程+barrier同步高斯消去法
        end_time = high_resolution_clock::now();                               // 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", 平凡算法静态+barrier同步耗时: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now();                             // 开始时间
        neon_barrier(A, sizes);                                             // 执行neon静态线程+barrier同步高斯消去法
        end_time = high_resolution_clock::now();                               // 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count(); // 计算时间差
        cout << "Sizes: " << sizes << ", neon优化静态barrier耗时: " << duration << " ms" << endl;
        clear(n);
    }
    return 0;
}
