#include <iostream>
#include <ctime>
#include <cstdlib>
#include <nmmintrin.h> // SSE 4,2
#include <immintrin.h> //AVX、AVX2
#include <pthread.h>
#include <semaphore.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

int N = 5000; // 定义最大矩阵大小
float** A;
#define NUM_THREADS 7 // 定义线程数量


sem_t sem_main;  //信号量
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

void init() {
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
//==================================================================
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
//==================================================================
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
//==================================================================
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
void* normal_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程编号
    int i = k + t_id + 1;   //获取任务

    for (int j = k + 1; j < N; j++) {
        A[i][j] -= A[i][k] * A[k][j];
    }

    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}
//平凡算法动态线程
void normal_dynamic() {
    for (int k = 0; k < N; k++) {
        //主线程做除法操作
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        //创建工作线程，执行消去操作
        int work_count = N - 1 - k;//工作线程数量
        pthread_t* handles = (pthread_t*)malloc(work_count * sizeof(pthread_t));
        threadParam_t * param = (threadParam_t *)malloc(work_count * sizeof(threadParam_t ));
        //分配任务
        for (int t_id = 0; t_id < work_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //创建线程
        for (int t_id = 0; t_id < work_count; t_id++) {
            pthread_create(&handles[t_id], NULL, normal_threadFunc, &param[t_id]);
        }
        for (int t_id = 0; t_id < work_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        free(handles);
        free(param);
    }
}
//==================================================================
void* sse_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程id
    int i = k + t_id + 1;   //获取任务
    __m128 vik = _mm_set1_ps(A[i][k]);
    int j;
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
    pthread_exit(NULL);
    return NULL;
}
//sse优化算法动态线程
void sse_dynamic() {
    for (int k = 0; k < N; k++) {
        //主线程做除法操作
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

        //创建工作线程，执行消去操作
        int work_count = N - 1 - k;//工作线程数量
        pthread_t* handles = (pthread_t*)malloc(work_count * sizeof(pthread_t));
        threadParam_t * param = (threadParam_t *)malloc(work_count * sizeof(threadParam_t ));
        //分配任务
        for (int t_id = 0; t_id < work_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //创建线程
        for (int t_id = 0; t_id < work_count; t_id++) {
            pthread_create(&handles[t_id], NULL, sse_threadFunc, &param[t_id]);
        }
        for (int t_id = 0; t_id < work_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        free(handles);
        free(param);
    }
}
//==================================================================
void* avx_threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程id
    int i = k + t_id + 1;   //获取任务
    __m256 vaik = _mm256_set1_ps(A[i][k]);
    int j;
    for (j = k + 1; j + 8 <= N; j += 8) {
        __m256 vakj = _mm256_loadu_ps(&A[k][j]);
        __m256 vaij = _mm256_loadu_ps(&A[i][j]);
        __m256 vx = _mm256_mul_ps(vakj, vaik);
        vaij = _mm256_sub_ps(vaij, vx);
        _mm256_storeu_ps(&A[i][j], vaij);
    }
    for (; j < N; j++) {
        A[i][j] -= A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}
//avx优化算法动态线程
void avx_dynamic() {
    for (int k = 0; k < N; k++) {
        //主线程做除法操作
        __m256 vt = _mm256_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[k][j]);
            va = _mm256_div_ps(va, vt);
            _mm256_storeu_ps(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        //创建工作线程，执行消去操作
        int work_count = N - 1 - k;//工作线程数量
        pthread_t* handles = (pthread_t*)malloc(work_count * sizeof(pthread_t));
        threadParam_t* param = (threadParam_t*)malloc(work_count * sizeof(threadParam_t));
        //分配任务
        for (int t_id = 0; t_id < work_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //创建线程
        for (int t_id = 0; t_id < work_count; t_id++) {
            pthread_create(&handles[t_id], NULL, avx_threadFunc, &param[t_id]);
        }
        for (int t_id = 0; t_id < work_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        free(handles);
        free(param);
    }
}
//==================================================================
void* normal_sem_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workstart[t_id]);//阻塞，等待主线程除法完成

        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            //执行消去操作
            for (int j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        sem_post(&sem_main);//唤醒主线程
        sem_wait(&sem_workend[t_id]);//阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return NULL;
}
//平凡算法静态线程+信号量同步
void normal_sem_static() {
    sem_init(&sem_main, 0, 0);//初始化信号量
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    //创建线程
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t * param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t ));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, normal_sem_threadFunc, &param[t_id]);
    }
    for (int k = 0; k < N; k++) {
        //主线程做除法操作
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        //开始唤醒工作线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workstart[t_id]);
        }
        //主线程睡眠
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_wait(&sem_main);
        }
        //主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //销毁线程
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void* sse_sem_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workstart[t_id]);//阻塞，等待主线程除法完成

        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            //执行消去操作
            __m128 vik = _mm_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0.0;
        }
        sem_post(&sem_main);//唤醒主线程
        sem_wait(&sem_workend[t_id]);//阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return NULL;
}
//sse优化静态线程+信号量同步
void sse_sem_static() {
    sem_init(&sem_main, 0, 0);//初始化信号量
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    //创建线程
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t * param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t ));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, sse_sem_threadFunc, &param[t_id]);

    }
    for (int k = 0; k < N; k++) {
        //主线程做除法操作
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

        //开始唤醒工作线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workstart[t_id]);
        }
        //主线程睡眠
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_wait(&sem_main);
        }
        //主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //销毁线程
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void* avx_sem_threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workstart[t_id]);//阻塞，等待主线程除法完成

        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            //执行消去操作
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vakj = _mm256_loadu_ps(&A[k][j]);
                __m256 vaij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(&A[i][j], vaij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0.0;
        }
        sem_post(&sem_main);//唤醒主线程
        sem_wait(&sem_workend[t_id]);//阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return NULL;
}
//avx优化静态线程+信号量同步
void avx_sem_static() {
    sem_init(&sem_main, 0, 0);//初始化信号量
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    //创建线程
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, avx_sem_threadFunc, &param[t_id]);

    }
    for (int k = 0; k < N; k++) {
        //主线程做除法操作
        __m256 vt = _mm256_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[k][j]);
            va = _mm256_div_ps(va, vt);
            _mm256_storeu_ps(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        //开始唤醒工作线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workstart[t_id]);
        }
        //主线程睡眠
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_wait(&sem_main);
        }
        //主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //销毁线程
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void* normal_sem_tri_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) {
        //0号线程做除法操作，其余等待
        if (t_id == 0) {
            for (int j = k + 1; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]);//阻塞，等待除法完成

        //0号线程唤醒其他工作线程，执行消去操作
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Division[i]);
            }
        }

        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            //执行消去操作
            for (int j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }
    pthread_exit(NULL);
    return NULL;
}
//平凡算法静态线程 + 信号量同步 + 三重循环全部纳入线程函数
void normal_sem_tri_static() {
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t * param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t ));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, normal_sem_tri_threadFunc, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //销毁线程
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void* sse_sem_tri_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) {
        //0号线程做除法操作，其余等待
        if (t_id == 0) {
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
        else
            sem_wait(&sem_Division[t_id - 1]);//阻塞，等待除法完成

        //0号线程唤醒其他工作线程，执行消去操作
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Division[i]);
            }
        }

        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            //执行消去操作
            __m128 vik = _mm_set1_ps(A[i][k]);
            int j;
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
            A[i][k] = 0.0;
        }
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }
    pthread_exit(NULL);
    return NULL;
}
//sse优化静态线程+信号量同步+三重循环全部纳入线程函数
void sse_sem_tri_static() {
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t * param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t ));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, sse_sem_tri_threadFunc, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //销毁线程
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void* avx_sem_tri_threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) {
        //0号线程做除法操作，其余等待
        if (t_id == 0) {
            __m256 vt = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 va = _mm256_loadu_ps(&A[k][j]);
                va = _mm256_div_ps(va, vt);
                _mm256_storeu_ps(&A[k][j], va);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]);//阻塞，等待除法完成

        //0号线程唤醒其他工作线程，执行消去操作
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Division[i]);
            }
        }

        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            //执行消去操作
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vakj = _mm256_loadu_ps(&A[k][j]);
                __m256 vaij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(&A[i][j], vaij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }
    pthread_exit(NULL);
    return NULL;
}
//avx优化静态线程+信号量同步+三重循环全部纳入线程函数
void avx_sem_tri_static() {
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, avx_sem_tri_threadFunc, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //销毁线程
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);
    free(handles);
    free(param);
}
//==================================================================
void* normal_barrier_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        //0号线程做除法操作，其余等待
        if (t_id == 0) {
            for (int j = k + 1; j < N; j++) {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }

        //第一个同步点
        pthread_barrier_wait(&barrier_Division);

        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            //执行消去操作
            for (int j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        //第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
//平凡算法静态线程 + barrier同步
void normal_barrier_static() {
    //初始化barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    //创建线程
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t * param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t ));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, normal_barrier_threadFunc, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //销毁所有的barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
    free(handles);
    free(param);
}
//==================================================================
void* sse_barrier_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        //0号线程做除法操作，其余等待
        if (t_id == 0) {
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

        //第一个同步点
        pthread_barrier_wait(&barrier_Division);

        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            int j;
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

            A[i][k] = 0.0;
        }

        //第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
//sse静态线程 + barrier同步
void sse_barrier_static()
{
    //初始化barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    //创建线程
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t * param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t ));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, sse_barrier_threadFunc, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //销毁所有的barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
    free(handles);
    free(param);
}
//==================================================================
void* avx_barrier_threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        //0号线程做除法操作，其余等待
        if (t_id == 0) {
            __m256 vt = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 va = _mm256_loadu_ps(&A[k][j]);
                va = _mm256_div_ps(va, vt);
                _mm256_storeu_ps(&A[k][j], va);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }

        //第一个同步点
        pthread_barrier_wait(&barrier_Division);

        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vakj = _mm256_loadu_ps(&A[k][j]);
                __m256 vaij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(&A[i][j], vaij);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0.0;
        }

        //第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
//avx静态线程 + barrier同步
void avx_barrier_static()
{
    //初始化barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    //创建线程
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, avx_barrier_threadFunc, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //销毁所有的barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
    free(handles);
    free(param);
}
//==================================================================

int main() {
    int N_values[] = { 500, 1000, 1500, 2000 };
    for (int i = 0; i < sizeof(N_values) / sizeof(N_values[0]); ++i) {
        N = N_values[i];
        cout << "数据规模: " << N << endl;

        init();
        auto start_time = high_resolution_clock::now();// 开始时间
        normal(); // 执行平凡高斯消去法
        auto end_time = high_resolution_clock::now();// 结束时间
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "平凡算法耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        sse(); // 执行sse优化高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "sse优化耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        avx(); // 执行avx优化高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "avx优化耗时: " << duration << " ms" << endl;
        
        //==================================================================
        init();
        start_time = high_resolution_clock::now(); // 开始时间
        normal_dynamic(); // 执行平凡算法动态线程高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "平凡算法动态线程耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        sse_dynamic(); // 执行sse优化动态线程高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "sse优化动态线程耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        avx_dynamic(); // 执行avx优化动态线程高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "avx优化动态线程耗时: " << duration << " ms" << endl;
        
        //==================================================================
        init();
        start_time = high_resolution_clock::now(); // 开始时间
        normal_sem_static(); // 执行平凡算法静态8线程+信号量同步高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "平凡算法静态线程+信号量同步耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        sse_sem_static(); // 执行sse静态8线程+信号量同步高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "sse优化静态线程+信号量同步耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        avx_sem_static(); // 执行avx静态8线程+信号量同步高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "avx优化静态线程+信号量同步耗时: " << duration << " ms" << endl;
        
        //==================================================================
        init();
        start_time = high_resolution_clock::now(); // 开始时间
        normal_sem_tri_static(); // 执行平凡算法静态8线程+信号量同步+三重循环全部纳入线程函数高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "平凡算法静态线程+信号量同步+三重循环全部纳入线程函数耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        sse_sem_tri_static(); // 执行sse静态8线程+信号量+三重循环全部纳入线程函数高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "sse优化静态线程+信号量同步+三重循环全部纳入线程函数耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        avx_sem_tri_static(); // 执行avx静态8线程+信号量+三重循环全部纳入线程函数高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "avx优化静态线程+信号量同步+三重循环全部纳入线程函数耗时: " << duration << " ms" << endl;
        
        //==================================================================
        init();
        start_time = high_resolution_clock::now(); // 开始时间
        normal_barrier_static(); // 执行平凡算法静态8线程+barrier同步高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "平凡算法静态+barrier同步耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        sse_barrier_static(); // 执行sse静态8线程+barrier同步高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "sse优化静态barrier耗时: " << duration << " ms" << endl;
        

        init();
        start_time = high_resolution_clock::now(); // 开始时间
        avx_barrier_static(); // 执行avx静态8线程+barrier同步高斯消去法
        end_time = high_resolution_clock::now();// 结束时间
        duration = duration_cast<milliseconds>(end_time - start_time).count();// 计算时间差
        cout << "avx优化静态barrier耗时: " << duration << " ms" << endl;
        
    }
    return 0;
}
