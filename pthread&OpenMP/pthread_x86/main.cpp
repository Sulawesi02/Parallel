#include <iostream>
#include <ctime>
#include <cstdlib>
#include<nmmintrin.h>  // SSE 4,2
#include <immintrin.h> //AVX��AVX2
#include <pthread.h>
#include <semaphore.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;


float** A;
int n; // ��������С
#define NUM_THREADS 7 // �����߳�����


sem_t sem_main;  //�ź���
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
//==================================================================
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
//==================================================================
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
//==================================================================
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
void* normal_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int k = p->k;           //��ȥ���ִ�
    int t_id = p->t_id;     //�̱߳��
    int i = k + t_id + 1;   //��ȡ����

    for (int j = k + 1; j < n; j++) {
        A[i][j] -= A[i][k] * A[k][j];
    }

    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}
//ƽ���㷨��̬�߳�
void normal_dynamic(float**& A, int n) {
    for (int k = 0; k < n; k++) {
        //���߳�����������
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        //���������̣߳�ִ����ȥ����
        int work_count = n - 1 - k;//�����߳�����
        pthread_t* handles = (pthread_t*)malloc(work_count * sizeof(pthread_t));
        threadParam_t * param = (threadParam_t *)malloc(work_count * sizeof(threadParam_t ));
        //��������
        for (int t_id = 0; t_id < work_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //�����߳�
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
    int k = p->k;           //��ȥ���ִ�
    int t_id = p->t_id;     //�߳�id
    int i = k + t_id + 1;   //��ȡ����
    __m128 vik = _mm_set1_ps(A[i][k]);
    int j;
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
    pthread_exit(NULL);
    return NULL;
}
//sse�Ż��㷨��̬�߳�
void sse_dynamic(float**& A, int n) {
    for (int k = 0; k < n; k++) {
        //���߳�����������
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

        //���������̣߳�ִ����ȥ����
        int work_count = n - 1 - k;//�����߳�����
        pthread_t* handles = (pthread_t*)malloc(work_count * sizeof(pthread_t));
        threadParam_t * param = (threadParam_t *)malloc(work_count * sizeof(threadParam_t ));
        //��������
        for (int t_id = 0; t_id < work_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //�����߳�
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
    int k = p->k;           //��ȥ���ִ�
    int t_id = p->t_id;     //�߳�id
    int i = k + t_id + 1;   //��ȡ����
    __m256 vaik = _mm256_set1_ps(A[i][k]);
    int j;
    for (j = k + 1; j + 8 <= n; j += 8) {
        __m256 vakj = _mm256_loadu_ps(&A[k][j]);
        __m256 vaij = _mm256_loadu_ps(&A[i][j]);
        __m256 vx = _mm256_mul_ps(vakj, vaik);
        vaij = _mm256_sub_ps(vaij, vx);
        _mm256_storeu_ps(&A[i][j], vaij);
    }
    for (; j < n; j++) {
        A[i][j] -= A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}
//avx�Ż��㷨��̬�߳�
void avx_dynamic(float**& A, int n) {
    for (int k = 0; k < n; k++) {
        //���߳�����������
        __m256 vt = _mm256_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= n; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[k][j]);
            va = _mm256_div_ps(va, vt);
            _mm256_storeu_ps(&A[k][j], va);
        }
        for (; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        //���������̣߳�ִ����ȥ����
        int work_count = n - 1 - k;//�����߳�����
        pthread_t* handles = (pthread_t*)malloc(work_count * sizeof(pthread_t));
        threadParam_t* param = (threadParam_t*)malloc(work_count * sizeof(threadParam_t));
        //��������
        for (int t_id = 0; t_id < work_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //�����߳�
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

    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workstart[t_id]);//�������ȴ����̳߳������

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            //ִ����ȥ����
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        sem_post(&sem_main);//�������߳�
        sem_wait(&sem_workend[t_id]);//�������ȴ����̻߳��ѽ�����һ��
    }
    pthread_exit(NULL);
    return NULL;
}
//ƽ���㷨��̬�߳�+�ź���ͬ��
void normal_sem_static(float**& A, int n) {
    sem_init(&sem_main, 0, 0);//��ʼ���ź���
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    //�����߳�
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t * param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t ));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, normal_sem_threadFunc, &param[t_id]);
    }
    for (int k = 0; k < n; k++) {
        //���߳�����������
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        //��ʼ���ѹ����߳�
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workstart[t_id]);
        }
        //���߳�˯��
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_wait(&sem_main);
        }
        //���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //�����߳�
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

    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workstart[t_id]);//�������ȴ����̳߳������

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            //ִ����ȥ����
            __m128 vik = _mm_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m128 vkj = _mm_loadu_ps(&A[k][j]);
                __m128 vij = _mm_loadu_ps(&A[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0.0;
        }
        sem_post(&sem_main);//�������߳�
        sem_wait(&sem_workend[t_id]);//�������ȴ����̻߳��ѽ�����һ��
    }
    pthread_exit(NULL);
    return NULL;
}
//sse�Ż���̬�߳�+�ź���ͬ��
void sse_sem_static(float**& A, int n) {
    sem_init(&sem_main, 0, 0);//��ʼ���ź���
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    //�����߳�
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t * param = (threadParam_t *)malloc(NUM_THREADS * sizeof(threadParam_t ));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, sse_sem_threadFunc, &param[t_id]);

    }
    for (int k = 0; k < n; k++) {
        //���߳�����������
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

        //��ʼ���ѹ����߳�
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workstart[t_id]);
        }
        //���߳�˯��
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_wait(&sem_main);
        }
        //���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //�����߳�
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

    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workstart[t_id]);//�������ȴ����̳߳������

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            //ִ����ȥ����
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vakj = _mm256_loadu_ps(&A[k][j]);
                __m256 vaij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(&A[i][j], vaij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0.0;
        }
        sem_post(&sem_main);//�������߳�
        sem_wait(&sem_workend[t_id]);//�������ȴ����̻߳��ѽ�����һ��
    }
    pthread_exit(NULL);
    return NULL;
}
//avx�Ż���̬�߳�+�ź���ͬ��
void avx_sem_static(float**& A, int n) {
    sem_init(&sem_main, 0, 0);//��ʼ���ź���
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    //�����߳�
    pthread_t* handles = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handles[t_id], NULL, avx_sem_threadFunc, &param[t_id]);

    }
    for (int k = 0; k < n; k++) {
        //���߳�����������
        __m256 vt = _mm256_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= n; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[k][j]);
            va = _mm256_div_ps(va, vt);
            _mm256_storeu_ps(&A[k][j], va);
        }
        for (; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        //��ʼ���ѹ����߳�
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workstart[t_id]);
        }
        //���߳�˯��
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_wait(&sem_main);
        }
        //���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            sem_post(&sem_workend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    //�����߳�
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

    for (int k = 0; k < n; k++) {
        //0���߳�����������������ȴ�
        if (t_id == 0) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]);//�������ȴ��������

        //0���̻߳������������̣߳�ִ����ȥ����
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Division[i]);
            }
        }

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            //ִ����ȥ����
            for (int j = k + 1; j < n; j++) {
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
//ƽ���㷨��̬�߳� + �ź���ͬ�� + ����ѭ��ȫ�������̺߳���
void normal_sem_tri_static(float**& A, int n) {
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
    //�����߳�
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

    for (int k = 0; k < n; k++) {
        //0���߳�����������������ȴ�
        if (t_id == 0) {
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
        else
            sem_wait(&sem_Division[t_id - 1]);//�������ȴ��������

        //0���̻߳������������̣߳�ִ����ȥ����
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Division[i]);
            }
        }

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            //ִ����ȥ����
            __m128 vik = _mm_set1_ps(A[i][k]);
            int j;
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
//sse�Ż���̬�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳���
void sse_sem_tri_static(float**& A, int n) {
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
    //�����߳�
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

    for (int k = 0; k < n; k++) {
        //0���߳�����������������ȴ�
        if (t_id == 0) {
            __m256 vt = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 va = _mm256_loadu_ps(&A[k][j]);
                va = _mm256_div_ps(va, vt);
                _mm256_storeu_ps(&A[k][j], va);
            }
            for (; j < n; j++) {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]);//�������ȴ��������

        //0���̻߳������������̣߳�ִ����ȥ����
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Division[i]);
            }
        }

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            //ִ����ȥ����
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vakj = _mm256_loadu_ps(&A[k][j]);
                __m256 vaij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(&A[i][j], vaij);
            }
            for (; j < n; j++) {
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
//avx�Ż���̬�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳���
void avx_sem_tri_static(float**& A, int n) {
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
    //�����߳�
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
    for (int k = 0; k < n; k++) {
        //0���߳�����������������ȴ�
        if (t_id == 0) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }

        //��һ��ͬ����
        pthread_barrier_wait(&barrier_Division);

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            //ִ����ȥ����
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        //�ڶ���ͬ����
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
//ƽ���㷨��̬�߳� + barrierͬ��
void normal_barrier_static(float**& A, int n) {
    //��ʼ��barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    //�����߳�
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
    //�������е�barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
    free(handles);
    free(param);
}
//==================================================================
void* sse_barrier_threadFunc(void* param) {
    threadParam_t * p = (threadParam_t *)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; k++) {
        //0���߳�����������������ȴ�
        if (t_id == 0) {
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

        //��һ��ͬ����
        pthread_barrier_wait(&barrier_Division);

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            __m128 vik = _mm_set1_ps(A[i][k]);
            int j;
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

            A[i][k] = 0.0;
        }

        //�ڶ���ͬ����
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
//sse��̬�߳� + barrierͬ��
void sse_barrier_static(float**& A, int n)
{
    //��ʼ��barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    //�����߳�
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
    //�������е�barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
    free(handles);
    free(param);
}
//==================================================================
void* avx_barrier_threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; k++) {
        //0���߳�����������������ȴ�
        if (t_id == 0) {
            __m256 vt = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 va = _mm256_loadu_ps(&A[k][j]);
                va = _mm256_div_ps(va, vt);
                _mm256_storeu_ps(&A[k][j], va);
            }
            for (; j < n; j++) {
                A[k][j] /= A[k][k];
            }

            A[k][k] = 1.0;
        }

        //��һ��ͬ����
        pthread_barrier_wait(&barrier_Division);

        //ѭ����������
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vakj = _mm256_loadu_ps(&A[k][j]);
                __m256 vaij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(&A[i][j], vaij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0.0;
        }

        //�ڶ���ͬ����
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}
//avx��̬�߳� + barrierͬ��
void avx_barrier_static(float**& A, int n)
{
    //��ʼ��barrier
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    //�����߳�
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
    //�������е�barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
    free(handles);
    free(param);
}
//==================================================================

int main() {
    // �Բ�ͬ�����ݹ�ģ���в���
    for (int sizes : {500, 1000, 1500, 2000}) {
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
        //==================================================================
        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_dynamic(A, sizes); // ִ��ƽ���㷨��̬�̸߳�˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬�̺߳�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_dynamic(A, sizes); // ִ��sse�Ż���̬�̸߳�˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", sse�Ż���̬�̺߳�ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_dynamic(A, sizes); // ִ��avx�Ż���̬�̸߳�˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", avx�Ż���̬�̺߳�ʱ: " << duration << " ms" << endl;
        clear(n);
        //==================================================================
        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_sem_static(A, sizes); // ִ��ƽ���㷨��̬8�߳�+�ź���ͬ����˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬�߳�+�ź���ͬ����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_sem_static(A, sizes); // ִ��sse��̬8�߳�+�ź���ͬ����˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", sse�Ż���̬�߳�+�ź���ͬ����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_sem_static(A, sizes); // ִ��avx��̬8�߳�+�ź���ͬ����˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", avx�Ż���̬�߳�+�ź���ͬ����ʱ: " << duration << " ms" << endl;
        clear(n);
        //==================================================================
        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_sem_tri_static(A, sizes); // ִ��ƽ���㷨��̬8�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳�����˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳�����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_sem_tri_static(A, sizes); // ִ��sse��̬8�߳�+�ź���+����ѭ��ȫ�������̺߳�����˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", sse�Ż���̬�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳�����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_sem_tri_static(A, sizes); // ִ��avx��̬8�߳�+�ź���+����ѭ��ȫ�������̺߳�����˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", avx�Ż���̬�߳�+�ź���ͬ��+����ѭ��ȫ�������̺߳�����ʱ: " << duration << " ms" << endl;
        clear(n);
        //==================================================================
        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        normal_barrier_static(A, sizes); // ִ��ƽ���㷨��̬8�߳�+barrierͬ����˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", ƽ���㷨��̬+barrierͬ����ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        sse_barrier_static(A, sizes); // ִ��sse��̬8�߳�+barrierͬ����˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", sse�Ż���̬barrier��ʱ: " << duration << " ms" << endl;
        clear(n);

        reset(A, sizes);
        start_time = high_resolution_clock::now(); // ��ʼʱ��
        avx_barrier_static(A, sizes); // ִ��avx��̬8�߳�+barrierͬ����˹��ȥ��
        end_time = high_resolution_clock::now();// ����ʱ��
        duration = duration_cast<milliseconds>(end_time - start_time).count();// ����ʱ���
        cout << "Sizes: " << sizes << ", avx�Ż���̬barrier��ʱ: " << duration << " ms" << endl;
        clear(n);
    }
    return 0;
}