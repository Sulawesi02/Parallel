#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

using namespace std;

float** A = NULL;
int n;                // 定义矩阵大小
#define NUM_THREADS 7 // 定义线程数量

void init(float**& A, int n) {
    A = new float* [n];
    srand(time(nullptr)); // 初始化随机数种子
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

void reset(float**& A, int n) {
    A = new float* [n];
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        memset(A[i], 0, n * sizeof(float));
    }

}

void deleteA(float**& A, int n) {
    for (int i = 0; i < n; i++) {
        delete[] A[i];
    }
    delete A;
}
// 平凡算法
void normal(int n) {
    double start_time;
    double end_time;
    init(A, n);
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++) {
        // 除法操作
        for (int j = k + 1; j < n; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        // 消去操作
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    cout << "平凡算法耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
    deleteA(A, n);

}
 //MPI块划分
void mpi_block(int n) {
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = n / num_proc * rank;
    int end = (rank == num_proc - 1) ? n : n / num_proc * (rank + 1);
    if (rank == 0) {  //0号进程初始化矩阵
        init(A, n);

        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);
            for (int i = b; i < e; i++) {
                MPI_Send(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1是初始矩阵信息，向每个进程发送数据
            }
        }
    }
    else {
        reset(A, n);
        for (int i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);// 同步  
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++) {
        if ((k >= begin && k < end)) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++) { 
                if (j != rank)
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0号消息表示除法完毕
            }
        }
        else {
            int src;
            if (k < n / num_proc * num_proc)//在可均分的任务量内
                src = k / (n / num_proc);
            else
                src = num_proc - 1;
            MPI_Recv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (int i = max(begin, k + 1); i < end; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);// 同步	
    if (rank == 0) {//0号进程中存有最终结果
        end_time = MPI_Wtime();
        cout << "MPI块划分耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//MPI循环划分
void mpi_circle(int n) {
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {  //0号进程初始化矩阵
        init(A, n);

        for (int j = 1; j < num_proc; j++) {
            for (int i = j; i < n; i += num_proc) {
                MPI_Send(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1是初始矩阵信息，向每个进程发送数据
            }
        }
    }
    else {
        reset(A, n);
        for (int i = rank; i < n; i += num_proc) {
            MPI_Recv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);// 同步
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++) {
        if (k % num_proc == rank) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++) {
                if (j != rank)
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0号消息表示除法完毕
            }
        }
        else {
            int src = k % num_proc;
            MPI_Recv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        int begin = k;
        while (begin % num_proc != rank)
            begin++;
        for (int i = begin; i < n; i += num_proc) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);// 同步	
    if (rank == 0) {//0号进程中存有最终结果
        end_time = MPI_Wtime();
        cout << "MPI循环划分耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//=====================================================================================
// MPI块划分+avx
void mpi_block_avx(int n)
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = n / num_proc * rank;
    int end = (rank == num_proc - 1) ? n : n / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);
            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // 同步
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++)
    {
        if ((k >= begin && k < end))
        {
            __m256 div = _mm256_set1_ps(A[k][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 va = _mm256_loadu_ps(&A[k][j]);
                va = _mm256_div_ps(va, div);
                _mm256_storeu_ps(&A[k][j], va);
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++)
            {
                if (j != rank)
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0号消息表示除法完毕
            }
        }
        else
        {
            int src;
            if (k < n / num_proc * num_proc) // 在可均分的任务量内
                src = k / (n / num_proc);
            else
                src = num_proc - 1;
            MPI_Recv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (int i = max(begin, k + 1); i < end; i++)
        {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8)
            {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 同步
    if (rank == 0)
    { // 0号进程中存有最终结果
        end_time = MPI_Wtime();
        cout << "MPI块划分+avx耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
// MPI块划分+OpenMP
void mpi_block_omp(int n)
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = n / num_proc * rank;
    int end = (rank == num_proc - 1) ? n : n / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // 同步
    start_time = MPI_Wtime();
    int i = 0, j = 0, k = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < n; k++)
    {
#pragma omp single
        if ((k >= begin && k < end))
        {
            for (j = k + 1; j < n; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < num_proc; j++)
            {
                if (j != rank)
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0号消息表示除法完毕
            }
        }
        else
        {
            int src;
            if (k < n / num_proc * num_proc) // 在可均分的任务量内
                src = k / (n / num_proc);
            else
                src = num_proc - 1;
            MPI_Recv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
#pragma omp for schedule(guided)
        for (i = max(begin, k + 1); i < end; i++)
        {
            for (j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 同步
    if (rank == 0)
    { // 0号进程中存有最终结果
        end_time = MPI_Wtime();
        cout << "MPI块划分+OpenMP耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
// MPI块划分+avx+OpenMP
void mpi_block_neon_omp(int n)
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = n / num_proc * rank;
    int end = (rank == num_proc - 1) ? n : n / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // 同步
    start_time = MPI_Wtime();
    int i = 0, j = 0, k = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)

    for (int k = 0; k < n; k++)
    {
#pragma omp single
        if ((k >= begin && k < end))
        {
            __m256 div = _mm256_set1_ps(A[k][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 va = _mm256_loadu_ps(&A[k][j]);
                va = _mm256_div_ps(va, div);
                _mm256_storeu_ps(&A[k][j], va);
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++)
            {
                if (j != rank)
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0号消息表示除法完毕
            }
        }
        else
        {
            int src;
            if (k < n / num_proc * num_proc) // 在可均分的任务量内
                src = k / (n / num_proc);
            else
                src = num_proc - 1;
            MPI_Recv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
#pragma omp for schedule(guided)
        for (int i = max(begin, k + 1); i < end; i++)
        {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8)
            {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 同步
    if (rank == 0)
    { // 0号进程中存有最终结果
        end_time = MPI_Wtime();
        cout << "MPI块划分+avx+OpenMP耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//=====================================================================================
//MPI块划分+非阻塞通信
void mpi_async(int n) {
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = n / num_proc * rank;
    int end = (rank == num_proc - 1) ? n : n / num_proc * (rank + 1);

    if (rank == 0) {  //0号进程初始化矩阵
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); //等待传递

    }
    else {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);// 同步
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++) {
        if ((k >= begin && k < end)) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[num_proc - 1 - rank];  //非阻塞传递
            for (int j = rank + 1; j < num_proc; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
            }
            MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; //若执行完自身的任务，可直接跳出
        }
        else {
            int src = k / (n / num_proc);
            MPI_Request request;
            MPI_Irecv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
        }
        for (int i = max(begin, k + 1); i < end; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);// 同步	
    if (rank == num_proc - 1) {
        end_time = MPI_Wtime();
        cout << "MPI块划分+非阻塞通信耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//MPI非阻塞通信+avx
void mpi_async_avx(int n) {
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = n / num_proc * rank;
    int end = (rank == num_proc - 1) ? n : n / num_proc * (rank + 1);

    if (rank == 0) {  //0号进程初始化矩阵
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); //等待传递
    }
    else {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);// 同步
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++) {
        {
            if ((k >= begin && k < end)) {
                __m256 vt = _mm256_set1_ps(A[k][k]);
                int j;
                for (j = k + 1; j + 8 <= n; j += 8) {
                    __m256 va = _mm256_loadu_ps(&A[k][j]);
                    va = _mm256_div_ps(va, vt);
                    _mm256_storeu_ps(&A[k][j], va);
                }
                for (; j < n; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[num_proc - 1 - rank];  //非阻塞传递
                for (j = 0; j < num_proc; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
                }
                MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < n / num_proc * num_proc)//在可均分的任务量内
                    src = k / (n / num_proc);
                else
                    src = num_proc - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
        for (int i = max(begin, k + 1); i < end; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);   
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);// 同步
    if (rank == num_proc - 1) {
        end_time = MPI_Wtime();
        cout << "MPI块划分+非阻塞通信+avx耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//MPI非阻塞通信+OpenMP
void mpi_async_omp(int n) {
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = n / num_proc * rank;
    int end = (rank == num_proc - 1) ? n : n / num_proc * (rank + 1);

    if (rank == 0) {  //0号进程初始化矩阵
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); //等待传递

    }
    else {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);// 同步
    start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (int k = 0; k < n; k++) {
#pragma omp single
        {
            if ((k >= begin && k < end)) {
                for (int j = k + 1; j < n; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[num_proc - 1 - rank];  //非阻塞传递
                for (int j = 0; j < num_proc; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
                }
                MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < n / num_proc * num_proc)//在可均分的任务量内
                    src = k / (n / num_proc);
                else
                    src = num_proc - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
#pragma omp for schedule(guided)  //向导调度
        for (int i = max(begin, k + 1); i < end; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);// 同步	
    if (rank == num_proc - 1) {
        end_time = MPI_Wtime();
        cout << "MPI块划分+非阻塞通信+OpenMP耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//MPI非阻塞通信+avx+OpenMP
void mpi_async_avx_omp(int n) {
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = n / num_proc * rank;
    int end = (rank == num_proc - 1) ? n : n / num_proc * (rank + 1);

    if (rank == 0) {  //0号进程初始化矩阵
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); //等待传递

    }
    else {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);// 同步
    start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (int k = 0; k < n; k++) {
#pragma omp single
        {
            if ((k >= begin && k < end)) {
                __m256 vt = _mm256_set1_ps(A[k][k]);
                int j;
                for (j = k + 1; j + 8 <= n; j += 8) {
                    __m256 va = _mm256_loadu_ps(&A[k][j]);
                    va = _mm256_div_ps(va, vt);
                    _mm256_storeu_ps(&A[k][j], va);
                }
                for (; j < n; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[num_proc - 1 - rank];  //非阻塞传递
                for (j = 0; j < num_proc; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
                }
                MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < n / num_proc * num_proc)//在可均分的任务量内
                    src = k / (n / num_proc);
                else
                    src = num_proc - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
#pragma omp for schedule(guided)  //开始多线程
        for (int i = max(begin, k + 1); i < end; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);   
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);// 同步
    if (rank == num_proc - 1) {
        end_time = MPI_Wtime();
        cout << "MPI块划分+非阻塞通信+avx+OpenMP耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}


int main(int argc, char* argv[]) {
    
    MPI_Init(&argc, &argv);
    // 对不同的数据规模进行测试
    for (int sizes : {500, 1000, 1500, 2000, 2500})
    {
        n = sizes;
        cout << "Sizes: " << n << endl;
        normal(n);//平凡算法
        mpi_block(n);//MPI块划分
        mpi_circle(n);//MPI循环划分
        mpi_block_avx(n);//MPI块划分+avx
        mpi_block_omp(n);//MPI块划分+OpenMP
        mpi_block_neon_omp(n);//MPI块划分+avx+OpenMP
        mpi_async(n);//MPI块划分+非阻塞通信
        mpi_async_avx(n);//MPI块划分+非阻塞通信+avx
        mpi_async_omp(n);//MPI块划分+非阻塞通信+OpenMP
        mpi_async_avx_omp(n);//MPI块划分+非阻塞通信+avx+OpenMP
    }
    MPI_Finalize();
    return 0;
}
