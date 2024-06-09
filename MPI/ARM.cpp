#include <iostream>
#include <cstdlib>
#include <arm_neon.h>
#include <omp.h>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

int N = 5000; // 定义最大矩阵大小
float **A;
#define NUM_THREADS 7 // 定义线程数量

void init()
{
    A = new float *[N];
    srand(time(nullptr)); // 初始化随机数种子
    for (int i = 0; i < N; i++)
    {
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

void reset()
{
    A = new float *[N];
    for (int i = 0; i < N; i++)
    {
        A[i] = new float[N];
        memset(A[i], 0, N * sizeof(float));
    }
}
// 平凡算法
void normal()
{
    double start_time;
    double end_time;
    init();
    start_time = MPI_Wtime();
    for (int k = 0; k < N; k++)
    {
        // 除法操作
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        // 消去操作
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    cout << "平凡算法耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
}
// MPI块划分
void mpi_block()
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num_proc * rank;
    int end = (rank == num_proc - 1) ? N : N / num_proc * (rank + 1);
    if (rank == 0)
    { // 0号进程初始化矩阵
        init();

        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (N / num_proc), e = (j == num_proc - 1) ? N : (j + 1) * (N / num_proc);
            for (int i = b; i < e; i++)
            {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD); // 1是初始矩阵信息，向每个进程发送数据
            }
        }
    }
    else
    {
        reset();
        for (int i = begin; i < end; i++)
        {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // 此时每个进程都拿到了数据
    start_time = MPI_Wtime();
    for (int k = 0; k < N; k++)
    {
        if ((k >= begin && k < end))
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++)
            {
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0号消息表示除法完毕
            }
        }
        else
        {
            int src;
            if (k < N / num_proc * num_proc) // 在可均分的任务量内
                src = k / (N / num_proc);
            else
                src = num_proc - 1;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (int i = max(begin, k + 1); i < end; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步
    if (rank == 0)
    { // 0号进程中存有最终结果
        end_time = MPI_Wtime();
        cout << "MPI块划分耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
    }
}
// MPI循环划分
void mpi_circle()
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    { // 0号进程初始化矩阵
        init();

        for (int j = 1; j < num_proc; j++)
        {
            for (int i = j; i < N; i += num_proc)
            {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD); // 1是初始矩阵信息，向每个进程发送数据
            }
        }
    }
    else
    {
        reset();
        for (int i = rank; i < N; i += num_proc)
        {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // 此时每个进程都拿到了数据
    start_time = MPI_Wtime();
    for (int k = 0; k < N; k++)
    {
        if (k % num_proc == rank)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++)
            {
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0号消息表示除法完毕
            }
        }
        else
        {
            int src = k % num_proc;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        int begin = k;
        while (begin % num_proc != rank)
            begin++;
        for (int i = begin; i < N; i += num_proc)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步
    if (rank == 0)
    { // 0号进程中存有最终结果
        end_time = MPI_Wtime();
        cout << "MPI循环划分耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
    }
}
//=====================================================================================
// MPI块划分+neon
void mpi_block_neon()
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num_proc * rank;
    int end = (rank == num_proc - 1) ? N : N / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init();
        MPI_Request *request = new MPI_Request[N - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (N / num_proc), e = (j == num_proc - 1) ? N : (j + 1) * (N / num_proc);
            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset();
        MPI_Request *request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // 同步
    start_time = MPI_Wtime();
    for (int k = 0; k < N; k++)
    {
        if ((k >= begin && k < end))
        {
            float32x4_t vt = vdupq_n_f32(A[k][k]);
            int j;
            for (j = k + 1; j + 4 <= N; j += 4)
            {
                float32x4_t va = vld1q_f32(&A[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&A[k][j], va);
            }
            for (; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++)
            {
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0号消息表示除法完毕
            }
        }
        else
        {
            int src;
            if (k < N / num_proc * num_proc) // 在可均分的任务量内
                src = k / (N / num_proc);
            else
                src = num_proc - 1;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (int i = max(begin, k + 1); i < end; i++)
        {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4)
            {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < N; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 同步
    if (rank == 0)
    { // 0号进程中存有最终结果
        end_time = MPI_Wtime();
        cout << "MPI块划分+neon耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
    }
}
// MPI块划分+OpenMP
void mpi_block_omp()
{

    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num_proc * rank;
    int end = (rank == num_proc - 1) ? N : N / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init();
        MPI_Request *request = new MPI_Request[N - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (N / num_proc), e = (j == num_proc - 1) ? N : (j + 1) * (N / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset();
        MPI_Request *request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // 同步
    start_time = MPI_Wtime();
    int i = 0, j = 0, k = 0;
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        if ((k >= begin && k < end))
        {
            for (j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < num_proc; j++)
            {
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0号消息表示除法完毕
            }
        }
        else
        {
            int src;
            if (k < N / num_proc * num_proc) // 在可均分的任务量内
                src = k / (N / num_proc);
            else
                src = num_proc - 1;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
#pragma omp for schedule(guided)
        for (i = max(begin, k + 1); i < end; i++)
        {
            for (j = k + 1; j < N; j++)
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
    }
}
// MPI块划分+neon+OpenMP
void mpi_block_neon_omp()
{

    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num_proc * rank;
    int end = (rank == num_proc - 1) ? N : N / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init();
        MPI_Request *request = new MPI_Request[N - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (N / num_proc), e = (j == num_proc - 1) ? N : (j + 1) * (N / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset();
        MPI_Request *request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // 同步
    start_time = MPI_Wtime();
    int i = 0, j = 0, k = 0;
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)

    for (int k = 0; k < N; k++)
    {
#pragma omp single
        if ((k >= begin && k < end))
        {
            float32x4_t vt = vdupq_n_f32(A[k][k]);
            int j;
            for (j = k + 1; j + 4 <= N; j += 4)
            {
                float32x4_t va = vld1q_f32(&A[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&A[k][j], va);
            }
            for (; j < N; j++)
            {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++)
            {
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0号消息表示除法完毕
            }
        }
        else
        {
            int src;
            if (k < N / num_proc * num_proc) // 在可均分的任务量内
                src = k / (N / num_proc);
            else
                src = num_proc - 1;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
#pragma omp for schedule(guided)
        for (int i = max(begin, k + 1); i < end; i++)
        {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4)
            {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < N; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 同步
    if (rank == 0)
    { // 0号进程中存有最终结果
        end_time = MPI_Wtime();
        cout << "MPI块划分+neon+OpenMP耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
    }
}
//=====================================================================================
// MPI块划分+非阻塞通信
void mpi_async()
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num_proc * rank;
    int end = (rank == num_proc - 1) ? N : N / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init();
        MPI_Request *request = new MPI_Request[N - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (N / num_proc), e = (j == num_proc - 1) ? N : (j + 1) * (N / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset();
        MPI_Request *request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (int k = 0; k < N; k++)
    {
        if ((k >= begin && k < end))
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request *request = new MPI_Request[num_proc - 1 - rank]; // 非阻塞传递
            for (int j = rank + 1; j < num_proc; j++)
            { // 块划分中，已经消元好且进行了除法置1的行向量仅

                MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]); // 0号消息表示除法完毕
            }
            MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; // 若执行完自身的任务，可直接跳出
        }
        else
        {
            int src = k / (N / num_proc);
            MPI_Request request;
            MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE); // 实际上仍然是阻塞接收，因为接下来的操作需要这些数据
        }
        for (int i = max(begin, k + 1); i < end; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步
    if (rank == num_proc - 1)
    {
        end_time = MPI_Wtime();
        cout << "MPI块划分+非阻塞通信耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
    }
}
// MPI非阻塞通信+neon
void mpi_async_neon()
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num_proc * rank;
    int end = (rank == num_proc - 1) ? N : N / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init();
        MPI_Request *request = new MPI_Request[N - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (N / num_proc), e = (j == num_proc - 1) ? N : (j + 1) * (N / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset();
        MPI_Request *request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (int k = 0; k < N; k++)
    {
        {
            if ((k >= begin && k < end))
            {
                float32x4_t vt = vdupq_n_f32(A[k][k]);
                int j;
                for (j = k + 1; j + 4 <= N; j += 4)
                {
                    float32x4_t va = vld1q_f32(&A[k][j]);
                    va = vdivq_f32(va, vt);
                    vst1q_f32(&A[k][j], va);
                }
                for (; j < N; j++)
                {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request *request = new MPI_Request[num_proc - 1 - rank]; // 非阻塞传递
                for (j = 0; j < num_proc; j++)
                { // 块划分中，已经消元好且进行了除法置1的行向量仅

                    MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]); // 0号消息表示除法完毕
                }
                MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else
            {
                int src;
                if (k < N / num_proc * num_proc) // 在可均分的任务量内
                    src = k / (N / num_proc);
                else
                    src = num_proc - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE); // 实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
        for (int i = max(begin, k + 1); i < end; i++)
        {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4)
            {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步
    if (rank == num_proc - 1)
    {
        end_time = MPI_Wtime();
        cout << "MPI块划分+非阻塞通信+neon耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
    }
}
// MPI非阻塞通信+OpenMP
void mpi_async_omp()
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num_proc * rank;
    int end = (rank == num_proc - 1) ? N : N / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init();
        MPI_Request *request = new MPI_Request[N - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (N / num_proc), e = (j == num_proc - 1) ? N : (j + 1) * (N / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset();
        MPI_Request *request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
    for (int k = 0; k < N; k++)
    {
#pragma omp single
        {
            if ((k >= begin && k < end))
            {
                for (int j = k + 1; j < N; j++)
                {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request *request = new MPI_Request[num_proc - 1 - rank]; // 非阻塞传递
                for (int j = 0; j < num_proc; j++)
                { // 块划分中，已经消元好且进行了除法置1的行向量仅

                    MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]); // 0号消息表示除法完毕
                }
                MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else
            {
                int src;
                if (k < N / num_proc * num_proc) // 在可均分的任务量内
                    src = k / (N / num_proc);
                else
                    src = num_proc - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE); // 实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
#pragma omp for schedule(guided) // 向导调度
        for (int i = max(begin, k + 1); i < end; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步
    if (rank == num_proc - 1)
    {
        end_time = MPI_Wtime();
        cout << "MPI块划分+非阻塞通信+OpenMP耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
    }
}
// MPI非阻塞通信+neon+OpenMP
void mpi_async_neon_omp()
{
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num_proc * rank;
    int end = (rank == num_proc - 1) ? N : N / num_proc * (rank + 1);

    if (rank == 0)
    { // 0号进程初始化矩阵
        init();
        MPI_Request *request = new MPI_Request[N - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (N / num_proc), e = (j == num_proc - 1) ? N : (j + 1) * (N / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); // 等待传递
    }
    else
    {
        reset();
        MPI_Request *request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
    for (int k = 0; k < N; k++)
    {
#pragma omp single
        {
            if ((k >= begin && k < end))
            {
                float32x4_t vt = vdupq_n_f32(A[k][k]);
                int j;
                for (j = k + 1; j + 4 <= N; j += 4)
                {
                    float32x4_t va = vld1q_f32(&A[k][j]);
                    va = vdivq_f32(va, vt);
                    vst1q_f32(&A[k][j], va);
                }
                for (; j < N; j++)
                {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request *request = new MPI_Request[num_proc - 1 - rank]; // 非阻塞传递
                for (j = 0; j < num_proc; j++)
                { // 块划分中，已经消元好且进行了除法置1的行向量仅

                    MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]); // 0号消息表示除法完毕
                }
                MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else
            {
                int src;
                if (k < N / num_proc * num_proc) // 在可均分的任务量内
                    src = k / (N / num_proc);
                else
                    src = num_proc - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE); // 实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
#pragma omp for schedule(guided) // 开始多线程
        for (int i = max(begin, k + 1); i < end; i++)
        {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4)
            {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步
    if (rank == num_proc - 1)
    {
        end_time = MPI_Wtime();
        cout << "MPI块划分+非阻塞通信+OpenMP+neon耗时：" << 1000 * (end_time - start_time) << "ms" << endl;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int N_values[] = {500, 1000, 1500, 2000, 2500};
    for (int i = 0; i < sizeof(N_values) / sizeof(N_values[0]); ++i)
    {
        N = N_values[i];
        cout << "数据规模: " << N << endl;
        normal(N);             // 平凡算法
        mpi_block(N);          // MPI块划分
        mpi_circle(N);         // MPI循环划分
        mpi_block_neon();      // MPI块划分+neon
        mpi_block_omp();       // MPI块划分+OpenMP
        mpi_block_neon_omp();  // MPI块划分+neon+OpenMP
        mpi_async(N);          // MPI块划分+非阻塞通信
        mpi_async_neon(N);     // MPI块划分+非阻塞通信+neon
        mpi_async_omp(N);      // MPI块划分+非阻塞通信+OpenMP
        mpi_async_neon_omp(N); // MPI块划分+非阻塞通信+OpenMP+neon
    }
    MPI_Finalize();
    return 0;
}
