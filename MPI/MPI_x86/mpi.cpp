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
int n;                // ��������С
#define NUM_THREADS 7 // �����߳�����

void init(float**& A, int n) {
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
// ƽ���㷨
void normal(int n) {
    double start_time;
    double end_time;
    init(A, n);
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++) {
        // ��������
        for (int j = k + 1; j < n; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        // ��ȥ����
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    cout << "ƽ���㷨��ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
    deleteA(A, n);

}
 //MPI�黮��
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
    if (rank == 0) {  //0�Ž��̳�ʼ������
        init(A, n);

        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);
            for (int i = b; i < e; i++) {
                MPI_Send(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1�ǳ�ʼ������Ϣ����ÿ�����̷�������
            }
        }
    }
    else {
        reset(A, n);
        for (int i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);// ͬ��  
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++) {
        if ((k >= begin && k < end)) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++) { 
                if (j != rank)
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0����Ϣ��ʾ�������
            }
        }
        else {
            int src;
            if (k < n / num_proc * num_proc)//�ڿɾ��ֵ���������
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
    MPI_Barrier(MPI_COMM_WORLD);// ͬ��	
    if (rank == 0) {//0�Ž����д������ս��
        end_time = MPI_Wtime();
        cout << "MPI�黮�ֺ�ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//MPIѭ������
void mpi_circle(int n) {
    double start_time;
    double end_time;
    int num_proc;
    int rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {  //0�Ž��̳�ʼ������
        init(A, n);

        for (int j = 1; j < num_proc; j++) {
            for (int i = j; i < n; i += num_proc) {
                MPI_Send(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1�ǳ�ʼ������Ϣ����ÿ�����̷�������
            }
        }
    }
    else {
        reset(A, n);
        for (int i = rank; i < n; i += num_proc) {
            MPI_Recv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);// ͬ��
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++) {
        if (k % num_proc == rank) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++) {
                if (j != rank)
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0����Ϣ��ʾ�������
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
    MPI_Barrier(MPI_COMM_WORLD);// ͬ��	
    if (rank == 0) {//0�Ž����д������ս��
        end_time = MPI_Wtime();
        cout << "MPIѭ�����ֺ�ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//=====================================================================================
// MPI�黮��+avx
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
    { // 0�Ž��̳�ʼ������
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);
            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // ���������ݾ�������
            }
        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); // �ȴ�����
    }
    else
    {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // ����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // ͬ��
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
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0����Ϣ��ʾ�������
            }
        }
        else
        {
            int src;
            if (k < n / num_proc * num_proc) // �ڿɾ��ֵ���������
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
    MPI_Barrier(MPI_COMM_WORLD); // ͬ��
    if (rank == 0)
    { // 0�Ž����д������ս��
        end_time = MPI_Wtime();
        cout << "MPI�黮��+avx��ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
// MPI�黮��+OpenMP
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
    { // 0�Ž��̳�ʼ������
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // ���������ݾ�������
            }
        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); // �ȴ�����
    }
    else
    {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // ����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // ͬ��
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
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0����Ϣ��ʾ�������
            }
        }
        else
        {
            int src;
            if (k < n / num_proc * num_proc) // �ڿɾ��ֵ���������
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
    MPI_Barrier(MPI_COMM_WORLD); // ͬ��
    if (rank == 0)
    { // 0�Ž����д������ս��
        end_time = MPI_Wtime();
        cout << "MPI�黮��+OpenMP��ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
// MPI�黮��+avx+OpenMP
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
    { // 0�Ž��̳�ʼ������
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++)
        {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // ���������ݾ�������
            }
        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); // �ȴ�����
    }
    else
    {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++)
        {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); // ����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD); // ͬ��
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
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0����Ϣ��ʾ�������
            }
        }
        else
        {
            int src;
            if (k < n / num_proc * num_proc) // �ڿɾ��ֵ���������
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
    MPI_Barrier(MPI_COMM_WORLD); // ͬ��
    if (rank == 0)
    { // 0�Ž����д������ս��
        end_time = MPI_Wtime();
        cout << "MPI�黮��+avx+OpenMP��ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//=====================================================================================
//MPI�黮��+������ͨ��
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

    if (rank == 0) {  //0�Ž��̳�ʼ������
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//���������ݾ�������
            }

        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); //�ȴ�����

    }
    else {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);// ͬ��
    start_time = MPI_Wtime();
    for (int k = 0; k < n; k++) {
        if ((k >= begin && k < end)) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[num_proc - 1 - rank];  //����������
            for (int j = rank + 1; j < num_proc; j++) { //�黮���У��Ѿ���Ԫ���ҽ����˳�����1����������

                MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0����Ϣ��ʾ�������
            }
            MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; //��ִ������������񣬿�ֱ������
        }
        else {
            int src = k / (n / num_proc);
            MPI_Request request;
            MPI_Irecv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         //ʵ������Ȼ���������գ���Ϊ�������Ĳ�����Ҫ��Щ����
        }
        for (int i = max(begin, k + 1); i < end; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);// ͬ��	
    if (rank == num_proc - 1) {
        end_time = MPI_Wtime();
        cout << "MPI�黮��+������ͨ�ź�ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//MPI������ͨ��+avx
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

    if (rank == 0) {  //0�Ž��̳�ʼ������
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//���������ݾ�������
            }
        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); //�ȴ�����
    }
    else {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);// ͬ��
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
                MPI_Request* request = new MPI_Request[num_proc - 1 - rank];  //����������
                for (j = 0; j < num_proc; j++) { //�黮���У��Ѿ���Ԫ���ҽ����˳�����1����������

                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0����Ϣ��ʾ�������
                }
                MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < n / num_proc * num_proc)//�ڿɾ��ֵ���������
                    src = k / (n / num_proc);
                else
                    src = num_proc - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //ʵ������Ȼ���������գ���Ϊ�������Ĳ�����Ҫ��Щ����
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
    MPI_Barrier(MPI_COMM_WORLD);// ͬ��
    if (rank == num_proc - 1) {
        end_time = MPI_Wtime();
        cout << "MPI�黮��+������ͨ��+avx��ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//MPI������ͨ��+OpenMP
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

    if (rank == 0) {  //0�Ž��̳�ʼ������
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//���������ݾ�������
            }

        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); //�ȴ�����

    }
    else {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);// ͬ��
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
                MPI_Request* request = new MPI_Request[num_proc - 1 - rank];  //����������
                for (int j = 0; j < num_proc; j++) { //�黮���У��Ѿ���Ԫ���ҽ����˳�����1����������

                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0����Ϣ��ʾ�������
                }
                MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < n / num_proc * num_proc)//�ڿɾ��ֵ���������
                    src = k / (n / num_proc);
                else
                    src = num_proc - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //ʵ������Ȼ���������գ���Ϊ�������Ĳ�����Ҫ��Щ����
            }
        }
#pragma omp for schedule(guided)  //�򵼵���
        for (int i = max(begin, k + 1); i < end; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);// ͬ��	
    if (rank == num_proc - 1) {
        end_time = MPI_Wtime();
        cout << "MPI�黮��+������ͨ��+OpenMP��ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}
//MPI������ͨ��+avx+OpenMP
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

    if (rank == 0) {  //0�Ž��̳�ʼ������
        init(A, n);
        MPI_Request* request = new MPI_Request[n - end];
        for (int j = 1; j < num_proc; j++) {
            int b = j * (n / num_proc), e = (j == num_proc - 1) ? n : (j + 1) * (n / num_proc);

            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], n, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//���������ݾ�������
            }

        }
        MPI_Waitall(n - end, request, MPI_STATUS_IGNORE); //�ȴ�����

    }
    else {
        reset(A, n);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);// ͬ��
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
                MPI_Request* request = new MPI_Request[num_proc - 1 - rank];  //����������
                for (j = 0; j < num_proc; j++) { //�黮���У��Ѿ���Ԫ���ҽ����˳�����1����������

                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0����Ϣ��ʾ�������
                }
                MPI_Waitall(num_proc - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < n / num_proc * num_proc)//�ڿɾ��ֵ���������
                    src = k / (n / num_proc);
                else
                    src = num_proc - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], n, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //ʵ������Ȼ���������գ���Ϊ�������Ĳ�����Ҫ��Щ����
            }
        }
#pragma omp for schedule(guided)  //��ʼ���߳�
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
    MPI_Barrier(MPI_COMM_WORLD);// ͬ��
    if (rank == num_proc - 1) {
        end_time = MPI_Wtime();
        cout << "MPI�黮��+������ͨ��+avx+OpenMP��ʱ��" << 1000 * (end_time - start_time) << "ms" << endl;
        deleteA(A, n);
    }
}


int main(int argc, char* argv[]) {
    
    MPI_Init(&argc, &argv);
    // �Բ�ͬ�����ݹ�ģ���в���
    for (int sizes : {500, 1000, 1500, 2000, 2500})
    {
        n = sizes;
        cout << "Sizes: " << n << endl;
        normal(n);//ƽ���㷨
        mpi_block(n);//MPI�黮��
        mpi_circle(n);//MPIѭ������
        mpi_block_avx(n);//MPI�黮��+avx
        mpi_block_omp(n);//MPI�黮��+OpenMP
        mpi_block_neon_omp(n);//MPI�黮��+avx+OpenMP
        mpi_async(n);//MPI�黮��+������ͨ��
        mpi_async_avx(n);//MPI�黮��+������ͨ��+avx
        mpi_async_omp(n);//MPI�黮��+������ͨ��+OpenMP
        mpi_async_avx_omp(n);//MPI�黮��+������ͨ��+avx+OpenMP
    }
    MPI_Finalize();
    return 0;
}
