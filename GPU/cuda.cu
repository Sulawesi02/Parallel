#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<chrono>
#include<iomanip>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

using namespace std;
using namespace std::chrono;

int N = 1024;
int BLOCK_SIZE = 1024;
float** A;

//初始化矩阵
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

//普通消元算法
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

__global__ void division_kernel(float* data, int k, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;//计算线程索引
    int element = data[k * N + k];
    int temp = data[k * N + tid];
    //请同学们思考，如果分配的总线程数小于 N 应该怎么办
    data[k * N + tid] = (float)temp / element;
    return;
}

__global__ void eliminate_kernel(float* data, int k, int N) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx == 0)
        data[k * N + k] = 1.0;//对角线元素设为 1
    int row = k + 1 + blockIdx.x;//每个块负责一行
    while (row < N) {
        int tid = threadIdx.x;
        while (k + 1 + tid < N) {
            int col = k + 1 + tid;
            float temp_1 = data[(row * N) + col];
            float temp_2 = data[(row * N) + k];
            float temp_3 = data[k * N + col];
            data[(row * N) + col] = temp_1 - temp_2 * temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads();//块内同步
        if (threadIdx.x == 0) {
            data[row * N + k] = 0;
        }
        row += gridDim.x;
    }
    return;
}

int main() {
    int N_values[] = { 64, 128, 256, 512, 1024, 2048, 4096 };
    for (int i = 0; i < sizeof(N_values) / sizeof(N_values[0]); ++i) {
        N = N_values[i];
        cout << "数据规模: " << N << endl;

        //CPU
        init();
        auto start_time = high_resolution_clock::now();// 开始时间
        normal();
        auto end_time = high_resolution_clock::now();// 结束时间
        auto duration = duration_cast<nanoseconds>(end_time - start_time).count();// 计算时间差
        double duration_ms = duration / 1e6; // 将纳秒转换为毫秒
        cout << "CPU_LU:" << duration_ms << " ms" << endl;

        //GPU
        init();//初始化矩阵A
        float* temp = new float[N * N];
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                temp[i * N + j] = A[i][j];//赋值到一维数组temp中
            }
        }
        cudaError_t ret;//用于错误检查，当 CUDA 接口调用成功会返回 cudaSucess
        float* gpudata;
        float* result = new float[N * N];
        int size = N * N * sizeof(float);

        ret = cudaMalloc(&gpudata, size);//分配显存空间
        if (ret != cudaSuccess) {
            printf("cudaMalloc gpudata failed!\n");

        }
        ret = cudaMemcpy(gpudata, temp, size, cudaMemcpyHostToDevice);//将数据传输至 GPU 端

        if (ret != cudaSuccess) {
            printf("cudaMemcpyHostToDevice failed!\n");
        }

        dim3 dimBlock(BLOCK_SIZE, 1);//线程块
        dim3 dimGrid(1, 1);//线程网格

        cudaEvent_t start, stop;//计时器
        float elapsedTime = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);//开始计时

        for (int k = 0; k < N; k++) {
            division_kernel << <dimGrid, dimBlock >> > (gpudata, k, N);//负责除法任务的核函数
            cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
            ret = cudaGetLastError();
            if (ret != cudaSuccess) {
                printf("division_kernel failed, %s\n", cudaGetErrorString(ret));
            }
            eliminate_kernel << <dimGrid, dimBlock >> > (gpudata, k, N);//负责消去任务的核函数
            cudaDeviceSynchronize();
            ret = cudaGetLastError();
            if (ret != cudaSuccess) {
                printf("eliminate_kernel failed, %s\n", cudaGetErrorString(ret));
            }
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);//停止计时
        cudaEventElapsedTime(&elapsedTime, start, stop);

        printf("GPU_LU:%f ms\n", elapsedTime);

        cudaError_t cudaStatus2 = cudaGetLastError();
        if (cudaStatus2 != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
        }

        ret = cudaMemcpy(result, gpudata, size, cudaMemcpyDeviceToHost);//将数据传回 CPU 端
        if (ret != cudaSuccess) {
            printf("cudaMemcpyDeviceToHost failed!\n");
        }

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i][j] = result[i * N + j];//将result写回结果矩阵A
            }
        }

        cudaFree(gpudata);//释放显存空间，用 CUDA 接口分配的空间必须用 cudaFree 释放
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        delete[] temp;
        delete[] result;
    }
}
