// Matrix multiplication by parts
// Elements stored in row-major order

using namespace std;

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "Common/helper_timer.h"

#define BLOCK_SIZE 32
#define GRID_SIZE 16

typedef struct {
    int width;
    int height;
    float *elements;
} Matrix;

// Forward declaration of matrix mult
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Host code
void MatMul(const Matrix A, const Matrix B, Matrix C) {
    // Load matrices A and B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc((void **) &d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc((void **) &d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // allocate C in device
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = d_C.width * d_C.height * sizeof(float);
    cudaMalloc((void **) &d_C.elements, size);

    // call kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // define the block size (what is the best value?)
    dim3 dimGrid(C.width / BLOCK_SIZE, C.height / BLOCK_SIZE); //  choose grid size depending on problem size

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // copy C to host
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

//matrix multiplication kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (row < A.height && col < B.width) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < A.width; i++) {
            tmpSum += A.elements[row * A.width + i] * B.elements[i * B.width + col];
        }
        C.elements[row * C.width + col] = tmpSum;
    }
}

//CPU matrix multiplication{
void CPUMatMull(Matrix A, Matrix B, Matrix C) {
    float sum;
    for (int row = 0; row < A.height; row++) {
        for (int col = 0; col < B.width; col++) {
            sum = 0.f;
            for (int n = 0; n < A.width; n++) {
                sum += A.elements[row * A.width + n] * B.elements[n * B.width + col];
            }
            C.elements[row * C.width + col] = sum;
        }
    }
}

// Check the result
void check(const Matrix A, const Matrix B) {

    double err = 0;
    // Check the result and make sure it is correct
    for (int row = 0; row < A.height; row++) {
        for (int col = 0; col < A.width; col++) {
            err += A.elements[row * A.height + col] - B.elements[row * B.width + col];
        }
    }

    cout << "Error: " << err << endl;
}

void printPerformance(float timeGPU, float timeCPU, double flopsPerMatrixMul, double gigaFlops, double gigaFlopsGPU) {
    printf(
            "CPU Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            timeCPU,
            flopsPerMatrixMul);

    printf(
            "GPU Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlopsGPU,
            timeGPU,
            flopsPerMatrixMul);
}

void expand(int Height, Matrix &A, Matrix &B) {
    int fileDim = 16;
    for (int i = 1; i <= (Height / fileDim) * (Height / fileDim); i++) {
        for (int j = 0; j < fileDim * fileDim; j++) {
            A.elements[i * fileDim * fileDim + j] = A.elements[j];
            B.elements[i * fileDim * fileDim + j] = B.elements[j];
        }
    }

//    for (int i = 0; i < Width; i++) {
//        for (int j = 0; j < Width; j++)
//            cout << A.elements[i * Width + j] << "\t";
//        cout << endl;
//    }

}

int main() {
    FILE *fp;
    if ((fp = fopen("times_N.csv", "a")) == NULL) {
        printf("Can't open .csv in append mode!\n");
        exit(1);
    }
    fprintf(fp, "n,blocksPerGrid,threadsPerBlock,timeCPU,timeGPU,gflopsCPU,gflopsGPU\n");

    int Height = 512;
    int Width = Height;

    Matrix A;
    Matrix B;
    Matrix C;
    Matrix D;

    A.width = Width;
    B.width = Width;
    C.width = Width;
    D.width = Width;

    A.height = Height;
    B.height = Height;
    C.height = Width;
    D.height = Width;

    A.elements = new float[Width * Width];
    B.elements = new float[Width * Width];
    C.elements = new float[Width * Width];
    D.elements = new float[Width * Width];

    //fill matrices
    std::ifstream A_input;
    std::ifstream B_input;
    A_input.open("A.txt");
    B_input.open("B.txt");

    float a, b;
    A_input >> a;
    B_input >> b;
    int i = 0;
    while (!A_input.eof()) {
        A.elements[i] = a;
        B.elements[i] = b;
        A_input >> a;
        B_input >> b;
        i += 1;
    }
    A_input.close();
    B_input.close();

    expand(Height, A, B);

//przygotowanie i start timera
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    MatMul(A, B, C);

    //synchronizacja wątków i zatrzymanie timera
    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float timeGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);

    std::ofstream C_output;
    C_output.open("C.txt");
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++)
            C_output << C.elements[i * Width + j] << "\t";
        C_output << endl;
    }


    clock_t tStart = clock();

    CPUMatMull(A, B, D);

    clock_t tim = (clock() - tStart);
    float timeCPU = (float) tim / CLOCKS_PER_SEC * 1000;

    double flopsPerMatrixMul = 2.0 * (double) A.width * (double) A.height * (double) B.width;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (timeCPU / 1000.0f);
    double gigaFlopsGPU = (flopsPerMatrixMul * 1.0e-9f) / (timeGPU / 1000.0f);
    printPerformance(timeGPU, timeCPU, flopsPerMatrixMul, gigaFlops, gigaFlopsGPU);

    int grid = C.width / BLOCK_SIZE;
    fprintf(fp, "%i,%i,%i,%f,%f,%f,%f\n",
            Width * Width,
            grid,
            BLOCK_SIZE,
            timeCPU,
            timeGPU,
            gigaFlops,
            gigaFlopsGPU
    );


    check(C, D);
}
	
