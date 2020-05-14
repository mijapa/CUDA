#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <wait.h>
#include "Common/helper_timer.h"


void printResults(const int *a, const int *b, const int *c, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }
}

__global__ void addGPU(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void addCPU(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

bool checkIdentity(int *a, int *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

void test(FILE *fp, int n, int threadsPerBlock, int blocksPerGrid) {
    int *a = static_cast<int *>(malloc(n * sizeof(int)));
    int *b = static_cast<int *>(malloc(n * sizeof(int)));
    int *c = static_cast<int *>(malloc(n * sizeof(int)));
    int *d = static_cast<int *>(malloc(n * sizeof(int)));

    int *dev_a, *dev_b, *dev_c;
    //alokuj pamięć na GPU
    cudaMalloc((void **) &dev_a, n * sizeof(int));
    cudaMalloc((void **) &dev_b, n * sizeof(int));
    cudaMalloc((void **) &dev_c, n * sizeof(int));

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    //kopiuj z pamięci głównej komputera do pamięci GPU
    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, n * sizeof(int), cudaMemcpyHostToDevice);

    //przygotowanie i start timera
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    //wykonaj obliczenia na GPU
    addGPU <<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, n);

    //synchronizacja wątków i zatrzymanie timera
    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float timeGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);

    //prześlij wyniki do pamięci głownej
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);


//    printf("Execution time in CUDA: %f ms\n", time);

    clock_t tStart = clock();
    addCPU(a, b, d, n);
    clock_t tim = (clock() - tStart);
    float timeCPU = (float) tim / CLOCKS_PER_SEC * 1000;
//    printf("Execution time in CPU: %f ms\n", timeCPU);

    bool identity = checkIdentity(c, d, n);
    if (!identity) {
        printf("Calculation error in:\n");
        printf("N: %i, blocks per grid: %i, threads per block %i \n", n, blocksPerGrid, threadsPerBlock);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    fprintf(fp, "%i,%i,%i,%f,%f\n",
            n,
            blocksPerGrid,
            threadsPerBlock,
            timeCPU,
            timeGPU
    );
}

int main(void) {
    //changing N
    printf("Testing N\n");
    FILE *fp;
    if ((fp = fopen("times_N.csv", "a")) == NULL) {
        printf("Can't open .csv in append mode!\n");
        exit(1);
    }
    fprintf(fp, "n,blocksPerGrid,threadsPerBlock,timeCPU, timeGPU\n");

    int nn = 1000*1000;
    for (int n = 1; n <= nn; n += 10000) {
        int threadsPerBlock = 1024;
        int blocksPerGrid = ceil(1.0 * n / threadsPerBlock);
        test(fp, n, threadsPerBlock, blocksPerGrid);
    }

    fclose(fp);

    //N constant, change threadsPerBlock
    printf("Testing threadsPerBlock\n");
    if ((fp = fopen("times_threads.csv", "a")) == NULL) {
        printf("Can't open .csv in append mode!\n");
        exit(1);
    }
    fprintf(fp, "n,blocksPerGrid,threadsPerBlock,timeCPU, timeGPU\n");

    int maxThreadsPerBlock = 1024;
    int n = 1000*1000;
    for (int th = 1; th <= maxThreadsPerBlock; th += 32) {
        int blocksPerGrid = ceil(1.0 * n / th);
        test(fp, n, th, blocksPerGrid);
    }

    fclose(fp);

    //N constant, threadsPerBlock constant, change blocksPerGrid
    printf("Testing blocksPerGrid\n");
    if ((fp = fopen("times_blocks.csv", "a")) == NULL) {
        printf("Can't open .csv in append mode!\n");
        exit(1);
    }
    fprintf(fp, "n,blocksPerGrid,threadsPerBlock,timeCPU, timeGPU\n");

    int threadsPerBlock = 1024;
    int maxBlocksPerGrid = 100000;
    n = 100*1000;
    for (int bl = 100; bl <= maxBlocksPerGrid; bl += 1000) {
        test(fp, n, threadsPerBlock, bl);
    }

    fclose(fp);

    return 0;
}


