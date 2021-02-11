
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "unit_test.h"

__global__ void findKernel(unsigned int* gpu_result) {
    unsigned int i = threadIdx.x;
    if (i == 500)
        *gpu_result = i;
}

int main() {
    if (!check_sha1() || !check_sha1_2() || !check_sha1_3() || !check_sha1_4()) {
        printf("SHA1 is not working!\n");
        return -1;
    }
    else
        printf("Checking SHA1 is successful\n");

    unsigned int result = 0;

    cudaSetDevice(0);

    unsigned int* dev_result = NULL;
    cudaMalloc((void**)&dev_result, sizeof(unsigned int));

    findKernel<<<1, 1000>>>(dev_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, dev_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);


    printf("%i", result);
    cudaFree(dev_result); 
}
