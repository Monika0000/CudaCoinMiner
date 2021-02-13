#include <stdio.h>
// #include "unit_test.h"
#include <WinSock2.h>
#include "miner.cu"

DWORD WINAPI run_miner() {
    cudaSetDevice(0);

    /*if (!check_sha1() || !check_sha1_2() || !check_sha1_3() || !check_sha1_4()) {
        printf("SHA1 is not working!\n");
        return -1;
    }
    else
        printf("Checking SHA1 is successful\n");*/

    SOCKET sock = connect_to_server("51.15.127.80", 2811);
    // SOCKET sock = connect_to_server("51.195.65.23", 9999);

    if (sock == INVALID_SOCKET)
        return -1;

    unsigned int result = 0;

    unsigned int* dev_result = NULL;
    cudaMalloc((void**)&dev_result, sizeof(unsigned int));
    cudaError_t cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("dev_result malloc error: %s\n", cudaGetErrorString(cudaerror));
    }
    
    char* dev_prefix = NULL;
    cudaMalloc((void**)&dev_prefix, 41);

    byte* dev_target = NULL;
    cudaMalloc((void**)&dev_target, 20);

    unsigned int* dev_diff = NULL;
    cudaMalloc((void**)&dev_diff, sizeof(unsigned int));

    while (true) {
        if (request_job(sock, 3)) {
            result = process_job(sock, dev_result, dev_prefix, dev_target, dev_diff);
            send_job(sock, result);
        }
        else
            break;
    }

    cudaFree(dev_result);
    cudaFree(dev_prefix);
    cudaFree(dev_target);
    cudaFree(dev_diff);

    return 0;
}

int main() {
    cudaSetDevice(1);

    for (int i = 0; i < 4; i++) {
        CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)&run_miner, NULL, 0, NULL);
    }

    getchar();

    return 0;
}

/*
unsigned int* dev_result = NULL;
cudaMalloc((void**)&dev_result, sizeof(unsigned int));

char* dev_prefix = NULL;
cudaMalloc((void**)dev_prefix, 41);
cudaMemcpy(dev_prefix, prefix, 41, cudaMemcpyHostToDevice);

char* dev_target = NULL;
cudaMalloc((void**)dev_target, 41);
cudaMemcpy(dev_target, job, 41, cudaMemcpyHostToDevice);

unsigned int* dev_diff = NULL;
cudaMalloc((void**)dev_diff, sizeof(unsigned int));
cudaMemcpy(dev_diff, &diff, sizeof(unsigned int), cudaMemcpyHostToDevice);

sha1Kernel<<<1000, 256>>>(dev_result, dev_prefix, dev_target, dev_diff);
cudaDeviceSynchronize();

cudaMemcpy(&result, dev_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);


cudaFree(dev_result);
*/