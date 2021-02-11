#include <stdio.h>
// #include "unit_test.h"
#include <WinSock2.h>
#include "miner.cu"
#include "sha1.cu"

int main() {
    cudaSetDevice(0);

    /*if (!check_sha1() || !check_sha1_2() || !check_sha1_3() || !check_sha1_4()) {
        printf("SHA1 is not working!\n");
        return -1;
    }
    else
        printf("Checking SHA1 is successful\n");*/

    SOCKET sock = connect_to_server("51.15.127.80", 2811);
    if (sock == INVALID_SOCKET)
        return -1;

    if (request_job(sock, 3)) {
        int result = process_job(sock);
        printf("Result: %i\n", result);
    }


    //cudaSetDevice(0);
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