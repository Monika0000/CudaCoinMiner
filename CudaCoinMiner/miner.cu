#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdbool.h>

#pragma comment(lib,"Ws2_32.lib")

#include <string.h>
#include <windows.h>  //Äëÿ sleep

#include <winsock2.h>

#include "string_util.cu"
#include "sha1.c"

#define CUDA_THREADS 512

static char* username = NULL;

unsigned char request_job(SOCKET sock, unsigned char diff) {
    if (!username) {
        username = (char*)malloc(7);
        username[0] = 'M';
        username[1] = 'o';
        username[2] = 'n';
        username[3] = 'i';
        username[4] = 'k';
        username[5] = 'a';
        username[6] = '\0';
    }

    char* req = make_req(username, "EXTREME");

    if (send(sock, req, fast_strlen(req), 0) < 0) {
        printf("request_job() : failed\n");
        return 0;
    }
    else {
        free(req);
        return 1;
    }
}

SOCKET connect_to_server(char* ip, unsigned short port) {
    printf("connect_to_server(): connecting to %s:%hu...\n", ip, port);

    WSADATA wsa;
    SOCKET s;
    struct sockaddr_in server;
    char server_reply[4];

    printf("connect_to_server(): initializing socket...\n");
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        printf("connect_to_server(): failed. Error Code : %d", WSAGetLastError());
        return 0;
    }

    if ((s = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        printf("connect_to_server(): Could not create socket : %d", WSAGetLastError());
        return INVALID_SOCKET;
    }

    server.sin_addr.s_addr = inet_addr(ip);
    server.sin_family = AF_INET;
    server.sin_port = htons(port);

    //Connect to remote server
    if (connect(s, (struct sockaddr*)&server, sizeof(server)) < 0) {
        printf("connect_to_server(): connect error\n");
        return 0;
    }

    printf("connect_to_server(): connected successfully!\n");

    //Receive a reply from the server
    if ((recv(s, server_reply, 3, 0)) == SOCKET_ERROR) {
        printf("connect_to_server(): recv version failed\n");
        return 0;
    }

    server_reply[3] = '\0';

    printf("connect_to_server(): server version: "); printf("%s", server_reply); printf("\n");

    return s;
}

// Iterative function to implement itoa() function in C
__device__ char* cuda_itoa(char str[], int num)
{
    int i, rem, len = 0, n;

    n = num;
    while (n != 0)
    {
        len++;
        n /= 10;
    }
    for (i = 0; i < len; i++)
    {
        rem = num % 10;
        num = num / 10;
        str[len - (i + 1)] = rem + '0';
    }
    str[len] = '\0';
}

__device__ int cuda_bytecmp(const byte* str_a, const byte* str_b, unsigned int len) {
    for (int i = 0; i < len; i++) {
        if (str_a[i] != str_b[i]) {
            return false;
        }
    }
    return true;
}

__global__ void sha1Kernel(unsigned int* result, char* prefix, byte* target, unsigned int* diff) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (*result != 0 || index > *diff * 100) {
        return;
    }
    char buffer[32];
    byte final_hash[20];
    SHA1_CTX sha1;

    sha1_init(&sha1);
    sha1_update(&sha1, (const byte*)prefix, 40);

    cuda_itoa(buffer, index);
    sha1_update(&sha1, (const byte*)buffer, cuda_fast_strlen(buffer));

    sha1_final(&sha1, final_hash);

    if (cuda_bytecmp(final_hash, target, 10) == true) {
        *result = index;
    }
}


unsigned int process_job(SOCKET sock) {
    char buffer[100];
    int size = recv(sock, buffer, 100, 0);
    if (size == 0) {
        printf("process_job(): server return zero bytes!\n");
        return 0;
    }

    buffer[size] = '\0';

    if (buffer[0] == 'B' && buffer[1] == 'A' && buffer[2] == 'D' && buffer[3] == '\0')
        return 0;

    unsigned short id = 0;
    char* prefix = read_to(buffer, ',', &id);
    char* job = read_to(buffer + id, ',', &id);
    byte target[20];
    hexToBytes(target, job);
    int diff = atoi(buffer + id);
    unsigned int result = 0;
    cudaError_t cudaerror;

    //printf("%s\n%s\n%i\n", prefix, job, diff);


    unsigned int* dev_result = NULL;
    cudaMalloc((void**)&dev_result, sizeof(unsigned int));
    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("dev_result malloc error: %s\n", cudaGetErrorString(cudaerror));
    }

    char* dev_prefix = NULL;
    cudaMalloc((void**)&dev_prefix, 41);
    cudaMemcpy(dev_prefix, prefix, 41, cudaMemcpyHostToDevice);
    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("dev_prefix malloc error: %s\n", cudaGetErrorString(cudaerror));
    }

    byte* dev_target = NULL;
    cudaMalloc((void**)&dev_target, 41);
    cudaMemcpy(dev_target, &target, 41, cudaMemcpyHostToDevice);
    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("dev_target malloc error: %s\n", cudaGetErrorString(cudaerror));
    }

    unsigned int* dev_diff = NULL;
    cudaMalloc((void**)&dev_diff, sizeof(unsigned int));
    cudaMemcpy(dev_diff, &diff, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("dev_diff malloc error: %s\n", cudaGetErrorString(cudaerror));
    }


    sha1Kernel <<<(unsigned long)((100 * diff) / CUDA_THREADS) + 1, CUDA_THREADS>>> (dev_result, dev_prefix, dev_target, dev_diff);
    cudaDeviceSynchronize();

    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("sha1Kernel execution error: %s", cudaGetErrorString(cudaerror));
    }

    cudaMemcpy(&result, dev_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(dev_result);
    cudaFree(dev_diff);
    cudaFree(dev_prefix);
    cudaFree(dev_target);

    printf("%i\n", result);

    return result;
}