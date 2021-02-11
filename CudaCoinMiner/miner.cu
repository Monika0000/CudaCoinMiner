#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#pragma comment(lib,"Ws2_32.lib")

#include <string.h>
#include <windows.h>  //Äëÿ sleep

#include <winsock2.h>

#include "string_util.cu"
#include "sha1.cu"

#define CUDA_BLOCKS 10000
#define CUDA_THREADS 512
#define CUDA_TOTAL_THREADS CUDA_BLOCKS * CUDA_THREADS

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

    char* req = NULL;
    switch (diff) {
    case 0: req = make_req(username, "AVR\0"); break;
    case 1: req = make_req(username, "ESP\0"); break;
    case 2: req = make_req(username, "MEDIUM\0"); break;
    case 3: req = make_req(username, NULL); break;
    default:
        printf("Unknown diff!");
        return 0;
    }

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

// inline function to swap two numbers
__device__ inline void swap(char* x, char* y) {
    char t = *x; *x = *y; *y = t;
}

// function to reverse buffer[i..j]
__device__ char* reverse(char* buffer, int i, int j)
{
    while (i < j)
        swap(&buffer[i++], &buffer[j--]);

    return buffer;
}

// Iterative function to implement itoa() function in C
__device__ char* itoa(int value, char* buffer, int base)
{
    // invalid input
    if (base < 2 || base > 32)
        return buffer;

    // consider absolute value of number
    int n = abs(value);

    int i = 0;
    while (n)
    {
        int r = n % base;

        if (r >= 10)
            buffer[i++] = 65 + (r - 10);
        else
            buffer[i++] = 48 + r;

        n = n / base;
    }

    // if number is 0
    if (i == 0)
        buffer[i++] = '0';

    // If base is 10 and value is negative, the resulting string 
    // is preceded with a minus sign (-)
    // With any other base, value is always considered unsigned
    if (value < 0 && base == 10)
        buffer[i++] = '-';

    buffer[i] = '\0'; // null terminate string

    // reverse the string and return it
    return reverse(buffer, 0, i - 1);
}

__device__ int cuda_strcmp(const char* str_a, const char* str_b, unsigned len = 256) {
    int match = 0;
    unsigned i = 0;
    unsigned done = 0;
    while ((i < len) && (match == 0) && !done) {
        if ((str_a[i] == 0) || (str_b[i] == 0)) done = 1;
        else if (str_a[i] != str_b[i]) {
            match = i + 1;
            if (((int)str_a[i] - (int)str_b[i]) < 0)
                match = 0 - (i + 1);
        }
        i++;
    }
    return match;
}

__global__ void sha1Kernel(unsigned int* result, char* prefix, char* target, unsigned int* diff) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iterations_per_thread = 100 * (*diff) / CUDA_TOTAL_THREADS;
    unsigned int from = iterations_per_thread * index - iterations_per_thread;
    unsigned int to = 0;
    if (index == CUDA_TOTAL_THREADS - 1) {
        to = 100 * (*diff);
    }
    else {
        to = iterations_per_thread * index;
    }
    struct sha1* sha1 = newSHA1();
    struct sha1* sha1copy = newSHA1();

    update(sha1, prefix);
    for (unsigned int i = from; i <= to; i++) {
        char buffer[32];
        char final_hash[41];

        copySHA1(sha1, sha1copy);
        itoa(i, buffer, 10);
        update(sha1copy, buffer);
        final(sha1copy, final_hash);
        if (cuda_strcmp(final_hash, target, 8) == 0 ) {
            *result = i;
            return;
        }
        reset(sha1copy);        
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
    int diff = atoi(buffer + id);
    unsigned int* result = NULL;
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

    char* dev_target = NULL;
    cudaMalloc((void**)&dev_target, 41);
    cudaMemcpy(dev_target, &job, 41, cudaMemcpyHostToDevice);
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


    sha1Kernel <<<CUDA_BLOCKS, CUDA_THREADS>>> (dev_result, dev_prefix, dev_target, dev_diff);
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

    return 0;
}