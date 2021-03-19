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
unsigned char threads_count = 0;


bool request_job(SOCKET sock) {
    //const char*

    char* req = NULL;
    req = make_req(username, NULL);

    // printf("%s\n", req);

    if (send(sock, req, fast_strlen(req), 0) < 0) {
        printf("request_job() : failed\n");
        return 0;
    }
    else {
        free(req);
        return 1;
    }
}

bool parse_args(int argc, char** argv) {
    for (unsigned char i = 1; i < (unsigned char)argc; i += 2) {
        if (strcmp(argv[i], "--threads") == 0) {
            threads_count = (unsigned char)atoi(argv[i + 1]);
            printf("Threads: %i\n", threads_count);
        }
        else if (strcmp(argv[i], "--user") == 0) {
            username = argv[i + 1];
            printf("User: %s\n", username);
        }
    }

    if (!username) {
        printf("parse_args(): use --user to set a username!\n");
        return 0;
    }

    printf("parse_args(): you have %i threads\n", threads_count);

    return 1;
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

__device__ __forceinline__ int cuda_bytecmp(register const byte* s1, register const byte* s2) {
    register unsigned char n = 11;
    do {
        if (*s1 != *s2++)
            return 0;
        if (*s1++ == 0)
            break;
    } while (--n != 0);
    return 1;
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

    if (cuda_bytecmp(final_hash, target) == true) {
        *result = index;
    }

}


unsigned int process_job(SOCKET sock, unsigned int* dev_result, char* dev_prefix, byte* dev_target, unsigned int* dev_diff) {
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

    // printf("%s\n%s\n%i\n", prefix, job, diff);

    cudaMemcpy(dev_prefix, prefix, 41, cudaMemcpyHostToDevice);
    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("dev_prefix malloc error: %s\n", cudaGetErrorString(cudaerror));
    }

    cudaMemcpy(dev_target, &target, 20, cudaMemcpyHostToDevice);
    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("dev_target malloc error: %s\n", cudaGetErrorString(cudaerror));
    }

    cudaMemcpy(dev_diff, &diff, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("dev_diff malloc error: %s\n", cudaGetErrorString(cudaerror));
    }

    cudaMemcpy(dev_result, &result, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("dev_diff malloc error: %s\n", cudaGetErrorString(cudaerror));
    }

    sha1Kernel <<<(unsigned long)((100 * diff) / CUDA_THREADS) + 1, CUDA_THREADS>>> (dev_result, dev_prefix, dev_target, dev_diff);
    cudaDeviceSynchronize();

    cudaerror = cudaGetLastError();
    if (cudaerror != cudaSuccess) {
        printf("sha1Kernel execution error: %s\nIt's possible that you got blocked by the server.\n", cudaGetErrorString(cudaerror));
        closesocket(sock);
        exit(-1);
    }

    cudaMemcpy(&result, dev_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    free(prefix);
    free(job);

    return result;
}

void send_job(SOCKET sock, unsigned int job) {
    char job_result[64];
    itoa(job, job_result, 10);
    strcat(job_result, ",,CoinMiner");

    if (send(sock, job_result, fast_strlen(job_result), 0) < 0) {
        printf("send_job() : failed\n");
    }

    char server_reply[6];
    if ((recv(sock, server_reply, 6, 0)) == SOCKET_ERROR) {
        printf("send_job(): recv version failed\n");
    }
    printf("%s - %s\n", job_result, server_reply);

    server_reply[0] = '\0';
    server_reply[1] = '\0';
    server_reply[2] = '\0';
    server_reply[3] = '\0';
    server_reply[4] = '\0';
    server_reply[5] = '\0';
}