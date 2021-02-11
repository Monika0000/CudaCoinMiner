#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#pragma comment(lib,"Ws2_32.lib")

#include <string.h>
#include <windows.h>  //Äëÿ sleep

#include <winsock2.h>

#include "string_util.h"

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

__global__ void findKernel(unsigned int* gpu_result) {
    unsigned int i = threadIdx.x;
    if (i == 500)
        *gpu_result = i;
}

unsigned int process_job(SOCKET sock, struct sha1* sha1, struct sha1* sha1_copy) {
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

    //printf("%s\n%s\n%i\n", prefix, job, diff);

    unsigned int* dev_result = NULL;
    cudaMalloc((void**)&dev_result, sizeof(unsigned int));



    cudaFree(dev_result);

    return 0;
}