//
// Created by Nikita on 05.02.2021.
//

#ifndef COINMINER_SHA1_H
#define COINMINER_SHA1_H

#include "sha1_util.h"
#include "string_util.h"

struct sha1 {
    unsigned __int32 digest[5];
    char* buffer;
    unsigned __int64 transforms;
    unsigned __int32 buff_size;

    unsigned __int32* block; // non copy
};

inline static void reset(struct sha1* _sha1){
    _sha1->digest[0] = 0x67452301;
    _sha1->digest[1] = 0xefcdab89;
    _sha1->digest[2] = 0x98badcfe;
    _sha1->digest[3] = 0x10325476;
    _sha1->digest[4] = 0xc3d2e1f0;
    _sha1->buffer[0] = '\0';

    /* Reset counters */
    _sha1->buff_size = 0;
    _sha1->transforms = 0;
}

inline static struct sha1* newSHA1() {
    struct sha1* _sha1 = (struct sha1*)malloc(sizeof(struct sha1));

    _sha1->buffer = (char*)malloc(BLOCK_BYTES + 1);
    _sha1->buffer[BLOCK_BYTES] = '\0';

    _sha1->block = (unsigned __int32*)malloc(BLOCK_BYTES);

    reset(_sha1);

    return _sha1;
}

void string_copy( const char *from, char *to ) {
    for ( char *p = to; ( *p = *from ) != '\0'; ++p, ++from)
    {
        ;
    }
}

inline static void copy_buffer(const char* from, char* to){
    to[0]  = from[0];   to[16] = from[16];      to[32] = from[32];   to[48] = from[48];
    to[1]  = from[1];   to[17] = from[17];      to[33] = from[33];   to[49] = from[49];
    to[2]  = from[2];   to[18] = from[18];      to[34] = from[34];   to[50] = from[50];
    to[3]  = from[3];   to[19] = from[19];      to[35] = from[35];   to[51] = from[51];

    to[4]  = from[4];   to[20] = from[20];      to[36] = from[36];   to[52] = from[52];
    to[5]  = from[5];   to[21] = from[21];      to[37] = from[37];   to[53] = from[53];
    to[6]  = from[6];   to[22] = from[22];      to[38] = from[38];   to[54] = from[54];
    to[7]  = from[7];   to[23] = from[23];      to[39] = from[39];   to[55] = from[55];

    to[8]  = from[8];   to[24] = from[24];      to[40] = from[40];   to[56] = from[56];
    to[9]  = from[9];   to[25] = from[25];      to[41] = from[41];   to[57] = from[57];
    to[10] = from[10];  to[26] = from[26];      to[42] = from[42];   to[58] = from[58];
    to[11] = from[11];  to[27] = from[27];      to[43] = from[43];   to[59] = from[59];

    to[12] = from[12];  to[28] = from[28];      to[44] = from[44];   to[60] = from[60];
    to[13] = from[13];  to[29] = from[29];      to[45] = from[45];   to[61] = from[61];
    to[14] = from[14];  to[30] = from[30];      to[46] = from[46];   to[62] = from[62];
    to[15] = from[15];  to[31] = from[31];      to[47] = from[47];   to[63] = from[63];
}

inline static void copySHA1(struct sha1* from, struct sha1* to) {
    copy_buffer(from->buffer, to->buffer);
    to->buff_size = from->buff_size;
    to->digest[0] = from->digest[0];
    to->digest[1] = from->digest[1];
    to->digest[2] = from->digest[2];
    to->digest[3] = from->digest[3];
    to->digest[4] = from->digest[4];
    to->transforms = from->transforms;
}

//inline static void transform(struct sha1* sha1, unsigned __int32 block[BLOCK_INTS]){
inline static void transform(struct sha1* sha1, unsigned __int32* block){
    /* Copy digest[] to working vars */
    unsigned __int32 a = sha1->digest[0];
    unsigned __int32 b = sha1->digest[1];
    unsigned __int32 c = sha1->digest[2];
    unsigned __int32 d = sha1->digest[3];
    unsigned __int32 e = sha1->digest[4];

    /* 4 rounds of 20 operations each. Loop unrolled. */
    R0(block, a, &b, c, d, &e,  0);
    R0(block, e, &a, b, c, &d,  1);
    R0(block, d, &e, a, b, &c,  2);
    R0(block, c, &d, e, a, &b,  3);
    R0(block, b, &c, d, e, &a,  4);
    R0(block, a, &b, c, d, &e,  5);
    R0(block, e, &a, b, c, &d,  6);
    R0(block, d, &e, a, b, &c,  7);
    R0(block, c, &d, e, a, &b,  8);
    R0(block, b, &c, d, e, &a,  9);
    R0(block, a, &b, c, d, &e, 10);
    R0(block, e, &a, b, c, &d, 11);
    R0(block, d, &e, a, b, &c, 12);
    R0(block, c, &d, e, a, &b, 13);
    R0(block, b, &c, d, e, &a, 14);
    R0(block, a, &b, c, d, &e, 15);
    R1(block, e, &a, b, c, &d,  0);
    R1(block, d, &e, a, b, &c,  1);
    R1(block, c, &d, e, a, &b,  2);
    R1(block, b, &c, d, e, &a,  3);
    R2(block, a, &b, c, d, &e,  4);
    R2(block, e, &a, b, c, &d,  5);
    R2(block, d, &e, a, b, &c,  6);
    R2(block, c, &d, e, a, &b,  7);
    R2(block, b, &c, d, e, &a,  8);
    R2(block, a, &b, c, d, &e,  9);
    R2(block, e, &a, b, c, &d, 10);
    R2(block, d, &e, a, b, &c, 11);
    R2(block, c, &d, e, a, &b, 12);
    R2(block, b, &c, d, e, &a, 13);
    R2(block, a, &b, c, d, &e, 14);
    R2(block, e, &a, b, c, &d, 15);
    R2(block, d, &e, a, b, &c,  0);
    R2(block, c, &d, e, a, &b,  1);
    R2(block, b, &c, d, e, &a,  2);
    R2(block, a, &b, c, d, &e,  3);
    R2(block, e, &a, b, c, &d,  4);
    R2(block, d, &e, a, b, &c,  5);
    R2(block, c, &d, e, a, &b,  6);
    R2(block, b, &c, d, e, &a,  7);
    R3(block, a, &b, c, d, &e,  8);
    R3(block, e, &a, b, c, &d,  9);
    R3(block, d, &e, a, b, &c, 10);
    R3(block, c, &d, e, a, &b, 11);
    R3(block, b, &c, d, e, &a, 12);
    R3(block, a, &b, c, d, &e, 13);
    R3(block, e, &a, b, c, &d, 14);
    R3(block, d, &e, a, b, &c, 15);
    R3(block, c, &d, e, a, &b,  0);
    R3(block, b, &c, d, e, &a,  1);
    R3(block, a, &b, c, d, &e,  2);
    R3(block, e, &a, b, c, &d,  3);
    R3(block, d, &e, a, b, &c,  4);
    R3(block, c, &d, e, a, &b,  5);
    R3(block, b, &c, d, e, &a,  6);
    R3(block, a, &b, c, d, &e,  7);
    R3(block, e, &a, b, c, &d,  8);
    R3(block, d, &e, a, b, &c,  9);
    R3(block, c, &d, e, a, &b, 10);
    R3(block, b, &c, d, e, &a, 11);
    R4(block, a, &b, c, d, &e, 12);
    R4(block, e, &a, b, c, &d, 13);
    R4(block, d, &e, a, b, &c, 14);
    R4(block, c, &d, e, a, &b, 15);
    R4(block, b, &c, d, e, &a,  0);
    R4(block, a, &b, c, d, &e,  1);
    R4(block, e, &a, b, c, &d,  2);
    R4(block, d, &e, a, b, &c,  3);
    R4(block, c, &d, e, a, &b,  4);
    R4(block, b, &c, d, e, &a,  5);
    R4(block, a, &b, c, d, &e,  6);
    R4(block, e, &a, b, c, &d,  7);
    R4(block, d, &e, a, b, &c,  8);
    R4(block, c, &d, e, a, &b,  9);
    R4(block, b, &c, d, e, &a, 10);
    R4(block, a, &b, c, d, &e, 11);
    R4(block, e, &a, b, c, &d, 12);
    R4(block, d, &e, a, b, &c, 13);
    R4(block, c, &d, e, a, &b, 14);
    R4(block, b, &c, d, e, &a, 15);

    /* Add the working vars back into digest[] */
    sha1->digest[0] += a;
    sha1->digest[1] += b;
    sha1->digest[2] += c;
    sha1->digest[3] += d;
    sha1->digest[4] += e;

    /* Count the number of transformations */
    sha1->transforms++;
}

inline static void buffer_to_block(const char* buffer, unsigned __int32 block[BLOCK_INTS]) {
    /* Convert the const char* (byte buffer) to a unsigned __int32 array (MSB) */
    for (signed int i = 0; i < BLOCK_INTS; i++) {
        block[i] = (buffer[4*i+3] & 0xff)
            | (buffer[4*i+2] & 0xff) << 8
            | (buffer[4*i+1] & 0xff) << 16
            | (buffer[4*i+0] & 0xff) << 24;
    }
}

inline static void update(struct sha1* sha1, const char* str) {
    unsigned short position = 0;
    unsigned short size = fast_strlen(str);
    unsigned char count = 0;

    while (1) {
        if (size - position > BLOCK_BYTES){
            count = BLOCK_BYTES;
        } else
            count = size - position;

        //if (!sha1->buffer) {
        if (sha1->buffer[0] == '\0') {
            //sha1->buffer = (char *) malloc(sizeof(char) * BLOCK_BYTES + 1);
            sha1->buff_size = count;
            read_string(sha1->buffer, str + position, count);
            sha1->buffer[count] = '\0';
            position += count;
        } else {
            read_string(sha1->buffer + sha1->buff_size, str + position, count);
            sha1->buff_size += count;
            sha1->buffer[sha1->buff_size] = '\0';
            position += count;
        }

        //printf("%s\n", sha1->buffer);

        if (sha1->buff_size != BLOCK_BYTES)
            return;

        unsigned __int32 block[BLOCK_INTS];
        buffer_to_block(sha1->buffer, block);
        transform(sha1, block);

        //free(sha1->buffer);

        //sha1->buffer = NULL;
        sha1->buffer[0] = '\0';
        sha1->buff_size = 0;
    }
}

inline static void final(struct sha1* sha1, char* result) {
    /* Total number of hashed bits */
    unsigned __int64 total_bits = (sha1->transforms * BLOCK_BYTES + sha1->buff_size) * 8;
    // Padding
    unsigned __int32 orig_size = sha1->buff_size; // buff size is <= 63
    sha1->buffer[orig_size] = (char) 0x80;
    sha1->buff_size++;
    while (sha1->buff_size < BLOCK_BYTES) {
        sha1->buffer[sha1->buff_size] = (char) 0x00;
        sha1->buff_size++;
    }

    buffer_to_block(sha1->buffer, sha1->block);

    if (orig_size > BLOCK_BYTES - 8) {
        transform(sha1, sha1->block);
        for (size_t i = 0; i < BLOCK_INTS - 2; i++)
            sha1->block[i] = 0;
    }

    // Append total_bits, split this uint64_t into two uint32_t
    sha1->block[BLOCK_INTS - 1] = (unsigned __int32)total_bits;
    sha1->block[BLOCK_INTS - 2] = (unsigned __int32)(total_bits >> 32);
    //transform(sha1, block);
    transform(sha1, sha1->block);

    //unsigned int hex_count = sizeof(sha1->digest) / sizeof(sha1->digest[0]);
    result[5 * 8] = '\0';

    //if (hex_count != 5)
    //    printf("%u", hex_count);

    char hex[8];

    //for (int i = 0; i < hex_count; i++) {
    for (int i = 0; i < 5; i++) {
        //to_hex(hex, sha1->digest[i]);
        //UlongToHexString((unsigned __int64)sha1->digest[i], hex);
        lutHexString(sha1->digest[i], hex);

        result[0 + 8 * i] = hex[0];
        result[1 + 8 * i] = hex[1];
        result[2 + 8 * i] = hex[2];
        result[3 + 8 * i] = hex[3];

        result[4 + 8 * i] = hex[4];
        result[5 + 8 * i] = hex[5];
        result[6 + 8 * i] = hex[6];
        result[7 + 8 * i] = hex[7];
    }
}

#endif //COINMINER_SHA1_H
