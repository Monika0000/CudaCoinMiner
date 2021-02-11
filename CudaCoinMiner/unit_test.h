//
// Created by Monika on 07.02.2021.
//

#ifndef COINMINER_UNIT_TEST_H
#define COINMINER_UNIT_TEST_H

#include <malloc.h>
#include "sha1.h"

inline static unsigned char check_sha1() {
    struct sha1* _sha1 = newSHA1();
    //const char* checksum = "a57d9e8eaa343aed58a5b2e240336a929cafedc6\0";
    //const char* checksum = "da39a3ee5e6b4b0d3255bfef95601890afd80709\0";
    const char* checksum = "40bd001563085fc35165329ea1ff5c5ecbdbbeef\0";

    //nst char* input = "[11111111111111111111111111111111111111111111111111111111111111]\0";
    //const char* input = "dca48b0c3b30190511d6574595e2f412159626260\0";
    const char* input = "123\0";

    char* hash = (char*)malloc(BLOCK_BYTES + 1);
    update(_sha1, input);

    final(_sha1, hash);

    printf("Unit test 1: %s\n", hash);

    if (!string_compare(hash, checksum))
        return 0;

    reset(_sha1);

    free(hash);
    free(_sha1);

    return 1;
}

inline static unsigned char check_sha1_2() {
    struct sha1* _sha1 = newSHA1();
    const char* checksum = "167c15c0c50d0e99bc44a5f5e85408acfc7cf9d3\0";

    const char* input = "[11111111111111111111111111111111111111111111111111111111111111][11111111111111111111111111111111111111111111111111111111111111]\0";

    char* hash = (char*)malloc(BLOCK_BYTES + 1);
    update(_sha1, input);

    final(_sha1, hash);

    printf("Unit test 2: %s\n", hash);

    if (!string_compare(hash, checksum))
        return 0;

    reset(_sha1);

    free(hash);
    free(_sha1);

    return 1;
}

inline static unsigned char check_sha1_3() {
    struct sha1* _sha1 = newSHA1();
    const char* checksum = "fa57570e9e68444a0dfcecad7a2b9cfef8c8af25\0";

    const char* input = "[11111111111111111111111111111111111111111111111111111111111111]0\0";

    char* hash = (char*)malloc(BLOCK_BYTES + 1);
    update(_sha1, input);

    final(_sha1, hash);

    printf("Unit test 3: %s\n", hash);

    if (!string_compare(hash, checksum))
        return 0;

    reset(_sha1);

    free(hash);
    free(_sha1);

    return 1;
}
inline static unsigned char check_sha1_4() {
    struct sha1* _sha1 = newSHA1();
    const char* checksum = "c795c98f35a7877c742d215f0b9a0e5533d499b9\0";

    char* hash = (char*)malloc(BLOCK_BYTES + 1);
    update(_sha1, "2afcac6b9496921c3bd0739922f7896a8427b2c6");
    update(_sha1, "0");

    final(_sha1, hash);

    printf("Unit test 4: %s\n", hash);

    if (!string_compare(hash, checksum))
        return 0;

    reset(_sha1);

    free(hash);
    free(_sha1);

    return 1;
}

inline static unsigned char check_sha1_5() {
    struct sha1* _sha1 = newSHA1();
    const char* checksum = "c795c98f35a7877c742d215f0b9a0e5533d499b9\0";

    char* hash = (char*)malloc(BLOCK_BYTES + 1);

    //char* string = (char*)malloc(1000001);
    //for (size_t t = 0; t < 1000000; t++)
    //    string[t] = 'a';
    //string[1000000] = '\0';

    update(_sha1,
            "[11111111111111111111111111111111111111111111111111111111111111]"
            "[11111111111111111111111111111111111111111111111111111111111111]"
            "[11111111111111111111111111111111111111111111111111111111111111]"
            "[11111111111111111111111111111111111111111111111111111111111111]"
            "[11111111111111111111111111111111111111111111111111111111111111]"
            "[11111111111111111111111111111111111111111111111111111111111111]"
            "[11111111111111111111111111111111111111111111111111111111111111]"
                "[11111111111111111111111111111111111111111111111111111111111111]");

    final(_sha1, hash);

    //printf("%s\n", string);
    //printf("Unit test 5: %s\n", hash);

    if (!string_compare(hash, checksum))
        return 0;

    reset(_sha1);

    free(hash);
    free(_sha1);

    return 1;
}


#endif //COINMINER_UNIT_TEST_H
