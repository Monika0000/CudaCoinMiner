//
// Created by Nikita on 05.02.2021.
//

#ifndef COINMINER_STRING_UTIL_H
#define COINMINER_STRING_UTIL_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

inline static void print_char_array(const char* str, unsigned __int32 count) {
    while (count) {
        printf("%c", *str);
        str++;
        count--;
    }

}


//void read_to(char* out, const char* source, )

inline static short fast_strlen(const char* str) {
    unsigned short size = 0;
    while (*str) {
        size++;
        str++;
    }
    return size;
}

__device__ inline static short cuda_fast_strlen(const char* str) {
    unsigned short size = 0;
    while(*str) {
        size++;
        str++;
    }
    return size;
}

inline static char* make_req(const char* name, const char* diff) {
    unsigned char size1 = fast_strlen(name);

    if (diff) {
        unsigned char size2 = fast_strlen(diff);
        unsigned char size_final = 5 + size1 + 1 + size2;

        char *req = (char *) malloc(size_final);
        req[size_final - 1] = '\0';

        req[0] = 'J';
        req[1] = 'O';
        req[2] = 'B';
        req[3] = ',';

        unsigned char pos = 4;

        while (*name) {
            req[pos] = *name;
            name++;
            pos++;
        }

        req[pos] = ',';
        pos++;

        while (*diff) {
            req[pos] = *diff;
            diff++;
            pos++;
        }

        return req;
    }
    else {
        unsigned char size_final = 5 + size1;
        char *req = (char *) malloc(size_final);
        req[size_final - 1] = '\0';

        req[0] = 'J';
        req[1] = 'O';
        req[2] = 'B';
        req[3] = ',';


        unsigned char pos = 4;

        while (*name) {
            req[pos] = *name;
            name++;
            pos++;
        }

        return req;
    }
}

inline static unsigned char string_compare(const char* str1, const char* str2) {
    unsigned short size1 = fast_strlen(str1);
    unsigned short size2 = fast_strlen(str2);
    if (size1 != size2)
        return 0;
    else{
        while (size1){
            if (*str1 != *str2)
                return 0;
            str1++;
            str2++;
            size1--;
        }
        return 1;
    }
}

inline static void read_string(char* out, const char* source, unsigned char count) {
    while(count) {
        *out = *source;
        out++;
        source++;
        count--;
    }
}

__device__ inline static void cuda_read_string(char* out, const char* source, unsigned char count) {
    while (count) {
        *out = *source;
        out++;
        source++;
        count--;
    }
}

inline static void append(char* str, char symbol, unsigned char size, unsigned char count) {
   //realloc(str, size + count);
    /*str += size;
    while (count) {
        *str = symbol;
        str++;
        count--;
    }*/

}

inline static unsigned char char_to_number(char c){
    switch (c) {
        case '0': return 0;
        case '1': return 1;
        case '2': return 2;
        case '3': return 3;
        case '4': return 4;
        case '5': return 5;
        case '6': return 6;
        case '7': return 7;
        case '8': return 8;
        case '9': return 9;
        default: return -1;
    }
}

inline static unsigned int str_to_uint(const char* str) {
    unsigned short size = fast_strlen(str);
    unsigned int result = 0;
    for (unsigned short i = 0; i < size; i++)
        result += char_to_number(str[i]) * (pow(10, size - i - 1));
    return result;
}

inline static unsigned short str_to_ushort(const char* str) {
    unsigned char size = fast_strlen(str);
    unsigned short result = 0;
    for (unsigned char i = 0; i < size; i++)
        result += char_to_number(str[i]) * (pow(10, size - i - 1));
    return result;
}

inline static char* read_to(const char* str, char symbol, unsigned short* id) {
    unsigned short size = fast_strlen(str);
    for (unsigned short i = 0; i < size; i++) {
        if (str[i] == symbol){
            char* result = (char*)malloc((i + 1) * sizeof(char));
            read_string(result, str, i);
            result[i] = '\0';
            (*id) += i + 1;
            return result;
        }
    }
    return NULL;
}

__device__ void lutHexString(unsigned __int32 x, char *s) {
    static const char digits[513] =
            "000102030405060708090a0b0c0d0e0f"
            "101112131415161718191a1b1c1d1e1f"
            "202122232425262728292a2b2c2d2e2f"
            "303132333435363738393a3b3c3d3e3f"
            "404142434445464748494a4b4c4d4e4f"
            "505152535455565758595a5b5c5d5e5f"
            "606162636465666768696a6b6c6d6e6f"
            "707172737475767778797a7b7c7d7e7f"
            "808182838485868788898a8b8c8d8e8f"
            "909192939495969798999a9b9c9d9e9f"
            "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"
            "b0b1b2b3b4b5b6b7b8b9babbbcbdbebf"
            "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf"
            "d0d1d2d3d4d5d6d7d8d9dadbdcdddedf"
            "e0e1e2e3e4e5e6e7e8e9eaebecedeeef"
            "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff";
    int i = 3;
    char *lut = (char *)(digits);
    while (i >= 0)
    {
        int pos = (x & 0xFF) * 2;
        char ch = lut[pos];
        s[i * 2] = ch;

        ch = lut[pos + 1];
        s[i * 2 + 1] = ch;

        x >>= 8;
        i -= 1;
    }
}

void UlongToHexString(unsigned __int64 num, char *s) {
    unsigned __int64 x = num;

    // use bitwise-ANDs and bit-shifts to isolate
    // each nibble into its own byte
    // also need to position relevant nibble/byte into
    // proper location for little-endian copy
    x = ((x & 0xFFFF) << 32) | ((x & 0xFFFF0000) >> 16);
    x = ((x & 0x0000FF000000FF00) >> 8) | (x & 0x000000FF000000FF) << 16;
    x = ((x & 0x00F000F000F000F0) >> 4) | (x & 0x000F000F000F000F) << 8;

    // Now isolated hex digit in each byte
    // Ex: 0x1234FACE => 0x0E0C0A0F04030201

    // Create bitmask of bytes containing alpha hex digits
    // - add 6 to each digit
    // - if the digit is a high alpha hex digit, then the addition
    //   will overflow to the high nibble of the byte
    // - shift the high nibble down to the low nibble and mask
    //   to create the relevant bitmask
    //
    // Using above example:
    // 0x0E0C0A0F04030201 + 0x0606060606060606 = 0x141210150a090807
    // >> 4 == 0x0141210150a09080 & 0x0101010101010101
    // == 0x0101010100000000
    //
    unsigned __int64 mask = ((x + 0x0606060606060606) >> 4) & 0x0101010101010101;

    // convert to ASCII numeral characters
    x |= 0x3030303030303030;

    // if there are high hexadecimal characters, need to adjust
    // for uppercase alpha hex digits, need to add 0x07
    //   to move 0x3A-0x3F to 0x41-0x46 (A-F)
    // for lowercase alpha hex digits, need to add 0x27
    //   to move 0x3A-0x3F to 0x61-0x66 (a-f)
    // it's actually more expensive to test if mask non-null
    //   and then run the following stmt
    //x += ((lowerAlpha) ? 0x27 : 0x07) * mask;
    x += 0x27 * mask;

    //copy string to output buffer
    *(unsigned __int64 *)s = x;
}

byte nibble(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';

    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;

    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;

    return 0;  // Not a valid hexadecimal character
}

void hexToBytes(byte* byteArray, const char* hexString)
{
    bool oddLength = fast_strlen(hexString) & 1;

    byte currentByte = 0;
    byte byteIndex = 0;

    for (byte charIndex = 0; charIndex < strlen(hexString); charIndex++)
    {
        bool oddCharIndex = charIndex & 1;

        if (oddLength)
        {
            // If the length is odd
            if (oddCharIndex)
            {
                // odd characters go in high nibble
                currentByte = nibble(hexString[charIndex]) << 4;
            }
            else
            {
                // Even characters go into low nibble
                currentByte |= nibble(hexString[charIndex]);
                byteArray[byteIndex++] = currentByte;
                currentByte = 0;
            }
        }
        else
        {
            // If the length is even
            if (!oddCharIndex)
            {
                // Odd characters go into the high nibble
                currentByte = nibble(hexString[charIndex]) << 4;
            }
            else
            {
                // Odd characters go into low nibble
                currentByte |= nibble(hexString[charIndex]);
                byteArray[byteIndex++] = currentByte;
                currentByte = 0;
            }
        }
    }
}

#endif //COINMINER_STRING_UTIL_H
