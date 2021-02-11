//
// Created by Nikita on 05.02.2021.
//

#ifndef COINMINER_SHA1_UTIL_H
#define COINMINER_SHA1_UTIL_H

#define BLOCK_INTS 16
#define BLOCK_BYTES 4 * BLOCK_INTS

inline static unsigned __int32 rol(const unsigned __int32 value, const size_t bits) {
    return (value << bits) | (value >> (32 - bits));
}

inline static unsigned __int32 blk(const unsigned __int32 block[BLOCK_INTS], const size_t i) {
    return rol(block[(i+13)&15] ^ block[(i+8)&15] ^ block[(i+2)&15] ^ block[i], 1);
}

inline static void R0(const unsigned __int32 block[BLOCK_INTS], const unsigned __int32 v, unsigned __int32 *w, const unsigned __int32 x, const unsigned __int32 y, unsigned __int32 *z, const size_t i) {
    *z += (((*w) & (x^y))^y) + block[i] + 0x5a827999 + rol(v, 5);
    *w = rol(*w, 30);
}

inline static void R1(unsigned __int32 block[BLOCK_INTS], const unsigned __int32 v, unsigned __int32 *w, const unsigned __int32 x, const unsigned __int32 y, unsigned __int32 *z, const size_t i) {
    block[i] = blk(block, i);
    *z += (((*w) & (x^y))^y) + block[i] + 0x5a827999 + rol(v, 5);
    *w = rol(*w, 30);
}

inline static void R2(unsigned __int32 block[BLOCK_INTS], const unsigned __int32 v, unsigned __int32* w, const unsigned __int32 x, const unsigned __int32 y, unsigned __int32 *z, const size_t i) {
    block[i] = blk(block, i);
    *z += ((*w) ^x^y) + block[i] + 0x6ed9eba1 + rol(v, 5);
    *w = rol(*w, 30);
}

inline static void R3(unsigned __int32 block[BLOCK_INTS], const unsigned __int32 v, unsigned __int32 *w, const unsigned __int32 x, const unsigned __int32 y, unsigned __int32 *z, const size_t i) {
    block[i] = blk(block, i);
    *z += ((((*w)|x)&y)|((*w)&x)) + block[i] + 0x8f1bbcdc + rol(v, 5);
    *w = rol(*w, 30);
}

inline static void R4(unsigned __int32 block[BLOCK_INTS], const unsigned __int32 v, unsigned __int32 *w, const unsigned __int32 x, const unsigned __int32 y, unsigned __int32 *z, const size_t i) {
    block[i] = blk(block, i);
    *z += ((*w)^x^y) + block[i] + 0xca62c1d6 + rol(v, 5);
    *w = rol(*w, 30);
}

#endif //COINMINER_SHA1_UTIL_H
