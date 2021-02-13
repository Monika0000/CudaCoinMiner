// Modified version of
// https://github.com/B-Con/crypto-algorithms/blob/master/sha1.c
#include <stdio.h>

__device__ __constant__ unsigned int threadMax;

#define ROTLEFT(a, b) ((a << b) | (a >> (32 - b)))
#define SHA1_BLOCK_SIZE 20
#define TRAIL 24

#define R0(i) \
    t = ROTLEFT(a, 5) + ((b & c) ^ (~b & d)) + e + ctx->k[0] + m[i]; \
    e = d; \
    d = c; \
    c = ROTLEFT(b, 30); \
    b = a; \
    a = t; \

#define R1(i) \
    t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[1] + m[i]; \
    e = d; \
    d = c; \
    c = ROTLEFT(b, 30); \
    b = a; \
    a = t; \

#define R2(i) \
    t = ROTLEFT(a, 5) + ((b & c) ^ (b & d) ^ (c & d))  + e + ctx->k[2] + m[i]; \
    e = d; \
    d = c; \
    c = ROTLEFT(b, 30); \
    b = a; \
    a = t; \

#define R3(i) \
    t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[3] + m[i]; \
    e = d; \
    d = c; \
    c = ROTLEFT(b, 30); \
    b = a; \
    a = t; \

typedef unsigned char BYTE;             // 8-bit byte
// typedef unsigned int WORD; // 32-bit word, change to "long" for 16-bit machines

typedef struct {
    BYTE data[64];
    unsigned int datalen;
    unsigned long long bitlen;
    unsigned int state[5];
    unsigned int k[4];
} SHA1_CTX;


__device__ void sha1_transform(SHA1_CTX *ctx, const BYTE data[])
{
    unsigned int a, b, c, d, e, i, j, t, m[80];

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) + (data[j + 1] << 16) + (data[j + 2] << 8) + (data[j + 3]);
    for ( ; i < 80; ++i) {
        m[i] = (m[i - 3] ^ m[i - 8] ^ m[i - 14] ^ m[i - 16]);
        m[i] = (m[i] << 1) | (m[i] >> 31);
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];

    R0(0);
    R0(1);
    R0(2);
    R0(3);
    R0(4);
    R0(5);
    R0(6);
    R0(7);
    R0(8);
    R0(9);
    R0(10);
    R0(11);
    R0(12);
    R0(13);
    R0(14);
    R0(15);
    R0(16);
    R0(17);
    R0(18);
    R0(19);

    R1(20);
    R1(21);
    R1(22);
    R1(23);
    R1(24);
    R1(25);
    R1(26);
    R1(27);
    R1(28);
    R1(29);
    R1(30);
    R1(31);
    R1(32);
    R1(33);
    R1(34);
    R1(35);
    R1(36);
    R1(37);
    R1(38);
    R1(39);

    R2(40);
    R2(41);
    R2(42);
    R2(43);
    R2(44);
    R2(45);
    R2(46);
    R2(47);
    R2(48);
    R2(49);
    R2(50);
    R2(51);
    R2(52);
    R2(53);
    R2(54);
    R2(55);
    R2(56);
    R2(57);
    R2(58);
    R2(59);

    R3(60);
    R3(61);
    R3(62);
    R3(63);
    R3(64);
    R3(65);
    R3(66);
    R3(67);
    R3(68);
    R3(69);
    R3(70);
    R3(71);
    R3(72);
    R3(73);
    R3(74);
    R3(75);
    R3(76);
    R3(77);
    R3(78);
    R3(79);

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
}

__device__ void sha1_init(SHA1_CTX *ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;
    ctx->state[4] = 0xc3d2e1f0;
    ctx->k[0] = 0x5a827999;
    ctx->k[1] = 0x6ed9eba1;
    ctx->k[2] = 0x8f1bbcdc;
    ctx->k[3] = 0xca62c1d6;
}

__device__ void sha1_update(SHA1_CTX *ctx, const BYTE data[], size_t len)
{
    size_t i;

    for (i = 0; i < len; ++i) 
    {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) 
        {
            sha1_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void sha1_final(SHA1_CTX *ctx, BYTE hash[])
{
    unsigned int i;

    i = ctx->datalen;

    // Pad whatever data is left in the buffer.
    if (ctx->datalen < 56) 
    {
        ctx->data[i++] = 0x80;
        while (i < 56)
            ctx->data[i++] = 0x00;
    }
    else 
    {
        ctx->data[i++] = 0x80;
        while (i < 64)
            ctx->data[i++] = 0x00;
        sha1_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }

    // Append to the padding the total message's length in bits and transform.
    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;
    ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16;
    ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32;
    ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48;
    ctx->data[56] = ctx->bitlen >> 56;
    sha1_transform(ctx, ctx->data);

    // Since this implementation uses little endian byte ordering and MD uses big endian,
    // reverse all the bytes when copying the final state to the output hash.
    for (i = 0; i < 4; ++i) 
    {
        hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
    }
}

__device__ bool check_trail(BYTE *ptr)
{

    return (ptr[19-0] == 0xff) && (ptr[19-1] == 0xff) && (ptr[19-2] == 0xff);
}

/*__global__ void run()
{
    int nnum =  blockIdx.x * 1024 + threadIdx.x;
    int num;

    SHA1_CTX ctx;
    BYTE text1[] = {
    				__PREFIX__
                   };

    BYTE buf[SHA1_BLOCK_SIZE];

    BYTE alphabet[] = {
    				   __ALPHABET__
                      };

    int k, r;

    do
    {
        r = 0;
        num = nnum;

        for (k = 0; k < __LENGTH__; k++)
        {
            r = num & __MASK__;
            text1[__PREFIXLEN__ + k] = alphabet[r];
            num = (num >> __LOG2ALPHABET__);
        }

        sha1_init(&ctx);
        sha1_update(&ctx, text1, __PREFIXLEN__+__LENGTH__);
        sha1_final(&ctx, buf);

        nnum += 256 * 1024;

    } while ((__INVMASKFUNC__) && nnum < __MAX_ITERATIONS__);


    if (__MASKFUNC__)
    {
        printf("%s\n", text1);
        printf("%.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x %.2x\n", 
        	buf[0], buf[1], buf[2], buf[3],
        	buf[4], buf[5], buf[6], buf[7],
        	buf[8], buf[9], buf[10], buf[11],
        	buf[12], buf[13], buf[14], buf[15],
        	buf[16], buf[17], buf[18], buf[19]
        	);
    }

}*/
