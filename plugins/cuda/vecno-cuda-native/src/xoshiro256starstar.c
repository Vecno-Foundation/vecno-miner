/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include <stdint.h>

__device__ static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

__device__ inline uint64_t xoshiro256_next(ulonglong4* s) {
    const uint64_t result = rotl(s->y * 5, 7) * 9;
    const uint64_t t = s->y << 17;
    s->z ^= s->x;
    s->w ^= s->y;
    s->y ^= s->z;
    s->x ^= s->w;
    s->z ^= t;
    s->w = rotl(s->w, 45);
    return result;
}

__device__ void xoshiro256_jump(ulonglong4* s) {
    static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for (int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s->x;
                s1 ^= s->y;
                s2 ^= s->z;
                s3 ^= s->w;
            }
            xoshiro256_next(s);
        }
    s->x = s0;
    s->y = s1;
    s->z = s2;
    s->w = s3;
}

__device__ void xoshiro256_long_jump(ulonglong4* s) {
    static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for (int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
        for (int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s->x;
                s1 ^= s->y;
                s2 ^= s->z;
                s3 ^= s->w;
            }
            xoshiro256_next(s);
        }
    s->x = s0;
    s->y = s1;
    s->z = s2;
    s->w = s3;
}

// Added: Initialize Xoshiro256** state with a seed
__device__ void xoshiro256starstar_init(ulonglong4* s, uint64_t seed) {
    // Use splitmix64 to generate four 64-bit values for the state
    uint64_t z = seed + 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    s->x = z ^ (z >> 31);
    z = s->x + 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    s->y = z ^ (z >> 31);
    z = s->y + 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    s->z = z ^ (z >> 31);
    z = s->z + 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    s->w = z ^ (z >> 31);
}