#include <stdint.h>
#include <string.h>
#include "xoshiro256starstar.c"
#include "blake3_compact.h"

typedef union _uint256_t {
    uint64_t number[4];
    uint8_t hash[32];
} uint256_t;

#define BLOCKDIM 128
#define SBOX_SIZE 64
#define HASH_HEADER_SIZE 72

#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

#define LT_U256(X, Y) (X.number[3] != Y.number[3] ? X.number[3] < Y.number[3] : \
                       X.number[2] != Y.number[2] ? X.number[2] < Y.number[2] : \
                       X.number[1] != Y.number[1] ? X.number[1] < Y.number[1] : \
                       X.number[0] < Y.number[0])

__constant__ uint8_t hash_header[HASH_HEADER_SIZE];
__constant__ uint256_t target;

__device__ __forceinline__ void blake3_hash_80bytes(const uint8_t* input, uint8_t* output) {
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = IV[i];

    uint32_t state[16];
    uint32_t block_words[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        block_words[i] = load32(input + i * 4);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = cv[i];
    state[8]  = IV[0]; state[9]  = IV[1]; state[10] = IV[2]; state[11] = IV[3];
    state[12] = 0; state[13] = 0; state[14] = 64; state[15] = CHUNK_START;

    #pragma unroll
    for (int r = 0; r < 7; r++) round_fn(state, block_words, r);

    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = state[i] ^ state[i + 8];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        block_words[i] = load32(input + 64 + i * 4);
    }
    #pragma unroll
    for (int i = 4; i < 16; i++) block_words[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = cv[i];
    state[8]  = IV[0]; state[9]  = IV[1]; state[10] = IV[2]; state[11] = IV[3];
    state[12] = 0; state[13] = 0; state[14] = 16; state[15] = CHUNK_END | ROOT;

    #pragma unroll
    for (int r = 0; r < 7; r++) round_fn(state, block_words, r);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store32(output + i * 4, state[i] ^ state[i + 8]);
    }
}

__device__ __forceinline__ void blake3_simple_hash(const uint8_t* input, int len, uint8_t* output) {
    uint32_t state[16];
    uint32_t block_words[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) block_words[i] = 0;
    memcpy(block_words, input, len);

    state[0] = IV[0]; state[1] = IV[1]; state[2] = IV[2]; state[3] = IV[3];
    state[4] = IV[4]; state[5] = IV[5]; state[6] = IV[6]; state[7] = IV[7];
    state[8] = IV[0]; state[9] = IV[1]; state[10] = IV[2]; state[11] = IV[3];
    state[12] = 0; state[13] = 0;
    state[14] = (uint32_t)len;
    state[15] = CHUNK_START | CHUNK_END | ROOT;

    round_fn(state, block_words, 0);
    round_fn(state, block_words, 1);
    round_fn(state, block_words, 2);
    round_fn(state, block_words, 3);
    round_fn(state, block_words, 4);
    round_fn(state, block_words, 5);
    round_fn(state, block_words, 6);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store32(output + i * 4, state[i] ^ state[i + 8]);
    }
}

__device__ __inline__ void bit_manipulations(uint8_t* data) {
    uint32_t* d = (uint32_t*)data;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t val = d[i];
        uint32_t xor1 = (val >> 8) & 0xFF;
        uint32_t xor2 = ((val >> 24) & 0xFF) << 16;
        d[i] = val ^ xor1 ^ xor2;
    }
}

__device__ __inline__ void byte_mixing(const uint8_t* b3_hash1, const uint8_t* b3_hash2, uint8_t* result) {
    const uint32_t* h1 = (const uint32_t*)b3_hash1;
    const uint32_t* h2 = (const uint32_t*)b3_hash2;
    uint32_t* res = (uint32_t*)result;
    #pragma unroll
    for (int i = 0; i < 8; i++) res[i] = h1[i] ^ h2[i];
}

__device__ __inline__ void generate_sbox(const uint8_t* input_bytes, uint8_t* sbox) {
    uint8_t seed[32];
    blake3_simple_hash(input_bytes, 32, seed);

    for (int i = 0; i < 64; i += 2) {
        sbox[i]     = seed[0];
        sbox[i + 1] = seed[1];
        blake3_simple_hash(seed, 32, seed);
    }
}

__device__ __inline__ uint64_t calculate_rounds(const uint8_t* input_bytes, uint64_t timestamp) {
    uint8_t buffer[40];
    memcpy(buffer, input_bytes, 32);
    memcpy(buffer + 32, &timestamp, sizeof(timestamp));
    uint8_t hash[32];
    blake3_simple_hash(buffer, 40, hash);
    uint32_t value = ((const uint32_t*)hash)[0];
    return (value % 8 + 16);
}

extern "C" {
    __global__ void mem_hash(const uint64_t nonce_mask, const uint64_t nonce_fixed,
                             const uint64_t nonces_len, uint8_t random_type,
                             void* states, uint64_t* final_nonce, uint64_t timestamp) {

        uint8_t sbox[SBOX_SIZE];

        int nonceId = threadIdx.x + blockIdx.x * blockDim.x;
        if (nonceId >= nonces_len) return;

        if (nonceId == 0) *final_nonce = 0;

        uint64_t nonce;
        switch (random_type) {
            case RANDOM_LEAN:
                nonce = ((uint64_t*)states)[0] ^ nonceId;
                break;
            case RANDOM_XOSHIRO:
            default:
                nonce = xoshiro256_next(((ulonglong4*)states) + nonceId);
                break;
        }
        nonce = (nonce & nonce_mask) | nonce_fixed;

        uint8_t input[80];
        memcpy(input, hash_header, HASH_HEADER_SIZE);
        memcpy(input + HASH_HEADER_SIZE, &nonce, 8);

        uint8_t input_hash_bytes[32];
        blake3_hash_80bytes(input, input_hash_bytes);

        uint256_t input_hash;
        memcpy(input_hash.hash, input_hash_bytes, 32);

        generate_sbox(input_hash.hash, sbox);
        uint64_t rounds = calculate_rounds(input_hash.hash, timestamp);

        uint32_t result[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            result[i] = ((const uint32_t*)input_hash.hash)[i];
        }

        uint8_t hash_bytes[32];
        memcpy(hash_bytes, input_hash.hash, 32);

        for (uint64_t r = 0; r < 2 * rounds; r++) {
            blake3_simple_hash(hash_bytes, 32, hash_bytes);
            bit_manipulations(hash_bytes);
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            result[i] = ((const uint32_t*)hash_bytes)[i];
        }

        for (uint64_t round = 0; round < rounds; round++) {
            uint8_t state_input[20];
            uint8_t state_bytes[32];
            uint8_t mixed_bytes[32];
            uint32_t v, branch, next, shift;
            uint8_t b[4];
            uint32_t idx_base;

            // i = 0
            *(uint32_t*)state_input         = result[0];
            *(uint64_t*)(state_input + 4)   = round;
            *(uint64_t*)(state_input + 12)  = nonce;
            blake3_simple_hash(state_input, 20, state_bytes);
            byte_mixing(state_bytes, (uint8_t*)result, mixed_bytes);
            v = ((const uint32_t*)mixed_bytes)[0] ^ result[0];
            branch = v & 3;
            next = result[1];
            if (branch == 0) v += next;
            else if (branch == 1) v -= next;
            else if (branch == 2) { shift = next & 0x1F; v = (v << shift) | (v >> (32 - shift)); }
            else v ^= next;
            b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24;
            idx_base = v & 63;
            b[0] = sbox[(idx_base + b[0]) & 63];
            b[1] = sbox[(idx_base + b[1]) & 63];
            b[2] = sbox[(idx_base + b[2]) & 63];
            b[3] = sbox[(idx_base + b[3]) & 63];
            result[0] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);

            // i = 1
            *(uint32_t*)state_input         = result[1];
            *(uint64_t*)(state_input + 4)   = round;
            *(uint64_t*)(state_input + 12)  = nonce;
            blake3_simple_hash(state_input, 20, state_bytes);
            byte_mixing(state_bytes, (uint8_t*)result, mixed_bytes);
            v = ((const uint32_t*)mixed_bytes)[0] ^ result[1];
            branch = v & 3;
            next = result[2];
            if (branch == 0) v += next;
            else if (branch == 1) v -= next;
            else if (branch == 2) { shift = next & 0x1F; v = (v << shift) | (v >> (32 - shift)); }
            else v ^= next;
            b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24;
            idx_base = v & 63;
            b[0] = sbox[(idx_base + b[0]) & 63];
            b[1] = sbox[(idx_base + b[1]) & 63];
            b[2] = sbox[(idx_base + b[2]) & 63];
            b[3] = sbox[(idx_base + b[3]) & 63];
            result[1] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);

            // i = 2
            *(uint32_t*)state_input         = result[2];
            *(uint64_t*)(state_input + 4)   = round;
            *(uint64_t*)(state_input + 12)  = nonce;
            blake3_simple_hash(state_input, 20, state_bytes);
            byte_mixing(state_bytes, (uint8_t*)result, mixed_bytes);
            v = ((const uint32_t*)mixed_bytes)[0] ^ result[2];
            branch = v & 3;
            next = result[3];
            if (branch == 0) v += next;
            else if (branch == 1) v -= next;
            else if (branch == 2) { shift = next & 0x1F; v = (v << shift) | (v >> (32 - shift)); }
            else v ^= next;
            b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24;
            idx_base = v & 63;
            b[0] = sbox[(idx_base + b[0]) & 63];
            b[1] = sbox[(idx_base + b[1]) & 63];
            b[2] = sbox[(idx_base + b[2]) & 63];
            b[3] = sbox[(idx_base + b[3]) & 63];
            result[2] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);

            // i = 3
            *(uint32_t*)state_input         = result[3];
            *(uint64_t*)(state_input + 4)   = round;
            *(uint64_t*)(state_input + 12)  = nonce;
            blake3_simple_hash(state_input, 20, state_bytes);
            byte_mixing(state_bytes, (uint8_t*)result, mixed_bytes);
            v = ((const uint32_t*)mixed_bytes)[0] ^ result[3];
            branch = v & 3;
            next = result[4];
            if (branch == 0) v += next;
            else if (branch == 1) v -= next;
            else if (branch == 2) { shift = next & 0x1F; v = (v << shift) | (v >> (32 - shift)); }
            else v ^= next;
            b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24;
            idx_base = v & 63;
            b[0] = sbox[(idx_base + b[0]) & 63];
            b[1] = sbox[(idx_base + b[1]) & 63];
            b[2] = sbox[(idx_base + b[2]) & 63];
            b[3] = sbox[(idx_base + b[3]) & 63];
            result[3] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);

            // i = 4
            *(uint32_t*)state_input         = result[4];
            *(uint64_t*)(state_input + 4)   = round;
            *(uint64_t*)(state_input + 12)  = nonce;
            blake3_simple_hash(state_input, 20, state_bytes);
            byte_mixing(state_bytes, (uint8_t*)result, mixed_bytes);
            v = ((const uint32_t*)mixed_bytes)[0] ^ result[4];
            branch = v & 3;
            next = result[5];
            if (branch == 0) v += next;
            else if (branch == 1) v -= next;
            else if (branch == 2) { shift = next & 0x1F; v = (v << shift) | (v >> (32 - shift)); }
            else v ^= next;
            b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24;
            idx_base = v & 63;
            b[0] = sbox[(idx_base + b[0]) & 63];
            b[1] = sbox[(idx_base + b[1]) & 63];
            b[2] = sbox[(idx_base + b[2]) & 63];
            b[3] = sbox[(idx_base + b[3]) & 63];
            result[4] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);

            // i = 5
            *(uint32_t*)state_input         = result[5];
            *(uint64_t*)(state_input + 4)   = round;
            *(uint64_t*)(state_input + 12)  = nonce;
            blake3_simple_hash(state_input, 20, state_bytes);
            byte_mixing(state_bytes, (uint8_t*)result, mixed_bytes);
            v = ((const uint32_t*)mixed_bytes)[0] ^ result[5];
            branch = v & 3;
            next = result[6];
            if (branch == 0) v += next;
            else if (branch == 1) v -= next;
            else if (branch == 2) { shift = next & 0x1F; v = (v << shift) | (v >> (32 - shift)); }
            else v ^= next;
            b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24;
            idx_base = v & 63;
            b[0] = sbox[(idx_base + b[0]) & 63];
            b[1] = sbox[(idx_base + b[1]) & 63];
            b[2] = sbox[(idx_base + b[2]) & 63];
            b[3] = sbox[(idx_base + b[3]) & 63];
            result[5] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);

            // i = 6
            *(uint32_t*)state_input         = result[6];
            *(uint64_t*)(state_input + 4)   = round;
            *(uint64_t*)(state_input + 12)  = nonce;
            blake3_simple_hash(state_input, 20, state_bytes);
            byte_mixing(state_bytes, (uint8_t*)result, mixed_bytes);
            v = ((const uint32_t*)mixed_bytes)[0] ^ result[6];
            branch = v & 3;
            next = result[7];
            if (branch == 0) v += next;
            else if (branch == 1) v -= next;
            else if (branch == 2) { shift = next & 0x1F; v = (v << shift) | (v >> (32 - shift)); }
            else v ^= next;
            b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24;
            idx_base = v & 63;
            b[0] = sbox[(idx_base + b[0]) & 63];
            b[1] = sbox[(idx_base + b[1]) & 63];
            b[2] = sbox[(idx_base + b[2]) & 63];
            b[3] = sbox[(idx_base + b[3]) & 63];
            result[6] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);

            *(uint32_t*)state_input         = result[7];
            *(uint64_t*)(state_input + 4)   = round;
            *(uint64_t*)(state_input + 12)  = nonce;
            blake3_simple_hash(state_input, 20, state_bytes);
            byte_mixing(state_bytes, (uint8_t*)result, mixed_bytes);
            v = ((const uint32_t*)mixed_bytes)[0] ^ result[7];
            branch = v & 3;
            next = result[0];
            if (branch == 0) v += next;
            else if (branch == 1) v -= next;
            else if (branch == 2) { shift = next & 0x1F; v = (v << shift) | (v >> (32 - shift)); }
            else v ^= next;
            b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24;
            idx_base = v & 63;
            b[0] = sbox[(idx_base + b[0]) & 63];
            b[1] = sbox[(idx_base + b[1]) & 63];
            b[2] = sbox[(idx_base + b[2]) & 63];
            b[3] = sbox[(idx_base + b[3]) & 63];
            result[7] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
        }

        uint8_t* output = (uint8_t*)result;
        bit_manipulations(output);

        uint8_t final_output[32];
        blake3_simple_hash(output, 32, final_output);

        uint256_t final_hash;
        memcpy(final_hash.hash, final_output, 32);

        if (LT_U256(final_hash, target)) {
            atomicCAS((unsigned long long int*)final_nonce, 0ULL, (unsigned long long int)nonce);
        }
    }
}