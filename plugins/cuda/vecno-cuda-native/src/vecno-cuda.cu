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

#define LT_U256(X, Y) (X.number[3] != Y.number[3] ? X.number[3] < Y.number[3] : X.number[2] != Y.number[2] ? X.number[2] < Y.number[2] : X.number[1] != Y.number[1] ? X.number[1] < Y.number[1] : X.number[0] < Y.number[0])

__constant__ uint8_t hash_header[HASH_HEADER_SIZE];
__constant__ uint256_t target;

__device__ __forceinline__ void blake3_simple_hash(const uint8_t* input, int len, uint8_t* output) {
    uint32_t state[16];
    uint32_t block_words[16];

    #pragma unroll
    for(int i=0; i<16; i++) block_words[i] = 0;

    // Copy input (len is small, 20 or 32)
    // Using memcpy which should be optimized
    memcpy(block_words, input, len);

    // Initialize state with IV and flags
    state[0] = IV[0]; state[1] = IV[1]; state[2] = IV[2]; state[3] = IV[3];
    state[4] = IV[4]; state[5] = IV[5]; state[6] = IV[6]; state[7] = IV[7];
    state[8] = IV[0]; state[9] = IV[1]; state[10] = IV[2]; state[11] = IV[3];
    state[12] = 0; // counter_low
    state[13] = 0; // counter_high
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
    for(int i=0; i<8; i++) {
        store32(output + i*4, state[i] ^ state[i+8]);
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
    for (int i = 0; i < 8; i++) {
        res[i] = h1[i] ^ h2[i];
    }
}

__device__ __inline__ void u32_array_to_u8_array(const uint32_t* result, uint8_t* output) {
    memcpy(output, result, 32);
}

__device__ __inline__ void generate_sbox(const uint8_t* input_bytes, uint8_t* sbox) {
    uint8_t seed[32];
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, input_bytes, 32);
    blake3_hasher_finalize(&hasher, seed, BLAKE3_OUT_LEN);

    for (int i = 0; i < 64; i += 2) {
        sbox[i] = seed[0];
        sbox[i + 1] = seed[1];
        blake3_simple_hash(seed, 32, seed);
    }
}

__device__ __inline__ uint64_t calculate_rounds(const uint8_t* input_bytes, uint64_t timestamp) {
    uint8_t hash[32];
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, input_bytes, 32);
    blake3_hasher_update(&hasher, &timestamp, 8);
    blake3_hasher_finalize(&hasher, hash, BLAKE3_OUT_LEN);

    uint32_t value = ((const uint32_t*)hash)[0];
    return (value % 8 + 16);
}

extern "C" {
    __global__ void mem_hash(const uint64_t nonce_mask, const uint64_t nonce_fixed, const uint64_t nonces_len, uint8_t random_type, void* states, uint64_t* final_nonce, uint64_t timestamp) {
        uint8_t sbox[SBOX_SIZE];
        uint64_t rounds;

        int nonceId = threadIdx.x + blockIdx.x * blockDim.x;
        if (nonceId < nonces_len) {
            if (nonceId == 0) *final_nonce = 0;
            uint64_t nonce;
            switch (random_type) {
                case RANDOM_LEAN:
                    nonce = ((uint64_t *)states)[0] ^ nonceId;
                    break;
                case RANDOM_XOSHIRO:
                default:
                    nonce = xoshiro256_next(((ulonglong4 *)states) + nonceId);
                    break;
            }
            nonce = (nonce & nonce_mask) | nonce_fixed;

            // Step 1: BLAKE3(hash_header || nonce) to get input_hash
            uint8_t input[80];
            memcpy(input, hash_header, HASH_HEADER_SIZE);
            uint256_t input_hash;
            memcpy(input + HASH_HEADER_SIZE, (uint8_t*)&nonce, 8);
            blake3_hasher pow_hasher;
            blake3_hasher_init(&pow_hasher);
            blake3_hasher_update(&pow_hasher, input, 80);
            blake3_hasher_finalize(&pow_hasher, input_hash.hash, BLAKE3_OUT_LEN);

            // Step 2: Compute sbox and rounds
            generate_sbox(input_hash.hash, sbox);
            rounds = calculate_rounds(input_hash.hash, timestamp);

            // Step 3: Initialize result
            uint32_t result[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                result[i] = ((const uint32_t*)input_hash.hash)[i];
            }
            uint8_t hash_bytes[32];
            memcpy(hash_bytes, input_hash.hash, 32);

            // Step 4 & 5: BLAKE3 loops with bit manipulations
            for (uint64_t r = 0; r < rounds; r++) {
                blake3_hasher hasher;
                blake3_hasher_init(&hasher);
                blake3_hasher_update(&hasher, hash_bytes, 32);
                blake3_hasher_finalize(&hasher, hash_bytes, BLAKE3_OUT_LEN);
                bit_manipulations(hash_bytes);
            }
            for (uint64_t r = 0; r < rounds; r++) {
                blake3_hasher hasher;
                blake3_hasher_init(&hasher);
                blake3_hasher_update(&hasher, hash_bytes, 32);
                blake3_hasher_finalize(&hasher, hash_bytes, BLAKE3_OUT_LEN);
                bit_manipulations(hash_bytes);
            }

            // Step 6: Update result
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                result[i] = ((const uint32_t*)hash_bytes)[i];
            }

            // Step 7: Main MemHash loop
            for (uint64_t round = 0; round < rounds; round++) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    uint8_t state_input[20];
                    memcpy(state_input, &result[i], 4);
                    uint64_t round64 = round;
                    memcpy(state_input + 4, &round64, 8);
                    memcpy(state_input + 12, &nonce, 8);
                    uint8_t state_bytes[32];
                    blake3_simple_hash(state_input, 20, state_bytes);

                    uint8_t mixed_bytes[32];
                    byte_mixing(state_bytes, (uint8_t*)result, mixed_bytes);
                    uint32_t v = ((const uint32_t*)mixed_bytes)[0];
                    v ^= result[i];

                    uint32_t branch = (v & 0xFF) % 4;
                    if (branch == 0) {
                        v = v + result[(i + 1) % 8];
                    } else if (branch == 1) {
                        v = v - result[(i + 1) % 8];
                    } else if (branch == 2) {
                        v = (v << (result[(i + 1) % 8] & 0x1F)) | (v >> (32 - (result[(i + 1) % 8] & 0x1F)));
                    } else {
                        v ^= result[(i + 1) % 8];
                    }

                    uint8_t b[4];
                    b[0] = (v >> 0) & 0xFF;
                    b[1] = (v >> 8) & 0xFF;
                    b[2] = (v >> 16) & 0xFF;
                    b[3] = (v >> 24) & 0xFF;
                    uint32_t idx_base = v % 64;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        b[j] = sbox[(idx_base + b[j]) % SBOX_SIZE];
                    }
                    v = ((uint32_t)b[0]) |
                        ((uint32_t)b[1] << 8) |
                        ((uint32_t)b[2] << 16) |
                        ((uint32_t)b[3] << 24);

                    result[i] = v;
                }
            }

            // Step 8: Finalize
            uint8_t* output = (uint8_t*)result;
            bit_manipulations(output);

            // Step 9: Apply VecnoHash
            uint8_t final_output[32];
            blake3_hasher vecno_hasher;
            blake3_hasher_init(&vecno_hasher);
            blake3_hasher_update(&vecno_hasher, output, 32);
            blake3_hasher_finalize(&vecno_hasher, final_output, BLAKE3_OUT_LEN);

            // Step 10: Check against target
            uint256_t final_hash;
            memcpy(final_hash.hash, final_output, 32);

            if (LT_U256(final_hash, target)) {
                atomicCAS((unsigned long long int*)final_nonce, 0, (unsigned long long int)nonce);
            }
        }
    }
}