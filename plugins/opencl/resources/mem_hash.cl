// Catering for different OpenCL flavors
#ifdef OPENCL_PLATFORM_AMD
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#endif

#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#define STATIC
#else
#define STATIC static
#endif

#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

/* TYPES */
typedef uchar uint8_t;
typedef char int8_t;
typedef ushort uint16_t;
typedef short int16_t;
typedef uint uint32_t;
typedef int int32_t;
typedef ulong uint64_t;
typedef long int64_t;

typedef union _uint256_t {
    uint64_t number[4];
    uint8_t hash[32];
} uint256_t;

/* BLAKE3 IMPLEMENTATION */
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024

constant static const uint32_t IV[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

constant static const uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

enum blake3_flags {
    CHUNK_START = 1 << 0,
    CHUNK_END = 1 << 1,
    PARENT = 1 << 2,
    ROOT = 1 << 3,
};

typedef struct {
    uint32_t cv[8];
    uint64_t chunk_counter;
    uint8_t buf[BLAKE3_BLOCK_LEN];
    uint8_t buf_len;
    uint8_t blocks_compressed;
    uint8_t flags;
} blake3_chunk_state;

typedef struct {
    uint32_t key[8];
    blake3_chunk_state chunk;
} blake3_hasher;

STATIC inline uint32_t load32_private(__private const uint8_t* src) {
    return ((uint32_t)(src[0]) << 0) | ((uint32_t)(src[1]) << 8) |
           ((uint32_t)(src[2]) << 16) | ((uint32_t)(src[3]) << 24);
}

STATIC inline void store32(__private uint8_t* dst, uint32_t w) {
    dst[0] = (uint8_t)(w >> 0);
    dst[1] = (uint8_t)(w >> 8);
    dst[2] = (uint8_t)(w >> 16);
    dst[3] = (uint8_t)(w >> 24);
}

STATIC inline void store_cv_words(__private uint8_t bytes_out[32], __private const uint32_t cv_words[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store32(&bytes_out[i * 4], cv_words[i]);
    }
}

STATIC inline uint32_t rotr32(uint32_t w, uint32_t c) {
    return (w >> c) | (w << (32 - c));
}

STATIC inline uint32_t counter_low(uint64_t counter) {
    return (uint32_t)counter;
}

STATIC inline uint32_t counter_high(uint64_t counter) {
    return (uint32_t)(counter >> 32);
}

STATIC inline void chunk_state_init(__private blake3_chunk_state* self, __constant const uint32_t key[8], uint8_t flags) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        self->cv[i] = key[i];
    }
    self->chunk_counter = 0;
    #pragma unroll
    for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) {
        self->buf[i] = 0;
    }
    self->buf_len = 0;
    self->blocks_compressed = 0;
    self->flags = flags;
}

STATIC inline void chunk_state_reset(__private blake3_chunk_state* self, __private const uint32_t key[8], uint64_t chunk_counter) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        self->cv[i] = key[i];
    }
    self->chunk_counter = chunk_counter;
    self->blocks_compressed = 0;
    #pragma unroll
    for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) {
        self->buf[i] = 0;
    }
    self->buf_len = 0;
}

STATIC inline size_t chunk_state_len(__private const blake3_chunk_state* self) {
    return (BLAKE3_BLOCK_LEN * (size_t)self->blocks_compressed) + ((size_t)self->buf_len);
}

STATIC inline size_t chunk_state_fill_buf(__private blake3_chunk_state* self, __private const uint8_t* input, size_t input_len) {
    size_t take = BLAKE3_BLOCK_LEN - ((size_t)self->buf_len);
    if (take > input_len) {
        take = input_len;
    }
    #pragma unroll
    for (size_t i = 0; i < take; i++) {
        self->buf[self->buf_len + i] = input[i];
    }
    self->buf_len += (uint8_t)take;
    return take;
}

STATIC inline uint8_t chunk_state_maybe_start_flag(__private const blake3_chunk_state* self) {
    return (self->blocks_compressed == 0) ? CHUNK_START : 0;
}

STATIC inline void g(__private uint32_t* state, size_t a, size_t b, size_t c, size_t d, uint32_t x, uint32_t y) {
    state[a] = state[a] + state[b] + x;
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + y;
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7);
}

STATIC inline void round_fn(__private uint32_t state[16], __private const uint32_t* msg, size_t round) {
    __constant const uint8_t* schedule = MSG_SCHEDULE[round];
    g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
    g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
    g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
    g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);
    g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
    g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
    g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
    g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

STATIC inline void compress_pre(__private uint32_t state[16], __private const uint32_t cv[8],
    __private const uint8_t block[BLAKE3_BLOCK_LEN], uint8_t block_len, uint64_t counter, uint8_t flags) {
    __private uint32_t block_words[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        block_words[i] = load32_private(block + 4 * i);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = cv[i];
        state[i + 8] = IV[i];
    }
    state[12] = counter_low(counter);
    state[13] = counter_high(counter);
    state[14] = (uint32_t)block_len;
    state[15] = (uint32_t)flags;

    #pragma unroll
    for (size_t round = 0; round < 7; round++) {
        round_fn(state, block_words, round);
    }
}

STATIC inline void blake3_compress_in_place(__private uint32_t cv[8], __private const uint8_t block[BLAKE3_BLOCK_LEN],
    uint8_t block_len, uint64_t counter, uint8_t flags) {
    __private uint32_t state[16];
    compress_pre(state, cv, block, block_len, counter, flags);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        cv[i] = state[i] ^ state[i + 8];
    }
}

STATIC inline void blake3_hasher_init(__private blake3_hasher* self, __constant const uint32_t key[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        self->key[i] = key[i];
    }
    chunk_state_init(&self->chunk, key, 0);
}

STATIC inline void blake3_hasher_update(__private blake3_hasher* self, __private const uint8_t* input, size_t input_len) {
    if (input_len == 0) return;

    if (chunk_state_len(&self->chunk) > 0) {
        size_t take = BLAKE3_CHUNK_LEN - chunk_state_len(&self->chunk);
        if (take > input_len) take = input_len;
        chunk_state_fill_buf(&self->chunk, input, take);
        input += take;
        input_len -= take;

        if (input_len > 0) {
            blake3_compress_in_place(self->chunk.cv, self->chunk.buf, BLAKE3_BLOCK_LEN,
                self->chunk.chunk_counter, self->chunk.flags | chunk_state_maybe_start_flag(&self->chunk));
            self->chunk.blocks_compressed += 1;
            self->chunk.buf_len = 0;
            #pragma unroll
            for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) self->chunk.buf[i] = 0;
            chunk_state_reset(&self->chunk, self->key, self->chunk.chunk_counter + 1);
        }
    }

    while (input_len > BLAKE3_BLOCK_LEN) {
        blake3_compress_in_place(self->chunk.cv, input, BLAKE3_BLOCK_LEN,
            self->chunk.chunk_counter, self->chunk.flags | chunk_state_maybe_start_flag(&self->chunk));
        self->chunk.blocks_compressed += 1;
        input += BLAKE3_BLOCK_LEN;
        input_len -= BLAKE3_BLOCK_LEN;
    }

    if (input_len > 0) {
        chunk_state_fill_buf(&self->chunk, input, input_len);
    }
}

STATIC inline void blake3_hasher_finalize(__private const blake3_hasher* self, __private uint8_t* out, size_t out_len) {
    if (out_len == 0) return;

    __private uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = self->chunk.cv[i];
    __private uint8_t block[BLAKE3_BLOCK_LEN];
    #pragma unroll
    for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) block[i] = self->chunk.buf[i];
    uint8_t block_len = self->chunk.buf_len;
    uint8_t flags = self->chunk.flags | chunk_state_maybe_start_flag(&self->chunk) | CHUNK_END;

    if (block_len > 0) {
        #pragma unroll
        for (int i = block_len; i < BLAKE3_BLOCK_LEN; i++) block[i] = 0;
    } else {
        #pragma unroll
        for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) block[i] = 0;
        block_len = 0;
    }

    blake3_compress_in_place(cv, block, block_len, self->chunk.chunk_counter, flags | ROOT);
    store_cv_words(out, cv);
}

/* XOSHIRO256** */
STATIC inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

STATIC inline uint64_t xoshiro256_next(__private ulong4* s) {
    const uint64_t result = rotl(s->s0 * 5, 7) * 9;
    const uint64_t t = s->s1 << 17;
    s->s2 ^= s->s0;
    s->s3 ^= s->s1;
    s->s1 ^= s->s2;
    s->s0 ^= s->s3;
    s->s2 ^= t;
    s->s3 = rotl(s->s3, 45);
    return result;
}

/* MEMORY HASH IMPLEMENTATION */
#define BLOCKDIM 128
#define SBOX_SIZE 64
#define HASH_HEADER_SIZE 72
#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

#define LT_U256(X, Y) (X.number[3] != Y.number[3] ? X.number[3] < Y.number[3] : X.number[2] != Y.number[2] ? X.number[2] < Y.number[2] : X.number[1] != Y.number[1] ? X.number[1] < Y.number[1] : X.number[0] < Y.number[0])

STATIC inline void bit_manipulations(__private uint8_t* data) {
    __private uint32_t* d = (__private uint32_t*)data;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t val = d[i];
        uint32_t xor1 = (val >> 8) & 0xFF;
        uint32_t xor2 = ((val >> 24) & 0xFF) << 16;
        d[i] = val ^ xor1 ^ xor2;
    }
}

STATIC inline void byte_mixing(__private const uint8_t* b3_hash1, __private const uint8_t* b3_hash2, __private uint8_t* result) {
    __private const uint32_t* h1 = (__private const uint32_t*)b3_hash1;
    __private const uint32_t* h2 = (__private const uint32_t*)b3_hash2;
    __private uint32_t* res = (__private uint32_t*)result;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        res[i] = h1[i] ^ h2[i];
    }
}

STATIC inline void u32_array_to_u8_array(__private const uint32_t* result, __private uint8_t* output) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        output[i * 4 + 0] = (result[i] >> 0) & 0xFF;
        output[i * 4 + 1] = (result[i] >> 8) & 0xFF;
        output[i * 4 + 2] = (result[i] >> 16) & 0xFF;
        output[i * 4 + 3] = (result[i] >> 24) & 0xFF;
    }
}

STATIC inline void generate_sbox(__private const uint8_t* input_bytes, __private uint8_t* sbox) {
    __private uint8_t seed[32];
    __private blake3_hasher hasher;
    blake3_hasher_init(&hasher, IV);
    blake3_hasher_update(&hasher, input_bytes, 32);
    blake3_hasher_finalize(&hasher, seed, BLAKE3_OUT_LEN);

    #pragma unroll
    for (int i = 0; i < 64; i += 2) {
        sbox[i] = seed[0];
        sbox[i + 1] = seed[1];
        blake3_hasher_init(&hasher, IV);
        blake3_hasher_update(&hasher, seed, 32);
        blake3_hasher_finalize(&hasher, seed, BLAKE3_OUT_LEN);
    }
}

STATIC inline uint64_t calculate_rounds(__private const uint8_t* input_bytes, uint64_t timestamp) {
    __private uint8_t hash[32];
    __private blake3_hasher hasher;
    blake3_hasher_init(&hasher, IV);
    blake3_hasher_update(&hasher, input_bytes, 32);
    blake3_hasher_update(&hasher, (__private const uint8_t*)&timestamp, 8);
    blake3_hasher_finalize(&hasher, hash, BLAKE3_OUT_LEN);

    uint32_t value = ((const uint32_t*)hash)[0];
    return (value % 8 + 16);
}

kernel void mem_hash(
    const uint64_t nonce_mask,
    const uint64_t nonce_fixed,
    const uint64_t nonces_len,
    const uint8_t random_type,
    __constant const uint8_t* hash_header,
    __constant const uint256_t* target,
    __global ulong4* states,
    __global volatile uint64_t* final_nonce,
    const uint64_t timestamp
) {
    __private uint8_t sbox[SBOX_SIZE];
    __private uint64_t rounds;

    int nonceId = get_global_id(0);
    if (nonceId < nonces_len) {
        if (nonceId == 0) *final_nonce = 0;
        __private uint64_t nonce;
        if (random_type == RANDOM_LEAN) {
            nonce = states[0].s0 ^ nonceId;
        } else {
            __private ulong4 state = states[nonceId];
            nonce = xoshiro256_next(&state);
        }
        nonce = (nonce & nonce_mask) | nonce_fixed;

        // Step 1: BLAKE3(hash_header || nonce) to get input_hash
        __private uint8_t input[80];
        #pragma unroll
        for (int i = 0; i < HASH_HEADER_SIZE; i++) {
            input[i] = hash_header[i];
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            input[HASH_HEADER_SIZE + i] = (nonce >> (8 * i)) & 0xFF;
        }
        __private uint256_t input_hash;
        __private blake3_hasher pow_hasher;
        blake3_hasher_init(&pow_hasher, IV);
        blake3_hasher_update(&pow_hasher, input, 80);
        blake3_hasher_finalize(&pow_hasher, input_hash.hash, BLAKE3_OUT_LEN);

        // Step 2: Compute sbox and rounds
        generate_sbox(input_hash.hash, sbox);
        rounds = calculate_rounds(input_hash.hash, timestamp);

        // Step 3: Initialize result
        __private uint32_t result[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            result[i] = ((const uint32_t*)input_hash.hash)[i];
        }
        __private uint8_t hash_bytes[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            hash_bytes[i] = input_hash.hash[i];
        }

        // Step 4 & 5: BLAKE3 loops with bit manipulations
        #pragma unroll
        for (uint64_t r = 0; r < rounds; r++) {
            __private blake3_hasher hasher;
            blake3_hasher_init(&hasher, IV);
            blake3_hasher_update(&hasher, hash_bytes, 32);
            blake3_hasher_finalize(&hasher, hash_bytes, BLAKE3_OUT_LEN);
            bit_manipulations(hash_bytes);
        }
        #pragma unroll
        for (uint64_t r = 0; r < rounds; r++) {
            __private blake3_hasher hasher;
            blake3_hasher_init(&hasher, IV);
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
        #pragma unroll
        for (uint64_t round = 0; round < rounds; round++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                __private uint8_t state_input[20];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    state_input[j] = ((uint8_t*)&result[i])[j];
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    state_input[4 + j] = ((uint8_t*)&round)[j];
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    state_input[12 + j] = ((uint8_t*)&nonce)[j];
                }
                __private uint8_t state_bytes[32];
                __private blake3_hasher state_hasher;
                blake3_hasher_init(&state_hasher, IV);
                blake3_hasher_update(&state_hasher, state_input, 20);
                blake3_hasher_finalize(&state_hasher, state_bytes, BLAKE3_OUT_LEN);

                __private uint8_t mixed_bytes[32];
                byte_mixing(state_bytes, (__private uint8_t*)result, mixed_bytes);
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

                __private uint8_t b[4];
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
        __private uint8_t* output = (__private uint8_t*)result;
        bit_manipulations(output);

        // Step 9: Apply VecnoHash
        __private uint8_t final_output[32];
        __private blake3_hasher vecno_hasher;
        blake3_hasher_init(&vecno_hasher, IV);
        blake3_hasher_update(&vecno_hasher, output, 32);
        blake3_hasher_finalize(&vecno_hasher, final_output, BLAKE3_OUT_LEN);

        // Step 10: Check against target
        __private uint256_t final_hash;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            final_hash.hash[i] = final_output[i];
        }
        __private uint256_t target_val = *target; // Copy to private memory to avoid address space issues
        if (LT_U256(final_hash, target_val)) {
            atom_cmpxchg(final_nonce, 0, nonce);
        }
    }
}