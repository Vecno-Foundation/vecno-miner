#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#define STATIC
#else
#define STATIC static
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

STATIC inline uint32_t load32(__private const uint8_t* src) {
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
        block_words[i] = load32(block + 4 * i);
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
STATIC inline ulong rotl(const ulong x, int k) {
    return (x << k) | (x >> (64 - k));
}

STATIC inline ulong xoshiro256_next(__private ulong4 *s) {
    const ulong result = rotl(s->y * 5, 7) * 9;
    const ulong t = s->y << 17;
    s->z ^= s->x;
    s->w ^= s->y;
    s->y ^= s->z;
    s->x ^= s->w;
    s->z ^= t;
    s->w = rotl(s->w, 45);
    return result;
}

/* MATRIX MULTIPLICATION */
#define BLOCKDIM 1024
#define MATRIX_SIZE 64
#define HALF_MATRIX_SIZE 32
#define QUARTER_MATRIX_SIZE 16
#define HASH_HEADER_SIZE 72

#ifndef cl_khr_int64_base_atomics
__global int lock = 0;
#endif

STATIC inline void _amul4bit(__private uchar4 packed_vec1[QUARTER_MATRIX_SIZE], __private uchar4 packed_vec2[QUARTER_MATRIX_SIZE], __private uint32_t *ret) {
    uint32_t res = 0;
    #pragma unroll
    for (int i = 0; i < QUARTER_MATRIX_SIZE; i++) {
        res += (packed_vec1[i].x * packed_vec2[i].x) +
               (packed_vec1[i].y * packed_vec2[i].y) +
               (packed_vec1[i].z * packed_vec2[i].z) +
               (packed_vec1[i].w * packed_vec2[i].w);
    }
    *ret = res;
}

#define amul4bit(X,Y,Z) _amul4bit(X, Y, Z)

typedef union _Hash {
    ulong4 hash;
    uint8_t bytes[32];
} Hash;

#define RANDOM_TYPE_LEAN 0
#define RANDOM_TYPE_XOSHIRO 1

STATIC inline bool compare_u256(__private ulong4 a, __private ulong4 b) {
    if (a.w != b.w) return a.w < b.w;
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

/* KERNEL CODE */
kernel void heavy_hash(
    const ulong local_size,
    const ulong nonce_mask,
    const ulong nonce_fixed,
    const ulong nonces_len,
    const uint8_t random_type,
    __constant const uint8_t *hash_header,
    __constant const uint8_t *matrix,
    __constant const ulong4 *target,
    __global ulong4 *restrict random_state,
    __global volatile uint64_t *final_nonce,
    __global volatile ulong4 *final_hash
) {
    // Safety checks
    if (!hash_header || !matrix || !target || !random_state || !final_nonce || !final_hash || nonces_len == 0) {
        if (get_global_id(0) == 0) {
            *final_nonce = 0xFFFFFFFFFFFFFFFFUL;
        }
        return;
    }

    #if defined(PAL)
    int nonceId = get_group_id(0) * local_size + get_local_id(0);
    #else
    int nonceId = get_global_id(0);
    #endif

    if (nonceId >= nonces_len) return;

    #ifndef cl_khr_int64_base_atomics
    if (nonceId == 0) lock = 0;
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    #endif

    // Nonce generation
    __private uint64_t nonce;
    __private ulong4 state_copy;
    if (random_type == RANDOM_TYPE_LEAN) {
        nonce = ((__global uint64_t*)random_state)[0] ^ nonceId;
    } else {
        state_copy = random_state[nonceId];
        nonce = xoshiro256_next(&state_copy);
    }
    nonce = (nonce & nonce_mask) | nonce_fixed;

    // First BLAKE3 hash
    __private uint8_t input[80];
    __private uint8_t header_cache[HASH_HEADER_SIZE];
    #pragma unroll
    for (int i = 0; i < HASH_HEADER_SIZE; i++) header_cache[i] = hash_header[i];
    #pragma unroll
    for (int i = 0; i < HASH_HEADER_SIZE; i++) input[i] = header_cache[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) input[HASH_HEADER_SIZE + i] = (nonce >> (8 * i)) & 0xFF;

    __private blake3_hasher hasher1;
    blake3_hasher_init(&hasher1, IV);
    blake3_hasher_update(&hasher1, input, 80);
    __private Hash hash_;
    blake3_hasher_finalize(&hasher1, hash_.bytes, BLAKE3_OUT_LEN);

    // Convert hash to 4-bit values
    __private uchar4 packed_hash[QUARTER_MATRIX_SIZE];
    #pragma unroll
    for (int i = 0; i < QUARTER_MATRIX_SIZE / 2; i++) {
        uint bytes = ((uint)hash_.bytes[4 * i] << 24) | ((uint)hash_.bytes[4 * i + 1] << 16) |
                     ((uint)hash_.bytes[4 * i + 2] << 8) | (uint)hash_.bytes[4 * i + 3];
        packed_hash[2 * i] = (uchar4)(
            (bytes >> 28) & 0x0F,
            (bytes >> 24) & 0x0F,
            (bytes >> 20) & 0x0F,
            (bytes >> 16) & 0x0F
        );
        packed_hash[2 * i + 1] = (uchar4)(
            (bytes >> 12) & 0x0F,
            (bytes >> 8) & 0x0F,
            (bytes >> 4) & 0x0F,
            bytes & 0x0F
        );
    }

    // Matrix multiplication
    __private uint8_t product_bytes[32];
    #pragma unroll
    for (int rowId = 0; rowId < HALF_MATRIX_SIZE; rowId++) {
        __private uint32_t product1, product2;
        __private uchar4 packed_matrix1[QUARTER_MATRIX_SIZE], packed_matrix2[QUARTER_MATRIX_SIZE];
        #pragma unroll
        for (int i = 0; i < QUARTER_MATRIX_SIZE; i++) {
            #ifdef EXPERIMENTAL_AMD
            // Packed matrix: each byte contains two 4-bit values
            uint8_t m1 = matrix[((2 * rowId) * MATRIX_SIZE + 4 * i) / 2];
            uint8_t m2 = matrix[((2 * rowId) * MATRIX_SIZE + 4 * i + 2) / 2];
            uint8_t m3 = matrix[((2 * rowId + 1) * MATRIX_SIZE + 4 * i) / 2];
            uint8_t m4 = matrix[((2 * rowId + 1) * MATRIX_SIZE + 4 * i + 2) / 2];
            packed_matrix1[i] = (uchar4)(
                (m1 & 0xF0) >> 4,
                (m1 & 0x0F),
                (m2 & 0xF0) >> 4,
                (m2 & 0x0F)
            );
            packed_matrix2[i] = (uchar4)(
                (m3 & 0xF0) >> 4,
                (m3 & 0x0F),
                (m4 & 0xF0) >> 4,
                (m4 & 0x0F)
            );
            #else
            // Unpacked matrix: each byte is a 4-bit value
            packed_matrix1[i] = (uchar4)(
                matrix[(2 * rowId) * MATRIX_SIZE + 4 * i] & 0x0F,
                matrix[(2 * rowId) * MATRIX_SIZE + 4 * i + 1] & 0x0F,
                matrix[(2 * rowId) * MATRIX_SIZE + 4 * i + 2] & 0x0F,
                matrix[(2 * rowId) * MATRIX_SIZE + 4 * i + 3] & 0x0F
            );
            packed_matrix2[i] = (uchar4)(
                matrix[(2 * rowId + 1) * MATRIX_SIZE + 4 * i] & 0x0F,
                matrix[(2 * rowId + 1) * MATRIX_SIZE + 4 * i + 1] & 0x0F,
                matrix[(2 * rowId + 1) * MATRIX_SIZE + 4 * i + 2] & 0x0F,
                matrix[(2 * rowId + 1) * MATRIX_SIZE + 4 * i + 3] & 0x0F
            );
            #endif
        }

        amul4bit(packed_matrix1, packed_hash, &product1);
        amul4bit(packed_matrix2, packed_hash, &product2);

        // Optimized reduction logic
        product_bytes[rowId] = hash_.bytes[rowId] ^ ((uint8_t)((product1 >> 6) & 0xF0) | (uint8_t)((product2 >> 10) & 0x0F));
    }

    // Second BLAKE3 hash
    __private blake3_hasher hasher2;
    blake3_hasher_init(&hasher2, IV);
    blake3_hasher_update(&hasher2, product_bytes, 32);
    blake3_hasher_finalize(&hasher2, hash_.bytes, BLAKE3_OUT_LEN);

    // Check target
    __private ulong4 target_value = *target;
    #ifdef cl_khr_int64_base_atomics
    if (*final_nonce == 0 && compare_u256(hash_.hash, target_value)) {
        atom_cmpxchg(final_nonce, 0, nonce);
        *final_hash = hash_.hash;
    }
    #else
    if (!atom_cmpxchg(&lock, 0, 1)) {
        *final_nonce = nonce;
        *final_hash = hash_.hash;
    }
    #endif
}