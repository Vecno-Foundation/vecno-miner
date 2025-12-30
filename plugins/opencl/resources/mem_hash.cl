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

/* BLAKE3 CONSTANTS */
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

STATIC inline uint32_t rotr32(uint32_t w, uint32_t c) {
    return (w >> c) | (w << (32 - c));
}

STATIC inline uint32_t load32(private const uint8_t* src) {
    return ((uint32_t)(src[0]) << 0) | ((uint32_t)(src[1]) << 8) |
           ((uint32_t)(src[2]) << 16) | ((uint32_t)(src[3]) << 24);
}

STATIC inline void store32(private uint8_t* dst, uint32_t w) {
    dst[0] = (uint8_t)(w >> 0);
    dst[1] = (uint8_t)(w >> 8);
    dst[2] = (uint8_t)(w >> 16);
    dst[3] = (uint8_t)(w >> 24);
}

STATIC inline void g(private uint32_t* state, size_t a, size_t b, size_t c, size_t d, uint32_t x, uint32_t y) {
    state[a] = state[a] + state[b] + x;
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + y;
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7);
}

STATIC inline void round_fn(private uint32_t state[16], private const uint32_t* msg, size_t round) {
    constant const uint8_t* schedule = MSG_SCHEDULE[round];
    g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
    g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
    g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
    g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);
    g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
    g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
    g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
    g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

STATIC inline void blake3_hash_80bytes(private const uint8_t* input, private uint8_t* output) {
    private uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = IV[i];

    private uint32_t state[16];
    private uint32_t block_words[16];

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

STATIC inline void blake3_simple_hash(private const uint8_t* input, int len, private uint8_t* output) {
    private uint32_t state[16];
    private uint32_t block_words[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) block_words[i] = 0;
    #pragma unroll
    for (int i = 0; i < len; i++) ((private uint8_t*)block_words)[i] = input[i];

    state[0] = IV[0]; state[1] = IV[1]; state[2] = IV[2]; state[3] = IV[3];
    state[4] = IV[4]; state[5] = IV[5]; state[6] = IV[6]; state[7] = IV[7];
    state[8] = IV[0]; state[9] = IV[1]; state[10] = IV[2]; state[11] = IV[3];
    state[12] = 0; state[13] = 0;
    state[14] = (uint32_t)len;
    state[15] = CHUNK_START | CHUNK_END | ROOT;

    #pragma unroll
    for (int r = 0; r < 7; r++) round_fn(state, block_words, r);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store32(output + i * 4, state[i] ^ state[i + 8]);
    }
}

/* XOSHIRO256** */
STATIC inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

STATIC inline uint64_t xoshiro256_next(private ulong4* s) {
    const uint64_t result = rotl(s->x * 5, 7) * 9;
    const uint64_t t = s->y << 17;
    s->z ^= s->x;
    s->w ^= s->y;
    s->y ^= s->z;
    s->x ^= s->w;
    s->z ^= t;
    s->w = rotl(s->w, 45);
    return result;
}

#define BLOCKDIM 128
#define SBOX_SIZE 64
#define HASH_HEADER_SIZE 72
#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

#define LT_U256(X, Y) (X.number[3] != Y.number[3] ? X.number[3] < Y.number[3] : \
                       X.number[2] != Y.number[2] ? X.number[2] < Y.number[2] : \
                       X.number[1] != Y.number[1] ? X.number[1] < Y.number[1] : \
                       X.number[0] < Y.number[0])

STATIC inline void bit_manipulations(private uint8_t* data) {
    private uint32_t* d = (private uint32_t*)data;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t val = d[i];
        uint32_t xor1 = (val >> 8) & 0xFF;
        uint32_t xor2 = ((val >> 24) & 0xFF) << 16;
        d[i] = val ^ xor1 ^ xor2;
    }
}

STATIC inline void byte_mixing(private const uint8_t* b3_hash1, private const uint8_t* b3_hash2, private uint8_t* result) {
    private const uint32_t* h1 = (private const uint32_t*)b3_hash1;
    private const uint32_t* h2 = (private const uint32_t*)b3_hash2;
    private uint32_t* res = (private uint32_t*)result;
    #pragma unroll
    for (int i = 0; i < 8; i++) res[i] = h1[i] ^ h2[i];
}

STATIC inline void generate_sbox(private const uint8_t* input_bytes, private uint8_t* sbox) {
    private uint8_t seed[32];
    blake3_simple_hash(input_bytes, 32, seed);

    #pragma unroll
    for (int i = 0; i < 64; i += 2) {
        sbox[i]     = seed[0];
        sbox[i + 1] = seed[1];
        blake3_simple_hash(seed, 32, seed);
    }
}

STATIC inline uint64_t calculate_rounds(private const uint8_t* input_bytes, uint64_t timestamp) {
    private uint8_t buffer[40];
    #pragma unroll
    for (int i = 0; i < 32; i++) buffer[i] = input_bytes[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) buffer[32 + i] = (timestamp >> (i * 8)) & 0xFF;
    private uint8_t hash[32];
    blake3_simple_hash(buffer, 40, hash);
    uint32_t value = ((private const uint32_t*)hash)[0];
    return (value % 8 + 16);
}

kernel void mem_hash(
    const uint64_t nonce_mask,
    const uint64_t nonce_fixed,
    const uint64_t nonces_len,
    const uint8_t random_type,
    constant const uint8_t* hash_header,
    constant const uint256_t* target,
    global ulong4* states,
    global volatile uint64_t* final_nonce,
    const uint64_t timestamp
) {
    private uint8_t sbox[SBOX_SIZE];

    int nonceId = get_global_id(0);
    if (nonceId >= nonces_len) return;

    if (nonceId == 0) *final_nonce = 0;

    uint64_t nonce;
    if (random_type == RANDOM_LEAN) {
        nonce = states[0].x ^ nonceId;
    } else {
        private ulong4 state = states[nonceId];
        nonce = xoshiro256_next(&state);
    }
    nonce = (nonce & nonce_mask) | nonce_fixed;

    private uint8_t input[80];
    #pragma unroll
    for (int i = 0; i < HASH_HEADER_SIZE; i++) input[i] = hash_header[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) input[HASH_HEADER_SIZE + i] = (nonce >> (i * 8)) & 0xFF;

    private uint8_t input_hash_bytes[32];
    blake3_hash_80bytes(input, input_hash_bytes);

    private uint256_t input_hash;
    #pragma unroll
    for (int i = 0; i < 32; i++) input_hash.hash[i] = input_hash_bytes[i];

    generate_sbox(input_hash.hash, sbox);
    uint64_t rounds = calculate_rounds(input_hash.hash, timestamp);

    private uint32_t result[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result[i] = ((private const uint32_t*)input_hash.hash)[i];
    }

    private uint8_t hash_bytes[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) hash_bytes[i] = input_hash.hash[i];

    #pragma unroll
    for (uint64_t r = 0; r < 2 * rounds; r++) {
        blake3_simple_hash(hash_bytes, 32, hash_bytes);
        bit_manipulations(hash_bytes);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result[i] = ((private const uint32_t*)hash_bytes)[i];
    }

    #pragma unroll
    for (uint64_t round = 0; round < rounds; round++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            private uint8_t state_input[20];
            *(private uint32_t*)state_input         = result[i];
            *(private uint64_t*)(state_input + 4)   = round;
            *(private uint64_t*)(state_input + 12)  = nonce;
            private uint8_t state_bytes[32];
            blake3_simple_hash(state_input, 20, state_bytes);
            private uint8_t mixed_bytes[32];
            byte_mixing(state_bytes, (private const uint8_t*)result, mixed_bytes);
            uint32_t v = ((private const uint32_t*)mixed_bytes)[0] ^ result[i];
            uint32_t branch = v & 3;
            uint32_t next = result[(i + 1) % 8];
            if (branch == 0) v += next;
            else if (branch == 1) v -= next;
            else if (branch == 2) { uint32_t shift = next & 0x1F; v = (v << shift) | (v >> (32 - shift)); }
            else v ^= next;
            private uint8_t b[4];
            b[0] = v & 0xFF; b[1] = (v >> 8) & 0xFF; b[2] = (v >> 16) & 0xFF; b[3] = (v >> 24) & 0xFF;
            uint32_t idx_base = v & 63;
            b[0] = sbox[(idx_base + b[0]) & 63];
            b[1] = sbox[(idx_base + b[1]) & 63];
            b[2] = sbox[(idx_base + b[2]) & 63];
            b[3] = sbox[(idx_base + b[3]) & 63];
            result[i] = ((uint32_t)b[0]) | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
        }
    }

    bit_manipulations((private uint8_t*)result);

    private uint8_t final_output[32];
    blake3_simple_hash((private const uint8_t*)result, 32, final_output);

    private uint256_t final_hash;
    #pragma unroll
    for (int i = 0; i < 32; i++) final_hash.hash[i] = final_output[i];

    private uint256_t target_val = *target;
    if (LT_U256(final_hash, target_val)) {
        atom_cmpxchg(final_nonce, 0UL, nonce);
    }
}