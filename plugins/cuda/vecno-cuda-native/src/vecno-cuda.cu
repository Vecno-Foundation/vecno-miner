#include<stdint.h>
#include <assert.h>
#include "xoshiro256starstar.c"
#include "blake3_compact.h"


typedef uint8_t Hash[32];

typedef union _uint256_t {
    uint64_t number[4];
    uint8_t hash[32];
} uint256_t;

#define BLOCKDIM 1024
#define MATRIX_SIZE 64
#define HALF_MATRIX_SIZE 32
#define QUARTER_MATRIX_SIZE 16
#define HASH_HEADER_SIZE 72

#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

#define LT_U256(X,Y) (X.number[3] != Y.number[3] ? X.number[3] < Y.number[3] : X.number[2] != Y.number[2] ? X.number[2] < Y.number[2] : X.number[1] != Y.number[1] ? X.number[1] < Y.number[1] : X.number[0] < Y.number[0])

__constant__ uint8_t matrix[MATRIX_SIZE][MATRIX_SIZE];
__constant__ uint8_t hash_header[HASH_HEADER_SIZE];
__constant__ uint256_t target;

__device__ __inline__ void amul4bit(uint32_t packed_vec1[32], uint32_t packed_vec2[32], uint32_t *ret) {
    // We assume each 32 bits have four values: A0 B0 C0 D0
    unsigned int res = 0;
    #if __CUDA_ARCH__ < 610
    char4 *a4 = (char4*)packed_vec1;
    char4 *b4 = (char4*)packed_vec2;
    #endif
    #pragma unroll
    for (int i=0; i<QUARTER_MATRIX_SIZE; i++) {
        #if __CUDA_ARCH__ >= 610
        res = __dp4a(packed_vec1[i], packed_vec2[i], res);
        #else
        res += a4[i].x*b4[i].x;
        res += a4[i].y*b4[i].y;
        res += a4[i].z*b4[i].z;
        res += a4[i].w*b4[i].w;
        #endif
    }

    *ret = res;
}


extern "C" {


    __global__ void heavy_hash(const uint64_t nonce_mask, const uint64_t nonce_fixed, const uint64_t nonces_len, uint8_t random_type, void* states, uint64_t *final_nonce) {
        // assuming header_len is 72
        int nonceId = threadIdx.x + blockIdx.x*blockDim.x;
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
            // header
            uint8_t input[80];
            memcpy(input, hash_header, HASH_HEADER_SIZE);
            // data
            // TODO: check endianity?
            uint256_t hash_;
            memcpy(input +  HASH_HEADER_SIZE, (uint8_t *)(&nonce), 8);
            blake3_hasher pow_hasher;
            blake3_hasher_init(&pow_hasher);
            blake3_hasher_update(&pow_hasher, input, 80);
            blake3_hasher_finalize(&pow_hasher, hash_.hash, BLAKE3_KEY_LEN);

            //assert((rowId != 0) || (hashId != 0) );
            uchar4 packed_hash[QUARTER_MATRIX_SIZE] = {0};
            #pragma unroll
            for (int i=0; i<QUARTER_MATRIX_SIZE; i++) {
                packed_hash[i] = make_uchar4(
                    (hash_.hash[2*i] & 0xF0) >> 4 ,
                    (hash_.hash[2*i] & 0x0F),
                    (hash_.hash[2*i+1] & 0xF0) >> 4,
                    (hash_.hash[2*i+1] & 0x0F)
                );
            }
            uint32_t product1, product2;
            #pragma unroll
            for (int rowId=0; rowId<HALF_MATRIX_SIZE; rowId++){

                amul4bit((uint32_t *)(matrix[(2*rowId)]), (uint32_t *)(packed_hash), &product1);
                amul4bit((uint32_t *)(matrix[(2*rowId+1)]), (uint32_t *)(packed_hash), &product2);
                product1 >>= 6;
                product1 &= 0xF0;
                product2 >>= 10;
                #if __CUDA_ARCH__ < 500 || __CUDA_ARCH__ > 700
                hash_.hash[rowId] = hash_.hash[rowId] ^ ((uint8_t)(product1) | (uint8_t)(product2));
                #else
                uint32_t lop_temp = hash_.hash[rowId];
                asm("lop3.b32" " %0, %1, %2, %3, 0x56;": "=r" (lop_temp): "r" (product1), "r" (product2), "r" (lop_temp));
                hash_.hash[rowId] = lop_temp;
                #endif
            }
            memset(input, 0, 80);
            memcpy(input, hash_.hash, 32);
            blake3_hasher heavy_hasher;
            blake3_hasher_init(&heavy_hasher);
            blake3_hasher_update(&heavy_hasher, input, 32);
            blake3_hasher_finalize(&heavy_hasher, hash_.hash, BLAKE3_KEY_LEN);
            if (LT_U256(hash_, target)){
                atomicCAS((unsigned long long int*) final_nonce, 0, (unsigned long long int) nonce);
            }
        }
    }

}
