// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/common.cl"
#include "include/fetch.cl"
#include "include/unit_type.cl"

#if TILE_K > SIMD_WIDTH
    #define BLOCK_READ_A(ptr, offset) CAT(UNIT_BLOCK_READ, A_VEC_SIZE)(ptr, offset)
#else
    #define BLOCK_READ_A(ptr, offset) UNIT_BLOCK_READ(ptr, offset)
#endif

#if TILE_N > SIMD_WIDTH
    #define BLOCK_READ_B(ptr, offset) CAT(UNIT_BLOCK_READ, B_VEC_SIZE)(ptr, offset)
    #define BLOCK_WRITE_C(ptr, offset, data) CAT(UNIT_BLOCK_WRITE, B_VEC_SIZE)(ptr, offset, data)
#else
    #define BLOCK_READ_B(ptr, offset) UNIT_BLOCK_READ(ptr, offset)
    #define BLOCK_WRITE_C(ptr, offset, data) UNIT_BLOCK_WRITE(ptr, offset, data)
#endif

__attribute__((intel_reqd_sub_group_size(SIMD_WIDTH)))
__attribute__((reqd_work_group_size(SIMD_WIDTH, 1, 1)))
KERNEL(gemm_nn_tiled)(
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
    #ifdef INPUT2_TYPE
        const __global INPUT2_TYPE* input2,
    #endif
    __global OUTPUT_TYPE* output
    )
{
    const uint tile_n_num = (uint)get_group_id(0);
    const uint global_id_n = (uint)get_global_id(0);
    const uint tile_m_num = (uint)get_group_id(1);
    const uint tile_m_size = (uint)get_global_size(1);
    const uint batch_number = (uint)get_group_id(2);

    #if TILE_M_NOT_DIVISIBLE
        const uint tile_m_iterations = tile_m_num == (tile_m_size - 1) ? TILE_M_LEFTOVER : TILE_M;
    #endif

    const uint local_simd_id = (uint)get_local_id(0);

    // Start pointers offsets
    const __global float* a_ptr = input0 + (batch_number * M * K) + tile_m_num * TILE_M * K;
    const __global float* b_ptr = input1 + (batch_number * K * N) + tile_n_num * TILE_N;
    #ifdef INPUT2_TYPE
        const __global float* c_ptr = input2 + (batch_number * M * N) + (tile_m_num * TILE_M * N) + (tile_n_num * TILE_N);
    #endif
    __global float* d_ptr = output + (batch_number * M * N) + (tile_m_num * TILE_M * N) + (tile_n_num * TILE_N);

    uint b_raw_global_id = tile_n_num * TILE_N + local_simd_id;

    B_FLOATN b_tile[TILE_K] = {};
    B_FLOATN c_tile[TILE_M] = {};

    // Full Tile calc
    for(uint k = 0; k < K_FULL_ITERATIONS; k++)
    {
        // Loading B tile
        __attribute__((opencl_unroll_hint(TILE_K)))
        for(uint b_load_id = 0; b_load_id < TILE_K; b_load_id++)
        {
            #if TILE_N_NOT_DIVISIBLE
                #if TILE_N == SIMD_WIDTH
                    if((b_raw_global_id) > N - 1)
                        b_tile[b_load_id] = 0;
                    else
                        b_tile[b_load_id] = b_ptr[local_simd_id];
                #else
                    __attribute__((opencl_unroll_hint(TILE_N / SIMD_WIDTH)))
                    for(uint br_id = 0; br_id < TILE_N; br_id += SIMD_WIDTH) {
                    if((b_raw_global_id + br_id) > N - 1)
                        b_tile[b_load_id][br_id / SIMD_WIDTH] = 0;
                    else
                        b_tile[b_load_id][br_id / SIMD_WIDTH] = b_ptr[br_id + local_simd_id];
                    }
                #endif
            #else
                b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
            #endif

                b_ptr += N;
        }
        #if TILE_M_NOT_DIVISIBLE
        for(uint dot_id = 0; dot_id < tile_m_iterations; dot_id++)
        #else
        __attribute__((opencl_unroll_hint(TILE_M)))
        for(uint dot_id = 0; dot_id < TILE_M; dot_id++)
        #endif
        {
            A_FLOATN a_read = BLOCK_READ_A(a_ptr, (dot_id * K));
            
            __attribute__((opencl_unroll_hint(TILE_K / SIMD_WIDTH)))
            for(uint subtile_k_id = 0; subtile_k_id < TILE_K / SIMD_WIDTH; subtile_k_id++){
                __attribute__((opencl_unroll_hint(SIMD_WIDTH)))
                for(uint simd_local_id = 0; simd_local_id < SIMD_WIDTH; simd_local_id++)
                {
                    c_tile[dot_id] = mad((float)(sub_group_broadcast(
                        #if TILE_K > SIMD_WIDTH
                                    a_read[subtile_k_id], simd_local_id)),
                                    b_tile[subtile_k_id * SIMD_WIDTH + simd_local_id],
                                    c_tile[dot_id]);
                        #else
                                    a_read, simd_local_id)),
                                    b_tile[simd_local_id],
                                    c_tile[dot_id]);
                        #endif
                }
            }
        }
        a_ptr += TILE_K; 
    }

    #if TILE_K_NOT_DIVISIBLE
        __attribute__((opencl_unroll_hint(TILE_K_LEFTOVER)))
        for(uint b_load_id = 0; b_load_id < TILE_K_LEFTOVER; b_load_id++)
        {
            #if TILE_N_NOT_DIVISIBLE
                #if TILE_N == SIMD_WIDTH
                    if(b_raw_global_id > N - 1)
                        b_tile[b_load_id] = 0;
                    else
                        b_tile[b_load_id] = b_ptr[local_simd_id];
                #else
                    __attribute__((opencl_unroll_hint(TILE_N / SIMD_WIDTH)))
                    for(uint br_id = 0; br_id < TILE_N; br_id += SIMD_WIDTH) {
                    if((tile_n_num * TILE_N + br_id + local_simd_id) > N - 1)
                        b_tile[b_load_id][br_id / SIMD_WIDTH] = 0;
                    else
                        b_tile[b_load_id][br_id / SIMD_WIDTH] = b_ptr[br_id + local_simd_id];
                    }
                #endif
            #else
                b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
            #endif
                b_ptr += N;
        }

        #if TILE_M_NOT_DIVISIBLE
        for(uint dot_id = 0; dot_id < tile_m_iterations; dot_id++)
        #else
        __attribute__((opencl_unroll_hint(TILE_M)))
        for(uint dot_id = 0; dot_id < TILE_M; dot_id++)
        #endif
        {
            float a_read = a_ptr[dot_id * K + local_simd_id];

            __attribute__((opencl_unroll_hint(TILE_K_LEFTOVER)))
            for(uint simd_id = 0; simd_id < TILE_K_LEFTOVER; simd_id++)
            {
                c_tile[dot_id] = mad((float)(sub_group_broadcast(
                                     a_read, simd_id)),
                                     b_tile[simd_id],
                                     c_tile[dot_id]);
            }
        }
    #endif
    
        #if TILE_M_NOT_DIVISIBLE
        for(uint write_id = 0; write_id < tile_m_iterations; write_id++)
        #else
        __attribute__((opencl_unroll_hint(TILE_M)))
        for(uint write_id = 0; write_id < TILE_M; write_id++)
        #endif
        {
            #if TILE_N_NOT_DIVISIBLE
                #if TILE_N == SIMD_WIDTH
                if(b_raw_global_id < N)
                    #ifdef INPUT2_TYPE
                        d_ptr[local_simd_id] = ALPHA * c_tile[write_id] + BETA * c_ptr[local_simd_id];
                    #else
                        d_ptr[local_simd_id] = ALPHA * c_tile[write_id];
                    #endif
                #else
                for(uint br_id = 0; br_id < TILE_N; br_id += SIMD_WIDTH){
                    if(b_raw_global_id + br_id < N)
                        #ifdef INPUT2_TYPE
                            d_ptr[br_id + local_simd_id] = ALPHA * c_tile[write_id][br_id / SIMD_WIDTH] + BETA * c_ptr[br_id + local_simd_id];
                        #else
                            d_ptr[br_id + local_simd_id] = ALPHA * c_tile[write_id][br_id / SIMD_WIDTH];
                        #endif
                }
                #endif
            #else
                #ifdef INPUT2_TYPE
                    B_FLOATN c_val = BLOCK_READ_B(c_ptr, 0);
                    BLOCK_WRITE_C(d_ptr, 0, ALPHA * c_tile[write_id] + BETA * c_val);
                #else
                    BLOCK_WRITE_C(d_ptr, 0, ALPHA * c_tile[write_id]);
                #endif
            #endif
            d_ptr += N;
            #ifdef INPUT2_TYPE
                c_ptr += N;
            #endif
        }
}
