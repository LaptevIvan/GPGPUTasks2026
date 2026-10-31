#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/merge_path.cl"

__attribute__((reqd_work_group_size(MERGE_GROUP_SIZE, 1, 1)))
__kernel void merge_sort_01_local_sort(
    __global const uint* input,
    __global       uint* output,
    uint n)
{
    uint lane = get_local_id(0);
    size_t begin = get_group_id(0) * (size_t)MERGE_TILE_SIZE;
    uint count = min((size_t)MERGE_TILE_SIZE, (size_t)n - begin);
    __local uint values[MERGE_TILE_SIZE];

    for (uint i = lane; i < count; i += MERGE_GROUP_SIZE) {
        values[i] = input[begin + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint thread_begin = lane * MERGE_VALUES_PER_THREAD;
    uint thread_count = thread_begin < count ? min((uint)MERGE_VALUES_PER_THREAD, count - thread_begin) : 0;
    uint result[MERGE_VALUES_PER_THREAD];
    for (uint width = 1; width < count; width *= 2) {
        uint step = min(2 * width, (uint)MERGE_VALUES_PER_THREAD);
        for (uint k = 0; k < thread_count; k += step) {
            uint index = thread_begin + k;
            uint pair_begin = index / (2 * width) * (2 * width);
            uint a_count = min(width, count - pair_begin);
            uint b_count = min(width, count - pair_begin - a_count);
            merge_local(values + pair_begin, a_count, values + pair_begin + a_count, b_count,
                        index - pair_begin, min(step, thread_count - k), result + k);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < thread_count; ++k) {
            values[thread_begin + k] = result[k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = lane; i < count; i += MERGE_GROUP_SIZE) {
        output[begin + i] = values[i];
    }
}
