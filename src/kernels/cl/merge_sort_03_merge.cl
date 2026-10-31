#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/merge_path.cl"

__attribute__((reqd_work_group_size(MERGE_GROUP_SIZE, 1, 1)))
__kernel void merge_sort_03_merge(
    __global const uint* input,
    __global const uint* partitions,
    __global       uint* output,
    uint n,
    ulong sorted_count)
{
    uint lane = get_local_id(0);
    size_t tile = get_group_id(0);
    ulong begin = (ulong)tile * MERGE_TILE_SIZE;
    ulong pair_begin = begin / (2 * sorted_count) * (2 * sorted_count);
    uint pair_a_count = min(sorted_count, (ulong)n - pair_begin);
    uint diag = begin - pair_begin;
    uint count = min((ulong)MERGE_TILE_SIZE, (ulong)n - begin);
    uint a_begin = partitions[2 * tile];
    uint a_count = partitions[2 * tile + 1] - a_begin;
    uint b_begin = diag - a_begin;
    uint b_count = count - a_count;
    __local uint values[MERGE_TILE_SIZE];

    for (uint i = lane; i < a_count; i += MERGE_GROUP_SIZE) {
        values[i] = input[pair_begin + a_begin + i];
    }
    for (uint i = lane; i < b_count; i += MERGE_GROUP_SIZE) {
        values[a_count + i] = input[pair_begin + pair_a_count + b_begin + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint thread_begin = lane * MERGE_VALUES_PER_THREAD;
    uint thread_count = thread_begin < count ? min((uint)MERGE_VALUES_PER_THREAD, count - thread_begin) : 0;
    uint result[MERGE_VALUES_PER_THREAD];
    if (thread_count > 0) {
        merge_local(values, a_count, values + a_count, b_count, thread_begin, thread_count, result);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < thread_count; ++k) {
        values[thread_begin + k] = result[k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = lane; i < count; i += MERGE_GROUP_SIZE) {
        output[begin + i] = values[i];
    }
}
