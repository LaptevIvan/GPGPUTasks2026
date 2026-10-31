#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/merge_path.cl"

__attribute__((reqd_work_group_size(MERGE_GROUP_SIZE, 1, 1)))
__kernel void merge_sort_02_partition(
    __global const uint* input,
    __global       uint* partitions,
    uint n,
    ulong sorted_count)
{
    size_t tile = get_global_id(0);
    ulong begin = (ulong)tile * MERGE_TILE_SIZE;
    if (begin >= n) {
        return;
    }

    ulong pair_begin = begin / (2 * sorted_count) * (2 * sorted_count);
    uint a_count = min(sorted_count, (ulong)n - pair_begin);
    uint b_count = min(sorted_count, (ulong)n - pair_begin - a_count);
    uint diag = begin - pair_begin;
    uint end_diag = diag + min((ulong)MERGE_TILE_SIZE, (ulong)n - begin);
    __global const uint* a = input + pair_begin;
    __global const uint* b = a + a_count;

    partitions[2 * tile] = merge_path_global(a, a_count, b, b_count, diag);
    partitions[2 * tile + 1] = merge_path_global(a, a_count, b, b_count, end_diag);
}
