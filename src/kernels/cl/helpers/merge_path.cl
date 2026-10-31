#ifndef MERGE_PATH_CL
#define MERGE_PATH_CL

inline uint merge_path_global(__global const uint* a, uint a_count,
                              __global const uint* b, uint b_count, uint diag)
{
    uint low = diag > b_count ? diag - b_count : 0;
    uint high = min(diag, a_count);
    while (low < high) {
        uint mid = low + (high - low) / 2;
        uint j = diag - mid;
        if (j > 0 && mid < a_count && b[j - 1] >= a[mid]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

inline uint merge_path_local(__local const uint* a, uint a_count,
                             __local const uint* b, uint b_count, uint diag)
{
    uint low = diag > b_count ? diag - b_count : 0;
    uint high = min(diag, a_count);
    while (low < high) {
        uint mid = low + (high - low) / 2;
        uint j = diag - mid;
        if (j > 0 && mid < a_count && b[j - 1] >= a[mid]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

inline void merge_local(__local const uint* a, uint a_count,
                        __local const uint* b, uint b_count, uint diag,
                        uint count, __private uint* result)
{
    uint i = merge_path_local(a, a_count, b, b_count, diag);
    uint j = diag - i;
    for (uint k = 0; k < count; ++k) {
        if (j == b_count || (i < a_count && a[i] <= b[j])) {
            result[k] = a[i++];
        } else {
            result[k] = b[j++];
        }
    }
}

#endif
