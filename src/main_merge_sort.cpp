#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <limits>
#include <random>

void runMergeSort(ocl::KernelSource& local_sort, ocl::KernelSource& partition,
                  ocl::KernelSource& merge, const gpu::gpu_mem_32u& input,
                  gpu::gpu_mem_32u& output, gpu::gpu_mem_32u& scratch,
                  gpu::gpu_mem_32u& partitions, unsigned int n)
{
    if (n == 0) {
        return;
    }
    size_t tiles = (static_cast<size_t>(n) + MERGE_TILE_SIZE - 1) / MERGE_TILE_SIZE;
    gpu::WorkSize merge_work(MERGE_GROUP_SIZE, tiles * MERGE_GROUP_SIZE);
    gpu::WorkSize partition_work(MERGE_GROUP_SIZE, tiles);
    unsigned int passes = 0;
    for (cl_ulong width = MERGE_TILE_SIZE; width < n; width *= 2) {
        ++passes;
    }
    gpu::gpu_mem_32u* values = passes % 2 == 0 ? &output : &scratch;
    gpu::gpu_mem_32u* next = passes % 2 == 0 ? &scratch : &output;
    if (passes == 0) {
        local_sort.exec(merge_work, input, output, n);
        return;
    }
    local_sort.execAsynchronized(merge_work, input, *values, n);
    for (cl_ulong width = MERGE_TILE_SIZE; width < n; width *= 2) {
        partition.execAsynchronized(partition_work, *values, partitions, n, width);
        if (2 * width >= n) {
            merge.exec(merge_work, *values, partitions, *next, n, width);
        } else {
            merge.execAsynchronized(merge_work, *values, partitions, *next, n, width);
        }
        std::swap(values, next);
    }
}

unsigned int parsePositive(const char* text)
{
    std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Expected a positive integer");
    }
    unsigned long long parsed = std::stoull(value);
    if (parsed == 0 || parsed > std::numeric_limits<unsigned int>::max()) {
        throw std::invalid_argument("Expected an integer in [1, UINT_MAX]");
    }
    return static_cast<unsigned int>(parsed);
}

void run(int argc, char** argv)
{
    if (argc > 4) {
        throw std::invalid_argument("Usage: main_merge_sort [device [n [iterations]]]");
    }
    unsigned int n = argc > 2 ? parsePositive(argv[2]) : 100*1000*1000;
    unsigned int iterations = argc > 3 ? parsePositive(argv[3]) : 10;

    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), std::min(argc, 2), argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource local_sort(ocl::getMergeSort01LocalSort());
    ocl::KernelSource partition(ocl::getMergeSort02Partition());
    ocl::KernelSource merge(ocl::getMergeSort03Merge());

    std::mt19937 r(42);
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r();
    }
    if (n > 1) {
        as[1] = as[0];
    }
    std::cout << "n=" << n << " values in range [0; " << std::numeric_limits<unsigned int>::max() << "]" << std::endl;

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        double elapsed = t.elapsed();
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / elapsed << " GB/s (" << n / 1e6 / elapsed << " uint millions/s)" << std::endl;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u buffer_output_gpu(n), scratch_gpu(n);
    size_t tiles = (static_cast<size_t>(n) + MERGE_TILE_SIZE - 1) / MERGE_TILE_SIZE;
    gpu::gpu_mem_32u partitions_gpu(2 * tiles);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (uint64_t iter = 0; iter <= iterations; ++iter) {
        timer t;

        runMergeSort(local_sort, partition, merge, input_gpu, buffer_output_gpu, scratch_gpu, partitions_gpu, n);

        if (iter > 0) {
            times.push_back(t.elapsed());
        }
    }
    std::cout << "GPU merge-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU merge-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = buffer_output_gpu.readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    }
    std::cout << "Verification passed: sorted output and unchanged input" << std::endl;
}

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit_code = 1;
    }

    // we need to gracefully clear Vulkan context before it is too late (otherwise we encounter segfault on some systems)
    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
