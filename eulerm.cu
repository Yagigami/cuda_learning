#include <stdio.h>
#include <math.h>
#include <assert.h>

__global__ void partial_sum(long num, double *out) {
        int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, index = x + y*blockDim.x*gridDim.x;
        double sum = 0.0;
        double cur = index*num + 1;
        for (long i = 0; i < num; ++i) {
                sum += 1.0/cur;
                cur += 1.0;
        }
        out[index] = sum;
}

__global__ void add_harmonics(double start, double *partials, long num) {
        partials[num] = start;
        for (long i = 0; i < num; ++i) {
                partials[num] += partials[i];
        }
}

int main(int argc, char **argv) {
        if (argc < 2) {
                printf("usage:\n%s <N_ITERATIONS>\n", *argv);
                return -1;
        }
        dim3 block(32, 8);
        long threads_per_block = block.x * block.y, block_w = 6, block_h = 2, blocks = block_w * block_h, threads = threads_per_block*blocks;
        long terms = (long)strtod(argv[1], 0), iterations_per_thread = terms/threads, iterations_left = terms%threads;
        long bytes = (threads+1) * sizeof(double); // last elem is sum of all
        dim3 grid(block_w, block_h);
        double *partials, harmonics = 0.0;
        for (long i = terms-iterations_left; i <= terms; ++i) {
                harmonics += 1.0/i;
        }
        cudaMalloc(&partials, bytes);
        partial_sum <<<grid, block>>> (iterations_per_thread, partials);
        cudaDeviceSynchronize();
        add_harmonics <<<1, 1>>> (harmonics, partials, threads); // we want to compute the sum of partial sums on the device
        cudaMemcpy(&harmonics, partials+threads, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(partials);
        double gamma = harmonics - log(terms);
        printf("%.17f\n", gamma);
        return 0;
}
