# matmul
Implementations of matrix multiplication
### Perfomance benchmark
Input: Matrix1 size = 1024x1024, Matrix2 size = 1024x1024<br>
<br>
CPU Intel Core i3 7100U inference time:<br>
matmul = 9,30166s (1x)<br>
matmul_parallel_omp (4 threads) = 5,86126s (~1.5x)<br>
<br>
matmul_blocks (blockSize=4x4) = 4,52108s (~2x)<br>
matmul_blocks_parallel_omp (blockSize=4x4, 4 threads) = 1,80083s (~5x)<br>
<br>
matmul_blocks_4x4_simd = 2,52983s (~4x)<br>
matmul_blocks_4x4_simd_parallel_omp (4 threads)  = 0,851757s (~11x)<br>
<br>
matmul_parallel_tbb1D (4 threads) = 5,58702s (~1.5x)<br>
matmul_parallel_tbb2D (4 threads) = 4,74627s (~2x)<br>
matmul_blocks_parallel_tbb1D (4 threads) = 1,78654s (~5x)<br>
matmul_blocks_parallel_tbb2D (4 threads) = 1,77245s (~5x)<br>
<br>
matmul_blocks MPI (blockSize=4x4, 4 localhost processes) = 2,28233s (~4x)<br>
<br>
GPU Intel HD Graphics 620 inference time:<br>
matmul OpenCL = 0,120521s (~77x)<br>
matmul2 OpenCL = 0,0578637s (~160x)<br>
matmul3 OpenCL = 0,0708066s (~130x)<br>