// naive implementation
__kernel void matmul(__global const float *m1, __global const float *m2, __global float *mr, int n)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

    float s = 0.0f;
    for (int k = 0; k < n; k++) {
        s += m1[row * n + k] * m2[k * w + col];
    }

    mr[row * w + col] = s;
}

// optimized implementation with using blocks and local memory
__kernel void matmul2(__global const float *m1, __global const float *m2, __global float *mr, int n,
__local float *block1, __local float *block2)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);
	int block_col = get_local_id(0);
	int block_row = get_local_id(1);
	int block_w = get_local_size(0);
	int block_h = get_local_size(1);

    float s = 0.0f;
    for (int blockIter = 0; blockIter < n; blockIter += block_w) {
        block1[block_row * block_w + block_col] = m1[row * n + blockIter + block_col];
		block2[block_row * block_h + block_col] = m2[blockIter * w + block_row * w];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < block_w; k++) {
			s += block1[block_row * block_w + k] * block2[block_col + k * block_h];
		}
    }

    mr[row * w + col] = s;
}

// nvidia implementation example (optimized implementation with using blocks and local memory)
#define BLOCK_SIZE 16
#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]

__kernel void matrixMul( __global float* C, __global float* A, __global float* B, 
	   __local float* As, __local float* Bs, int uiWA, int uiWB)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed by the block
    int aBegin = uiWA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + uiWA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * uiWB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0f;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + uiWA * ty + tx];
        BS(ty, tx) = B[b + uiWB * ty + tx];
	
        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //if (get_global_id(1) < trueLocalSize1)
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;
}

// TODO
// optimized implementation with using blocks, local memory and private memory
__kernel void matmul4(__global const float *m1, __global const float *m2, __global float *mr, int n,
__local float *block1, __local float *block2)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);
	int block_col = get_local_id(0);
	int block_row = get_local_id(1);
	int block_w = get_local_size(0);
	int block_h = get_local_size(1);

	float s_arr[16];
    for (int blockIter = 0; blockIter < n; blockIter += block_w) {
        block1[block_row * block_w + block_col] = m1[row * n + blockIter + block_col];
		block2[block_col * block_h + block_row] = m2[blockIter * w + block_row * w];

		barrier(CLK_LOCAL_MEM_FENCE);

		//for (int k = 0; k < block_w; k++) {
		//	s += block1[block_row * block_w + k] * block2[block_col + k * block_h];
		//}

		for (int k = 0; k < block_w; k++) {
			float elBlock1 = block1[block_row * block_w + k];
			for (int elIter = 0; elIter < 16; elIter++) {
				s_arr[elIter] += elBlock1 * block2[block_col * 16 + k * block_h + elIter];
			}
		}
    }

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int elIter = 0; elIter < 16; elIter++) {
		mr[row * w + 16 * col + elIter] = s_arr[elIter];
	}
}