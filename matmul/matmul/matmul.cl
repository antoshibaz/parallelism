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