#include "pch.h"
#include <iostream>
#include <omp.h>

struct matsize {
    int w;
    int h;
};

int equalsMat(const float *m1, const float*m2,
    const matsize s1, const matsize s2) {
    if (s1.w != s2.w || s1.h != s2.h) return -1;
    int c = 0;
    for (int i = 0; i < s1.w * s1.h; i++) {
        if (m1[i] != m2[i]) c++;
    }

    return c;
}

void test_omp() {
    #pragma omp parallel
    {
        printf("This thread id=%d; Total threads=%d\n", omp_get_thread_num(), omp_get_num_threads());
    }
}

void matmul_parallel_omp(const float *m1, const float *m2, float *rm,
    const matsize s1, const matsize s2) {
    if (s1.w != s2.h) return;

    #pragma omp parallel for
    for (int i = 0; i < s1.h; i++) {
        for (int j = 0; j < s2.w; j++) {
            float s = 0.0;
            for (int k = 0; k < s1.w; k++) {
                s += m1[s1.w * i + k] * m2[j + s2.w * k];
            }
            rm[s2.w * i + j] = s;
        }
    }
}

void matmul_blocks_parallel_omp(const float *m1, const float *m2, float *rm,
    const matsize s1, const matsize s2, const matsize blockSize) {
    if (s1.w != s2.h) return;

    int I, J, K;
    #pragma omp parallel for private (I, J, K)
    for (I = 0; I < s1.h; I += blockSize.h) {
        for (J = 0; J < s2.w; J += blockSize.w) {
            float *submr = rm + I * s2.w + J;
            for (K = 0; K < s1.w; K += blockSize.w) {
                const float *subm1 = m1 + I * s1.w + K;
                const float *subm2 = m2 + K * s2.w + J;

                int i, j, k;
                for (i = 0; i < blockSize.h; i++) {
                    for (j = 0; j < blockSize.w; j++) {
                        float acc = 0.0;
                        for (k = 0; k < blockSize.w; k++) {
                            acc += subm1[s1.w * i + k] * subm2[j + s2.w * k];
                        }
                        submr[s2.w * i + j] += acc;
                    }
                }
            }
        }
    }
}

int main()
{
    setlocale(LC_ALL, "Russian");

    // init matrix
    matsize s1 = { 1024, 1024 },
        s2 = { 1024, 1024 },
        s3 = { s2.w, s1.h };
    matsize blockSize = { 4, 4 };
    float *mat1 = new float[s1.w * s1.h];
    float *mat2 = new float[s2.w * s2.h];
    float *matres1 = new float[s3.w * s3.h];
    float *matres2 = new float[s3.w * s3.h];

    for (int i = 0; i < s1.w * s1.h; i++) {
        mat1[i] = 2;
    }
    for (int i = 0; i < s2.w * s2.h; i++) {
        mat2[i] = 2;
    }

    double t = omp_get_wtime();
    matmul_parallel_omp(mat1, mat2, matres1, s1, s2);
    t = omp_get_wtime() - t;
    printf("matmul_parallel_omp exec time = %lg\n", t);

    memset(matres2, 0, s2.w * s1.h * sizeof(float));
    t = omp_get_wtime();
    matmul_blocks_parallel_omp(mat1, mat2, matres2, s1, s2, blockSize);
    t = omp_get_wtime() - t;
    printf("matmul_blocks_parallel_omp exec time = %lg\n", t);
    int c = equalsMat(matres1, matres2, s3, s3);
    printf("equals = %i\n", c);
}