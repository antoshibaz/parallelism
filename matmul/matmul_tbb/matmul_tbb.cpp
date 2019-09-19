#include "pch.h"
#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>

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

class MatMulTBB1D
{
    const float *m1, *m2;
    float *rm;
    const matsize s1, s2;

public:
    MatMulTBB1D(const float *pm1, const float *pm2, float *prm,
        const matsize ps1, const matsize ps2) :
        m1(pm1), m2(pm2), rm(prm), s1(ps1), s2(ps2) {}

    void operator() (tbb::blocked_range<int> &r) const
    {
        for (int i = r.begin(); i < r.end(); i++) {
            for (int j = 0; j < s2.w; j++) {
                float s = 0.0;
                for (int k = 0; k < s1.w; k++) {
                    s += m1[s1.w * i + k] * m2[j + s2.w * k];
                }
                rm[s2.w * i + j] = s;
            }
        }
    }
};

class MatMulTBB2D
{
    const float *m1, *m2;
    float *rm;
    const matsize s1, s2;

public:
    MatMulTBB2D(const float *pm1, const float *pm2, float *prm,
        const matsize ps1, const matsize ps2) :
        m1(pm1), m2(pm2), rm(prm), s1(ps1), s2(ps2) {}

    void operator() (tbb::blocked_range2d<int> &r) const
    {
        for (int i = r.rows().begin(); i < r.rows().end(); i++) {
            for (int j = r.cols().begin(); j < r.cols().end(); j++) {
                float s = 0.0;
                for (int k = 0; k < s1.w; k++) {
                    s += m1[s1.w * i + k] * m2[j + s2.w * k];
                }
                rm[s2.w * i + j] = s;
            }
        }
    }
};

class MatMulBlocksTBB1D
{
    const float *m1, *m2;
    float *rm;
    const matsize s1, s2, blockSize;

public:
    MatMulBlocksTBB1D(const float *pm1, const float *pm2, float *prm,
        const matsize ps1, const matsize ps2, const matsize pblockSize) :
        m1(pm1), m2(pm2), rm(prm), s1(ps1), s2(ps2), blockSize(pblockSize) {}

    void operator() (tbb::blocked_range<int> &r) const
    {
        int Z, J, K;
        int I = 0;
        for (Z = r.begin(); Z < r.end(); Z++) {
            I = Z * blockSize.h;
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
};

class MatMulBlocksTBB2D
{
    const float *m1, *m2;
    float *rm;
    const matsize s1, s2, blockSize;

public:
    MatMulBlocksTBB2D(const float *pm1, const float *pm2, float *prm,
        const matsize ps1, const matsize ps2, const matsize pblockSize) :
        m1(pm1), m2(pm2), rm(prm), s1(ps1), s2(ps2), blockSize(pblockSize) {}

    void operator() (tbb::blocked_range2d<int> &r) const
    {
        int Z1, Z2, K;
        int I = 0, J = 0;
        for (Z1 = r.rows().begin(); Z1 < r.rows().end(); Z1++) {
            I = Z1 * blockSize.h;
            for (Z2 = r.cols().begin(); Z2 < r.cols().end(); Z2++) {
                J = Z2 * blockSize.w;
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
};

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

    tbb::task_scheduler_init task_scheduler_init;

    tbb::tick_count t = tbb::tick_count::now();
    tbb::parallel_for(tbb::blocked_range<int>(0, s1.h, s1.h / 8), MatMulTBB1D(mat1, mat2, matres1, s1, s2));
    printf("matmul_parallel_tbb1D exec time = %lg\n\n", (tbb::tick_count::now() - t).seconds());

    t = tbb::tick_count::now();
    tbb::parallel_for(tbb::blocked_range2d<int>(0, s1.h, 8, 0, s2.w, 8), MatMulTBB2D(mat1, mat2, matres2, s1, s2));
    printf("matmul_parallel_tbb2D exec time = %lg\n", (tbb::tick_count::now() - t).seconds());
    int c = equalsMat(matres1, matres2, s3, s3);
    printf("equals = %i\n\n", c);

    memset(matres2, 0, s2.w * s1.h * sizeof(float));
    t = tbb::tick_count::now();
    tbb::parallel_for(tbb::blocked_range<int>(0, s1.h / blockSize.h), MatMulBlocksTBB1D(mat1, mat2, matres2, s1, s2, blockSize));
    printf("matmul_blocks_parallel_tbb1D exec time = %lg\n", (tbb::tick_count::now() - t).seconds());
    c = equalsMat(matres1, matres2, s3, s3);
    printf("equals = %i\n\n", c);

    memset(matres2, 0, s2.w * s1.h * sizeof(float));
    t = tbb::tick_count::now();
    tbb::parallel_for(tbb::blocked_range2d<int>(0, s1.h / blockSize.h, 1, 0, s1.w / blockSize.w, 1),
        MatMulBlocksTBB2D(mat1, mat2, matres2, s1, s2, blockSize));
    printf("matmul_blocks_parallel_tbb2D exec time = %lg\n", (tbb::tick_count::now() - t).seconds());
    c = equalsMat(matres1, matres2, s3, s3);
    printf("equals = %i\n", c);

    task_scheduler_init.terminate();
}