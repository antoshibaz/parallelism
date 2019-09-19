#include "pch.h"
#include <iostream>

#include <mpi.h>

struct matsize {
    int w;
    int h;
};

void matmul_blocks(const float *m1, const float *m2, float *rm,
    const matsize s1, const matsize s2, const matsize blockSize) {
    if (s1.w != s2.h) return;

    int I, J, K;
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

int equalsMat(const float *m1, const float*m2,
    const matsize s1, const matsize s2) {
    if (s1.w != s2.w || s1.h != s2.h) return -1;
    int c = 0;
    for (int i = 0; i < s1.w * s1.h; i++) {
        if (m1[i] != m2[i]) c++;
    }

    return c;
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "Russian");

    int procNum, procId;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);

    std::cout << "Number of processes: " << procNum << ", process id: " << procId << std::endl;

    // init matrix variables
    matsize s1 = { 1024, 1024 },
        s2 = { 1024, 1024 },
        s3 = { s2.w, s1.h };
    matsize blockSize = { 4, 4 };
    float *mat1 = new float[s1.w * s1.h];
    float *mat2 = new float[s2.w * s2.h];
    float *matres = new float[s3.w * s3.h];
    float *matres2 = new float[s3.w * s3.h];

    int mat1PartH = s1.h / procNum;
    matsize partMat1Size = { s1.w, mat1PartH };
    float *partMat1 = new float[partMat1Size.w * partMat1Size.h];
    matsize partMatresSize = { s2.w, partMat1Size.h };
    float *partMatres = new float[partMatresSize.w * partMatresSize.h];

    if (procId == 0)
    {
        // init matrix values
        for (int i = 0; i < s1.w * s1.h; i++) {
            mat1[i] = 2;
        }
        for (int i = 0; i < s2.w * s2.h; i++) {
            mat2[i] = 2;
        }
    }

    double t = MPI_Wtime();
    
    // send all parts of mat1 to all processes
    MPI_Scatter(mat1, partMat1Size.w * partMat1Size.h, MPI_FLOAT, 
        partMat1, partMat1Size.w * partMat1Size.h, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // send mat2 to all processes
    MPI_Bcast(mat2, s2.w * s2.h, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // matmul operation
    matmul_blocks(partMat1, mat2, partMatres, partMat1Size, s2, blockSize);

    // agregate results of matmul operation from all processes to root process
    MPI_Gather(partMatres, partMat1Size.w * partMat1Size.h, MPI_FLOAT, 
        matres, partMat1Size.w * partMat1Size.h, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    t = MPI_Wtime() - t;

    if (procId == 0)
    {
        printf("\n");
        printf("matmul MPI exec time = %lg\n", t);
        t = MPI_Wtime();
        matmul_blocks(mat1, mat2, matres2, s1, s2, blockSize);
        t = MPI_Wtime() - t;
        printf("matmul exec time = %lg\n", t);
        int c = equalsMat(matres, matres2, s3, s3);
        printf("equals = %i\n", c);
    }

    MPI_Finalize();
}
