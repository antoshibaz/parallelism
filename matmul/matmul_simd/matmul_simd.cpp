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

void mattrans_4x4_simd(const float *m, float *mr) {
    /*
        example
        [0  1  2  3 ]
        [4  5  6  7 ]
        [8  9  10 11]
        [12 13 14 15]
    */
    _asm {
        mov eax, m
        movups xmm0, [eax]
        movups xmm1, [eax + 16]
        movups xmm2, [eax + 32]
        movups xmm3, [eax + 48]

        movups xmm4, xmm0
        shufps xmm0, xmm1, 044h; 5 4 0 1
        shufps xmm4, xmm1, 0EEh; 7 6 3 2

        movups xmm5, xmm2
        shufps xmm2, xmm3, 044h; 13 12 9 8
        shufps xmm5, xmm3, 0EEh; 15 14 11 10

        movups xmm1, xmm0
        shufps xmm0, xmm2, 088h; 12 8 4 0
        shufps xmm1, xmm2, 0DDh; 13 9 5 1

        movups xmm2, xmm4
        shufps xmm2, xmm5, 088h; 14 10 6 2
        shufps xmm4, xmm5, 0DDh; 15 11 7 3

        mov ebx, mr
        movups[ebx], xmm0
        movups[ebx + 16], xmm1
        movups[ebx + 32], xmm2
        movups[ebx + 48], xmm4
    }
}

void matmul_4x4_simd(const float *subm1, const float *subm2, float *submr,
    const matsize s1, const matsize s2) {
    _asm {
        mov eax, subm2
        movups xmm0, [eax]
        mov edx, s2.w
        movups xmm1, [eax + 4 * edx]
        add edx, s2.w
        movups xmm2, [eax + 4 * edx]
        add edx, s2.w
        movups xmm3, [eax + 4 * edx]

        ; transpose second matrix
        movups xmm4, xmm0
        shufps xmm0, xmm1, 044h
        shufps xmm4, xmm1, 0EEh

        movups xmm5, xmm2
        shufps xmm2, xmm3, 044h
        shufps xmm5, xmm3, 0EEh

        movups xmm1, xmm0
        shufps xmm0, xmm2, 088h
        shufps xmm1, xmm2, 0DDh

        movups xmm2, xmm4
        shufps xmm2, xmm5, 088h
        shufps xmm4, xmm5, 0DDh

        mov eax, subm1
        mov ebx, submr
        mov ecx, 4; rows
        ; matrix multiplication
        ; multiplying every row of first matrix on all rows of second matrix
        ; rows of second matrix - xmm0, xmm1, xmm2, xmm4
        l0 :
        movups xmm3, [eax]

            ; step 1
            movups xmm5, xmm3
            movups xmm6, xmm3

            mulps xmm5, xmm0
            mulps xmm6, xmm1

            movups xmm7, xmm5
            shufps xmm5, xmm6, 044h
            shufps xmm7, xmm6, 0EEh

            addps xmm5, xmm7

            ; step 2
            movups xmm6, xmm3
            mulps xmm3, xmm2
            mulps xmm6, xmm4

            movups xmm7, xmm3
            shufps xmm3, xmm6, 044h
            shufps xmm7, xmm6, 0EEh

            addps xmm3, xmm7

            ; step 3
            movups xmm6, xmm5
            shufps xmm6, xmm3, 088h
            shufps xmm5, xmm3, 0DDh
            addps xmm6, xmm5

            addps xmm6, [ebx]
            movups[ebx], xmm6

            ; movups[ebx], xmm6
            mov edx, s2.w
            imul edx, 4
            add ebx, edx
            mov edx, s1.w
            imul edx, 4
            add eax, edx

            loop l0
    }
}

void matmul_blocks_4x4_simd(const float *m1, const float *m2, float *rm,
    const matsize s1, const matsize s2) {
    if (s1.w != s2.h) return;

    int I, J, K;
    for (I = 0; I < s1.h; I += 4) {
        for (J = 0; J < s2.w; J += 4) {
            float *submr = rm + I * s2.w + J;
            for (K = 0; K < s1.w; K += 4) {
                const float *subm1 = m1 + I * s1.w + K;
                const float *subm2 = m2 + K * s2.w + J;

                _asm {
                    mov eax, subm2
                    movups xmm0, [eax]
                    mov edx, s2.w
                    movups xmm1, [eax + 4 * edx]
                    add edx, s2.w
                    movups xmm2, [eax + 4 * edx]
                    add edx, s2.w
                    movups xmm3, [eax + 4 * edx]

                    ; transpose second matrix
                    movups xmm4, xmm0
                    shufps xmm0, xmm1, 044h
                    shufps xmm4, xmm1, 0EEh

                    movups xmm5, xmm2
                    shufps xmm2, xmm3, 044h
                    shufps xmm5, xmm3, 0EEh

                    movups xmm1, xmm0
                    shufps xmm0, xmm2, 088h
                    shufps xmm1, xmm2, 0DDh

                    movups xmm2, xmm4
                    shufps xmm2, xmm5, 088h
                    shufps xmm4, xmm5, 0DDh

                    mov eax, subm1
                    mov ebx, submr
                    mov ecx, 4; rows
                    ; matrix multiplication
                    ; multiplying every row of first matrix on all rows of second matrix
                    ; rows of second matrix - xmm0, xmm1, xmm2, xmm4
                    l0 :
                    movups xmm3, [eax]

                        ; step 1
                        movups xmm5, xmm3
                        movups xmm6, xmm3

                        mulps xmm5, xmm0
                        mulps xmm6, xmm1

                        movups xmm7, xmm5
                        shufps xmm5, xmm6, 044h
                        shufps xmm7, xmm6, 0EEh

                        addps xmm5, xmm7

                        ; step 2
                        movups xmm6, xmm3
                        mulps xmm3, xmm2
                        mulps xmm6, xmm4

                        movups xmm7, xmm3
                        shufps xmm3, xmm6, 044h
                        shufps xmm7, xmm6, 0EEh

                        addps xmm3, xmm7

                        ; step 3
                        movups xmm6, xmm5
                        shufps xmm6, xmm3, 088h
                        shufps xmm5, xmm3, 0DDh
                        addps xmm6, xmm5

                        addps xmm6, [ebx]
                        movups[ebx], xmm6

                        ; movups[ebx], xmm6
                        mov edx, s2.w
                        imul edx, 4
                        add ebx, edx
                        mov edx, s1.w
                        imul edx, 4
                        add eax, edx

                        loop l0
                }
            }
        }
    }
}

void matmul_blocks_4x4_simd_parallel_omp(const float *m1, const float *m2, float *rm,
    const matsize s1, const matsize s2) {
    if (s1.w != s2.h) return;

    int I, J, K;
    #pragma omp parallel for private (I, J, K)
    for (I = 0; I < s1.h; I += 4) {
        for (J = 0; J < s2.w; J += 4) {
            float *submr = rm + I * s2.w + J;
            for (K = 0; K < s1.w; K += 4) {
                const float *subm1 = m1 + I * s1.w + K;
                const float *subm2 = m2 + K * s2.w + J;

                matmul_4x4_simd(subm1, subm2, submr, s1, s2);
            }
        }
    }
}

void test_omp() {
    #pragma omp parallel
    {
        printf("This thread id=%d; Total threads=%d\n", omp_get_thread_num(), omp_get_num_threads());
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
    matmul_blocks_4x4_simd(mat1, mat2, matres1, s1, s2);
    t = omp_get_wtime() - t;
    printf("matmul_blocks_4x4_simd exec time = %lg\n", t);

    t = omp_get_wtime();
    matmul_blocks_4x4_simd_parallel_omp(mat1, mat2, matres2, s1, s2);
    t = omp_get_wtime() - t;
    printf("matmul_blocks_4x4_simd_parallel_omp exec time = %lg\n", t);
    int c = equalsMat(matres1, matres2, s3, s3);
    printf("equals = %i\n", c);
}