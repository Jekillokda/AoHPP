#include <stdio.h> 
#include <time.h> 
#include <math.h> 
#include <stdlib.h> 
#include <Windows.h> 
#include "emmintrin.h" 
#include "immintrin.h" 

#define SIZE 800 
#define SIZE_2 12

void main() {
	double ****Matrix1, ****Matrix2, ****resultMatrix1, ****resultMatrix2;

	Matrix1 = (double****)malloc(SIZE * sizeof(double***));
	for (int i = 0; i < SIZE; i++) {
		Matrix1[i] = (double***)malloc(SIZE * sizeof(double**));
		for (int j = 0; j < SIZE; j++) {
			Matrix1[i][j] = (double**)_aligned_malloc(SIZE_2 * sizeof(double*), 16);
			for (int k = 0; k < SIZE_2; k++) {
				Matrix1[i][j][k] = (double*)_aligned_malloc(SIZE_2 * sizeof(double), 16);
			}
		}
	}

	Matrix2 = (double****)malloc(SIZE * sizeof(double***));
	for (int i = 0; i < SIZE; i++) {
		Matrix2[i] = (double***)malloc(SIZE * sizeof(double**));
		for (int j = 0; j < SIZE; j++) {
			Matrix2[i][j] = (double**)_aligned_malloc(SIZE_2 * sizeof(double*), 16);
			for (int k = 0; k < SIZE_2; k++) {
				Matrix2[i][j][k] = (double*)_aligned_malloc(SIZE_2 * sizeof(double), 16);
			}
		}
	}

	resultMatrix1 = (double****)malloc(SIZE * sizeof(double***));
	for (int i = 0; i < SIZE; i++) {
		resultMatrix1[i] = (double***)malloc(SIZE * sizeof(double**));
		for (int j = 0; j < SIZE; j++) {
			resultMatrix1[i][j] = (double**)_aligned_malloc(SIZE_2 * sizeof(double*), 16);
			for (int k = 0; k < SIZE_2; k++) {
				resultMatrix1[i][j][k] = (double*)_aligned_malloc(SIZE_2 * sizeof(double), 16);
			}
		}
	}

	resultMatrix2 = (double****)malloc(SIZE * sizeof(double***));
	for (int i = 0; i < SIZE; i++) {
		resultMatrix2[i] = (double***)malloc(SIZE * sizeof(double**));
		for (int j = 0; j < SIZE; j++) {
			resultMatrix2[i][j] = (double**)_aligned_malloc(SIZE_2 * sizeof(double*), 16);
			for (int k = 0; k < SIZE_2; k++) {
				resultMatrix2[i][j][k] = (double*)_aligned_malloc(SIZE_2 * sizeof(double), 16);
			}
		}
	}

	srand(time(NULL));

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			for (int k = 0; k < SIZE_2; k++) {
				for (int m = 0; m < SIZE_2; m++) {
					Matrix1[i][j][k][m] = rand() % 100;
					Matrix2[i][j][k][m] = rand() % 100;
					resultMatrix1[i][j][k][m] = 0;
					resultMatrix2[i][j][k][m] = 0;
				}
			}
		}
	}
	/*for (int m = 0; m < SIZE; m++) {
		for (int n = 0; n < SIZE; n++) {
			for (int i = 0; i < SIZE_2; i++)
			{
				for (int j = 0; j < SIZE_2; j++)
				{
					printf("%lf ", Matrix1[m][n][i][j]);
				}
				printf("\n");
			}
		}
	}*/

	//printf("\n");

	/*for (int m = 0; m < SIZE; m++) {
		for (int n = 0; n < SIZE; n++) {
			for (int i = 0; i < SIZE_2; i++)
			{
				for (int j = 0; j < SIZE_2; j++)
				{
					printf("%lf ", Matrix2[m][n][i][j]);
				}
				printf("\n");
			}
		}
	}*/

	clock_t start, end;

	start = clock();
	for (int m = 0; m < SIZE; ++m) {

		for (int n = 0; n < SIZE; ++n) {

			for (int i = 0; i < SIZE_2; ++i) {

				double *temp = resultMatrix1[m][n][i];

				for (int j = 0; j < SIZE_2; ++j) {
					double temp1 = Matrix1[m][n][i][j];
					double *temp2 = Matrix2[m][n][j];
					#pragma loop(no_vector)
					for (int k = 0; k < SIZE_2; ++k) {
						temp[k] += temp1 * temp2[k];
					}
				}
			}
		}
	}
	end = clock();
	printf("time c %f\n", (float)(end - start) / CLK_TCK);
	/*for (int m = 0; m < SIZE; m++) {
		for (int n = 0; n < SIZE; n++) {
			for (int i = 0; i < SIZE_2; i++)
			{
				for (int j = 0; j < SIZE_2; j++)
				{
					printf("%lf ", resultMatrix1[m][n][i][j]);
				}
				printf("\n");
			}
		}
	}*/

	start = clock();

	for (int m = 0; m < SIZE; ++m) {
		for (int n = 0; n < SIZE; ++n) {
			for (int i = 0; i < SIZE_2; ++i) {
				double *temp = resultMatrix2[m][n][i];
				for (int j = 0; j < SIZE_2; ++j) {
					double temp1 = Matrix1[m][n][i][j];
					double *__restrict temp2 = Matrix2[m][n][j];
					for (int k = 0; k < SIZE_2; k += 12) {
						const __m256d buff = { temp1,temp1,temp1,temp1 };
						__m256d res = _mm256_add_pd(_mm256_mul_pd(buff, *reinterpret_cast<__m256d*>(temp2)), *reinterpret_cast<__m256d*>(temp));
						memcpy(temp, &res, sizeof(res));
						__m256d res1 = _mm256_add_pd(_mm256_mul_pd(buff, *reinterpret_cast<__m256d*>(temp2 + 4)), *reinterpret_cast<__m256d*>(temp + 4));
						memcpy(temp + 4, &res1, sizeof(res1));
						__m256d res2 = _mm256_add_pd(_mm256_mul_pd(buff, *reinterpret_cast<__m256d*>(temp2 + 8)), *reinterpret_cast<__m256d*>(temp + 8));
						memcpy(temp + 8, &res2, sizeof(res2));
					}
				}
			}
		}
	}

	end = clock();
	printf("time SSE2 %f\n", (float)(end - start) / CLK_TCK);
	/*for (int m = 0; m < SIZE; m++) {
		for (int n = 0; n < SIZE; n++) {
			for (int i = 0; i < SIZE_2; i++)
			{
				for (int j = 0; j < SIZE_2; j++)
				{
					printf("%lf ", resultMatrix2[m][n][i][j]);
				}
				printf("\n");
			}
		}
	}*/
	for (int m = 0; m < SIZE; m++) {
		for (int n = 0; n < SIZE; n++) {
			for (int i = 0; i < SIZE_2; i++)
			{
				for (int j = 0; j < SIZE_2; j++)
				{
					if (resultMatrix1[m][n][i][j] != resultMatrix2[m][n][i][j]) {
						printf("not equal");
						printf("\n");
					}
				}
			}
		}
	}
	system("pause");
}
