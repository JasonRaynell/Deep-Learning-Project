//
//  main.c
//  DeepLearningProject
//
//  Created by Jason Raynell on 10/12/24.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <arm_neon.h>

typedef struct {
    int rows;
    int cols;
    float **array;
} Matrix;

clock_t start, end; // Global Variable
 
//  Create a 2D array
Matrix createMatrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
 
    // Allocate memory for the array of pointers (rows)
    m.array = (float **)malloc(rows * sizeof(float *));
    // Allocate memory for each row
    for (int i = 0; i < rows; i++) {
        m.array[i] = (float *)malloc(cols * sizeof(float));
    }
 
    return m;
}
 
//  Free the allocated memory
void freeMatrix(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->array[i]); // Free each row
    }
    free(m->array); // Free the array of pointers
}

//  General Matrix Multiplication Operation
void GEMM(int matrixsize, float *a[matrixsize], float *b[matrixsize], float *c[matrixsize]){
    for(int i=0;i<matrixsize;i++){
        for(int j=0; j<matrixsize;j++){
            for(int k=0;k<matrixsize;k++){
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return;
}

//  Loop Reordering
void LoopReordering(int matrixsize, float *a[matrixsize], float *b[matrixsize], float *c[matrixsize]){
    for(int i=0;i<matrixsize;i++){
        for(int k=0; k<matrixsize;k++){
            for(int j=0;j<matrixsize;j++){
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return;
}

//  Loop Tiling
#define TILE 32 // tile size
void LoopTiling(int matrixsize,  float *a[matrixsize], float *b[matrixsize], float *c[matrixsize]){
    int ii,jj,kk = 0;
    
    for (ii = 0; ii < matrixsize; ii += TILE) { // Outer loop
        for (kk = 0; kk < matrixsize; kk += TILE) {
            for (jj = 0; jj < matrixsize; jj += TILE) {
                for (int i = ii; i < ii + TILE && i < matrixsize; i++) { // Inner loop
                    for (int k = kk; k < kk + TILE && k < matrixsize; k++) {
                        for (int j = jj; j < jj + TILE && j < matrixsize; j++) {
                            c[i][j] += a[i][k] * b[k][j];
                            }
                        }
                    }
                }
            }
        }
    return;
}

//  Loop Unrolling
void LoopUnrolling(int matrixsize,  float *a[matrixsize], float *b[matrixsize], float *c[matrixsize]){
    for(int i = 0; i< matrixsize;i++){
        for(int k = 0; k < matrixsize; k++){
            for(int j = 0; j < matrixsize; j+= 4){ // Unroll by 4
                c[i][j] += a[i][k] * b[k][j];
                c[i][j+1] += a[i][k] * b[k][j+1];
                c[i][j+2] += a[i][k] * b[k][j+2];
                c[i][j+3] += a[i][k] * b[k][j+3];
            }
        }
    }
    return;
}

//  Loop Unrolling + Tiled
void LoopUnrolling_Tiled(int matrixsize,  float *a[matrixsize], float *b[matrixsize], float *c[matrixsize]){
    int ii,jj,kk = 0; // Initialize
    
    for (ii = 0; ii < matrixsize; ii += TILE) { // Outer Loop
        for (kk = 0; kk < matrixsize; kk += TILE) {
            for (jj = 0; jj < matrixsize; jj += TILE) {
                for (int i = ii; i < ii + TILE && i < matrixsize; i++) { // Inner Loop
                    for (int k = kk; k < kk + TILE && k < matrixsize; k++) {
                        for (int j = jj; j < jj + TILE && j < matrixsize; j += 4) { // Unroll by 4
                            c[i][j] += a[i][k] * b[k][j];
                            c[i][j+1] += a[i][k] * b[k][j+1];
                            c[i][j+2] += a[i][k] * b[k][j+2];
                            c[i][j+3] += a[i][k] * b[k][j+3];
                            }
                        }
                    }
                }
            }
        }
    return;
}

//  Arm Neon
void Neon(int matrixsize,  float *a[matrixsize], float *b[matrixsize], float *c[matrixsize]) {
    for (int i = 0; i < matrixsize; i++) {
        for (int j = 0; j < matrixsize; j += 4) { // Process 4 elements at a time
            float32x4_t c_vec = vdupq_n_f32(0); // Initialize NEON vector to zero
            for (int k = 0; k < matrixsize; k++) {
                float32x4_t a_vec = vdupq_n_f32(a[i][k]);   // Broadcast a[i][k] to a NEON vector
                float32x4_t b_vec = vld1q_f32(&b[k][j]);    // Load 4 elements of b[k][j]
                c_vec = vmlaq_f32(c_vec, a_vec, b_vec);     // Perform multiply-accumulate
            }
            vst1q_f32(&c[i][j], c_vec); // Store the result back to c[i][j]
        }
    }
}

//  Add Random Value to matrix
void RandomValue(int size, float *a[size], float *b[size]){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[i][j] = (float)(rand() % 10 + 1); // Assign some value between 1.0 - 10.0
            b[i][j] = (float)(rand() % 10 + 1);
        }
    }
    return;
}

//  Print Matrix
void PrintMatrix(int size, float *a[size]){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }
    return;
}


//  Check Previous Matrix operation result
bool CheckMatrix(int size, float *a[size], float *b[size]){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if(a[i][j] != b[i][j]){
                //  Not equal
                printf("When i = %d and j = %d\n", i,j);
                printf("Matrix A value is %f and B value is %f\n",a[i][j],b[i][j]);
//                PrintMatrix(size, a);
//                PrintMatrix(size, b);
                return false;
            }
        }
    }
    return true;
}

//  Reset Matrix C
void Reset(int size, float *a[],float *b[]){
    for(int i = 0; i < size;i++){
        for(int j = 0; j < size; j++){
            b[i][j] = a[i][j];  // Set Matrix E to C
            a[i][j] = 0;    // Set Matrix C to 0
        }
    }
    return;
}

//  Menu
void Menu(int size){
    
    // Get the size of the matrix from the user
    printf("Enter the size of matrix (N x N): ");
    scanf("%d", &size);
    
    // Go to Menu
    if(size >= 2 && size <= 5000 ){
        int menu = 0;
        int exit = 0;
        char ch;
        double timeTaken = 0;
        bool equal;
        
        // Create matrices
        Matrix MatrixA = createMatrix(size, size);
        Matrix MatrixB = createMatrix(size, size);
        Matrix MatrixC = createMatrix(size, size);
        Matrix MatrixE = createMatrix(size, size);
        
        // Initialize matrices A and B with random values
        RandomValue(size, MatrixA.array, MatrixB.array);
        
        while(menu !=9 && exit != 1){
            printf("------------------------\n");
            printf("Welcome to the Menu: \n");
            printf("1: Run GEMM\n");
            printf("2: Run Loop Reordering\n");
            printf("3: Run Loop Tiling\n");
            printf("4: Run Loop Unrolling\n");
            printf("5: Run Loop Unrolling_Tiled\n");
            printf("6: Run Neon\n");
            printf("7: Change Matrix size\n");
            printf("8: Check Previous Matrix Result\n");
            printf("9: Exit\n");
            printf("-------------------------\n");
            scanf("%d",&menu);
            switch (menu) {
                case 1:
                    printf("Calculation in progress...\n");
                    start = clock();
                    GEMM(size, MatrixA.array, MatrixB.array, MatrixC.array);
                    end = clock();
                    timeTaken = ((double) (end - start)) / CLOCKS_PER_SEC;
                    printf("GEMM time taken : %0.fms\n",timeTaken * 1000);
                    printf("Press enter to continue...\n");
                    ch = getchar();
                    scanf("%c",&ch);
                    Reset(size,MatrixC.array,MatrixE.array);
                    break;
                case 2:
                    printf("Calculation in progress...\n");
                    start = clock();
                    LoopReordering(size, MatrixA.array, MatrixB.array, MatrixC.array);
                    end = clock();
                    timeTaken = ((double) (end - start)) / CLOCKS_PER_SEC;
                    printf("Loop Reordering time taken : %0.fms\n",timeTaken * 1000);
                    printf("Press enter to continue...\n");
                    ch = getchar();
                    scanf("%c",&ch);
                    Reset(size,MatrixC.array,MatrixE.array);
                    break;
                case 3:
                    printf("Calculation in progress...\n");
                    start = clock();
                    LoopTiling(size, MatrixA.array, MatrixB.array, MatrixC.array);
                    end = clock();
                    timeTaken = ((double) (end - start)) / CLOCKS_PER_SEC;
                    printf("Loop Tiling time taken : %0.fms\n",timeTaken * 1000);
                    printf("Press enter to continue...\n");
                    ch = getchar();
                    scanf("%c",&ch);
                    Reset(size,MatrixC.array,MatrixE.array);
                    break;
                case 4:
                    printf("Calculation in progress...\n");
                    start = clock();
                    LoopUnrolling(size, MatrixA.array, MatrixB.array, MatrixC.array);
                    end = clock();
                    timeTaken = ((double) (end - start)) / CLOCKS_PER_SEC;
                    printf("Loop Unrolling time taken : %0.fms\n",timeTaken * 1000);
                    printf("Press enter to continue...\n");
                    ch = getchar();
                    scanf("%c",&ch);
                    Reset(size,MatrixC.array,MatrixE.array);
                    break;
                case 5:
                    printf("Calculation in progress...\n");
                    start = clock();
                    LoopUnrolling_Tiled(size, MatrixA.array, MatrixB.array, MatrixC.array);
                    end = clock();
                    timeTaken = ((double) (end - start)) / CLOCKS_PER_SEC;
                    printf("Loop Unrolling_Tiled time taken : %0.fms\n",timeTaken * 1000);
                    printf("Press enter to continue...\n");
                    ch = getchar();
                    scanf("%c",&ch);
                    Reset(size,MatrixC.array,MatrixE.array);
                    break;
                case 6:
                    printf("Calculation in progress...\n");
                    start = clock();
                    Neon(size, MatrixA.array, MatrixB.array, MatrixC.array);
                    end = clock();
                    timeTaken = ((double) (end - start)) / CLOCKS_PER_SEC;
                    printf("Neon time taken : %0.fms\n",timeTaken * 1000);
                    printf("Press enter to continue...\n");
                    ch = getchar();
                    scanf("%c",&ch);
                    Reset(size,MatrixC.array,MatrixE.array);
                    break;
                case 7:
                    // Free the allocated memory
                    freeMatrix(&MatrixA);
                    freeMatrix(&MatrixB);
                    freeMatrix(&MatrixC);
                    freeMatrix(&MatrixE);
                    exit = 1;
                    
                    // Return to first step
                    Menu(0);
                    break;
                case 8:
                    if(MatrixE.array[0][0] != 0){
                        Matrix MatrixD = createMatrix(size, size);
                        GEMM(size, MatrixA.array, MatrixB.array, MatrixD.array);
                        equal = CheckMatrix(size, MatrixE.array, MatrixD.array);
                        printf("This matrix is %s\n",equal ? "True" : "False");
                        freeMatrix(&MatrixD);
                    }
                    else{
                        printf("There is no matrix to check!\n");
                    }
                    printf("Press enter to continue...\n");
                    ch = getchar();
                    scanf("%c",&ch);
                    break;
                case 9:
                    // Free the allocated memory
                    freeMatrix(&MatrixA);
                    freeMatrix(&MatrixB);
                    freeMatrix(&MatrixC);
                    freeMatrix(&MatrixE);
                    exit = 1;
                    break;
                default:
                    break;
            }
        }
    }
    // Repeat until the size matrix is in between 2-5000.
    else{
        printf("Please input the size matrix in between 2~5000\n");
        Menu(0);
    }
}


 
int main(void) {
    int size = 0;
    srand(time(NULL)); //   Initialize random number generator
    
    Menu(size);

    return 0;
}
