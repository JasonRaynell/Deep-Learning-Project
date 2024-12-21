//
//  main.c
//  MM_Thread
//
//  Created by Jason Raynell on 15/12/24.
//

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    int size;
    float **A;
    float **B;
    float **C;
} ThreadData;


// Create a 2D array
float** allocate_matrix(int rows, int cols) {
    float **matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
    }
    return matrix;
}

// Free the allocated memory
void free_matrix(float **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Naive Matrix Multiplication
void* multiply_matrices(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->size; j++) {
            data->C[i][j] = 0;
            for (int k = 0; k < data->size; k++) {
                data->C[i][j] += data->A[i][k] * data->B[k][j];
            }
        }
    }

    return NULL;
}

// Loop Unrolling
void* multiply_unrolling(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->size; j++) {
            float sum = 0.0f;
            int k;
            // Unroll by 4
            for (k = 0; k <= data->size - 4; k += 4) {
                sum += data->A[i][k] * data->B[k][j] +
                data-> A[i][k + 1] * data->B[k + 1][j] +
                data-> A[i][k + 2] * data->B[k + 2][j] +
                data-> A[i][k + 3] * data->B[k + 3][j];
            }
            // Handle remaining
            for (; k < data->size; k++) {
                sum += data-> A[i][k] * data->B[k][j];
            }
            data -> C[i][j] = sum;
        }
    }

    return NULL;
}



int main(void) {
    int size = 0;
    int num_threads = 0;
    double timeTaken = 0;
    struct timespec start,end;
    
    while(size <= 0 || num_threads <= 0 || num_threads > 10){
        
        // Get the size of the matrix and the number of threads from the user
        printf("Enter the size of the matrix (N x N): ");
        scanf("%d", &size);
        
        printf("Enter the number of threads: ");
        scanf("%d", &num_threads);
        
        if (size <= 0 || num_threads <= 0) {
            printf("Size and number of threads must be positive.\n");
        }
        if(num_threads > 10){
            printf("The maximum thread is 10.\n");
        }
    }

    // Allocate matrices
    float **A = allocate_matrix(size, size);
    float **B = allocate_matrix(size, size);
    float **C = allocate_matrix(size, size);

    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i][j] = (float)(rand() % 10 + 1); // Random value between 1 and 10
            B[i][j] = (float)(rand() % 10 + 1);
        }
    }

    // Create threads
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    int rows_per_thread = size / num_threads;
    int extra_rows = size % num_threads; // Handle remainder rows

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].thread_id = t;
        thread_data[t].size = size;
        thread_data[t].A = A;
        thread_data[t].B = B;
        thread_data[t].C = C;

        thread_data[t].start_row = t * rows_per_thread;
        thread_data[t].end_row = (t + 1) * rows_per_thread;

        // Distribute extra rows to the last thread
        if (t == num_threads - 1) {
            thread_data[t].end_row += extra_rows;
        }

        if (pthread_create(&threads[t], NULL, multiply_unrolling, &thread_data[t]) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }

    // Join threads
    for (int t = 0; t < num_threads; t++) {
        if (pthread_join(threads[t], NULL) != 0) {
            perror("Failed to join thread");
            return 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    timeTaken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time taken is %0.fms\n",timeTaken * 1000);
//     Print result matrix
//    printf("Result matrix:\n");
//    for (int i = 0; i < size; i++) {
//        for (int j = 0; j < size; j++) {
//            printf("%.2f ", C[i][j]);
//        }
//        printf("\n");
//    }

    // Free allocated memory
    free_matrix(A, size);
    free_matrix(B, size);
    free_matrix(C, size);

    return 0;
}
