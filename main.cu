#include <iostream>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"

#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


#define THREAD_DIM 32
#define GRID_SIZE_X 40
#define GRID_SIZE_Y 22

typedef struct CustomPoints_
{
    int value;
    int x;
    int y;
} CustomPoint;


__device__ CustomPoint array_result[GRID_SIZE_X * GRID_SIZE_Y];


__global__ void GetMaxGrid(unsigned char* image, int rows, int cols){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > cols || y > rows){
        return;
    }

    __shared__ CustomPoint first[THREAD_DIM][THREAD_DIM/2];
    __shared__ CustomPoint second[THREAD_DIM][THREAD_DIM/4];
    __shared__ CustomPoint third[THREAD_DIM][THREAD_DIM/8];
    __shared__ CustomPoint fourth[THREAD_DIM][THREAD_DIM/16];

    __shared__ CustomPoint VerticalBlockRes[THREAD_DIM];

    __shared__ CustomPoint firstVert[THREAD_DIM/2];
    __shared__ CustomPoint secondVert[THREAD_DIM/4];
    __shared__ CustomPoint thirdVert[THREAD_DIM/8];
    __shared__ CustomPoint fourthVert[THREAD_DIM/16];

    CustomPoint block_result;

    if(threadIdx.x % 2 == 0){
        if(image[y*cols + x] >= image[y*cols + x + 1]){
            first[threadIdx.y][threadIdx.x / 2].x = x;
            first[threadIdx.y][threadIdx.x / 2].y = y;
            first[threadIdx.y][threadIdx.x / 2].value = (int)image[y*cols + x];
        }
        else{
            CustomPoint p;
            first[threadIdx.y][threadIdx.x / 2].x = x + 1;
            first[threadIdx.y][threadIdx.x / 2].y = y;
            first[threadIdx.y][threadIdx.x / 2].value = (int)image[y*cols + x + 1];
        }
    }
    __syncthreads();
    if(threadIdx.x % 2 == 0  && threadIdx.x < 16){
        if(first[threadIdx.y][threadIdx.x].value >= first[threadIdx.y][ threadIdx.x + 1].value ){
            second[threadIdx.y][threadIdx.x/2] = first[threadIdx.y][threadIdx.x];
        }
        else{
            second[threadIdx.y][threadIdx.x/2] = first[threadIdx.y][threadIdx.x + 1];
        }
    }
    __syncthreads();
    if(threadIdx.x % 2 == 0  && threadIdx.x < 8){
        if(second[threadIdx.y][threadIdx.x].value  >= second[threadIdx.y][ threadIdx.x + 1].value ){
            third[threadIdx.y][threadIdx.x / 2] = second[threadIdx.y][threadIdx.x];
        }
        else{
            third[threadIdx.y][threadIdx.x / 2] = second[threadIdx.y][threadIdx.x + 1];
        }
    }
    __syncthreads();
    if(threadIdx.x % 2 == 0 && threadIdx.x < 4){
        if(third[threadIdx.y][threadIdx.x].value  >= third[threadIdx.y][ threadIdx.x + 1].value ){
            fourth[threadIdx.y][threadIdx.x / 2] = third[threadIdx.y][threadIdx.x];
        }
        else{
            fourth[threadIdx.y][threadIdx.x / 2] = third[threadIdx.y][threadIdx.x + 1];
        }
    }
    __syncthreads();
    if(threadIdx.x % 2 == 0 && threadIdx.x < 2){
        if(fourth[threadIdx.y][threadIdx.x].value  >= fourth[threadIdx.y][ threadIdx.x + 1].value ){
            VerticalBlockRes[threadIdx.y] = fourth[threadIdx.y][threadIdx.x];
        }
        else{
            VerticalBlockRes[threadIdx.y] = fourth[threadIdx.y][threadIdx.x + 1];
        }
    }
    __syncthreads();

    if(threadIdx.x == 0){
        if(threadIdx.y % 2 == 0){
            if(VerticalBlockRes[threadIdx.y].value >= VerticalBlockRes[ threadIdx.y + 1].value ){
                firstVert[threadIdx.y / 2] = VerticalBlockRes[threadIdx.y];
            }
            else{
                firstVert[threadIdx.y / 2] = VerticalBlockRes[threadIdx.y + 1];
            }
        }
        __syncthreads();
        if(threadIdx.y % 2 == 0 && threadIdx.y < 16){
            if(firstVert[threadIdx.y].value  >= firstVert[ threadIdx.y + 1].value ){
                secondVert[threadIdx.y / 2] = firstVert[threadIdx.y];
            }
            else{
                secondVert[threadIdx.y / 2] = firstVert[threadIdx.y + 1];
            }
        }
        __syncthreads();
        if(threadIdx.y % 2 == 0 && threadIdx.y < 8){
            if(secondVert[threadIdx.y].value  >= secondVert[ threadIdx.y + 1].value ){
                thirdVert[threadIdx.y / 2] = secondVert[threadIdx.y];
            }
            else{
                thirdVert[threadIdx.y / 2] = secondVert[threadIdx.y + 1];
            }
        }
        __syncthreads();
        if(threadIdx.y % 2 == 0 && threadIdx.y < 4){
            if(thirdVert[threadIdx.y].value  >= thirdVert[ threadIdx.y + 1].value ){
                fourthVert[threadIdx.y / 2] = thirdVert[threadIdx.y];
            }
            else{
                fourthVert[threadIdx.y / 2] = thirdVert[threadIdx.y + 1];
            }
        }
        __syncthreads();
        // для послдней строки не работает. Где-то выход за пределы скорее всего
        if(threadIdx.y % 2 == 0 && threadIdx.y < 2){
            if(fourthVert[threadIdx.y].value  >= fourthVert[ threadIdx.y + 1].value){
                //result_x[blockIdx.x + blockIdx.y * GRID_SIZE_X] = (int)fourthVert[threadIdx.y].x;
                //result_y[blockIdx.x + blockIdx.y * GRID_SIZE_X] = (int)fourthVert[threadIdx.y].y;
                //result_value[blockIdx.x + blockIdx.y * GRID_SIZE_X] = (int)fourthVert[threadIdx.y].value;

                if(fourthVert[threadIdx.y].value <= 255){
                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X] = fourthVert[threadIdx.y];
                    __threadfence();
                }
                else{
                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].x = 0;
                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].y = 0;
                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].value = 0;
                }
            }
            else{
                if(fourthVert[threadIdx.y + 1].value <= 255){
                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X] = fourthVert[threadIdx.y + 1];
                    __threadfence();
                }
                else{
                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].x = 0;
                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].y = 0;
                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].value = 0;
                }
            }
//            printf("val %d x %d y %d\n", array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].value,
//                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].x,
//                    array_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].y);
        }
    }
}


__global__ void FindVertMax(int* result_x, int* result_y, int* result_value){
    __shared__ CustomPoint VerticalBlockRes[GRID_SIZE_Y];

    CustomPoint max = array_result[threadIdx.y * GRID_SIZE_X];
    for(int i = 1; i != GRID_SIZE_X; i++){
        if(array_result[i + threadIdx.y * GRID_SIZE_X].value > max.value){
            max = array_result[i + threadIdx.y * GRID_SIZE_X];
        }
    }
    VerticalBlockRes[threadIdx.y] = max;
    __syncthreads();

    if(threadIdx.y == 0){
        CustomPoint maxI = VerticalBlockRes[0];
        for(int i = 1; i != GRID_SIZE_Y; i++){
            if(VerticalBlockRes[i].value > maxI.value){
                maxI = VerticalBlockRes[i];
            }
        }
        *result_x = maxI.x;
        *result_y = maxI.y;
        *result_value = maxI.value;
    }
}


int main()
{
    cv::Mat ImageData = cv::imread("E:/Qt/Projects/MaxOnCuda/for_find_max.jpg");
    cv::circle(ImageData, cv::Point(500, 500), 0, cv::Scalar(255,255,255));
    cv::Mat GrayImage;
    cv::cvtColor(ImageData, GrayImage, cv::COLOR_BGR2GRAY);
    GrayImage = GrayImage(cv::Rect(0,0,1280,704));
//    cv::imshow("f", GrayImage);
//    cv::waitKey(0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    unsigned char *dev_Image;
    cudaMalloc((void**)&dev_Image, sizeof(unsigned char) * ImageData.cols * ImageData.rows);
    cudaMemcpy(dev_Image, (unsigned char *)GrayImage.ptr<unsigned char>(0), GrayImage.cols * GrayImage.rows * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 gridSize (ceil(ImageData.cols / (float)THREAD_DIM), ceil(ImageData.rows / (float)THREAD_DIM));
    dim3 blockSize (THREAD_DIM, THREAD_DIM);

    GetMaxGrid<<<gridSize, blockSize>>>(
                dev_Image,
                ImageData.rows,
                ImageData.cols
            );
    cudaDeviceSynchronize();

    int* dev_res_x;
    int* dev_res_y;
    int* dev_res_value;
    cudaMalloc((void**)&dev_res_x, sizeof(int));
    cudaMalloc((void**)&dev_res_y, sizeof(int));
    cudaMalloc((void**)&dev_res_value, sizeof(int));

    FindVertMax<<<dim3(1,1), dim3(1,GRID_SIZE_Y)>>>(
                  dev_res_x,
                  dev_res_y,
                  dev_res_value
                  );

    int* result_x = (int*)malloc(sizeof(int));
    int* result_y = (int*)malloc(sizeof(int));
    int* result_value = (int*)malloc(sizeof(int) * GRID_SIZE_Y);

    cudaMemcpy(result_x, dev_res_x, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_y, dev_res_y, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_value, dev_res_value, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    std::cout << "result is: value = " << *result_value << " x = " << *result_x << "  y = " << *result_y << "\n";

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time of CUDA Work: %3.1f ms\n", elapsedTime);

    cv::Point max{*result_x, *result_y};
    cv::circle(ImageData, max, 3, cv::Scalar(0,0,255));
    cv::imshow("f", ImageData);
    cv::waitKey(0);

    cudaEventDestroy( start );
    cudaEventDestroy( stop  );

    free(result_x);
    free(result_y);
    free(result_value);

    cudaFree(dev_Image);
    cudaFree(dev_res_x);
    cudaFree(dev_res_y);
    cudaFree(dev_res_value);
}









