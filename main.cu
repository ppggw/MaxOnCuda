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
#define NUMBER_OF_POINTS 10


typedef struct CustomPoints_
{
    int value;
    int x;
    int y;
} CustomPoint;


__device__ CustomPoint cache_for_vert_array[GRID_SIZE_Y];
__device__ CustomPoint cached_block_result[GRID_SIZE_Y * GRID_SIZE_X];


__global__ void GetMaxGrid(unsigned char* image, int rows, int cols, int offset_X, int offset_Y){
    int x, y;
    if(offset_Y == 0 && offset_X == 0){
        x = blockIdx.x * blockDim.x + threadIdx.x;
        y = blockIdx.y * blockDim.y + threadIdx.y;
    }
    else{
        x = offset_X * blockDim.x + threadIdx.x;
        y = offset_Y * blockDim.y + threadIdx.y;
    }

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
                if(fourthVert[threadIdx.y].value <= 255){
                    if(offset_X == 0 && offset_Y == 0){
                        cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X] = fourthVert[threadIdx.y];
                    }
                    else{
                        cached_block_result[offset_X + offset_Y * GRID_SIZE_X] = fourthVert[threadIdx.y];
                    }
                    __threadfence();
                }
                else{
                    cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].x = 0;
                    cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].y = 0;
                    cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].value = 0;
                }
            }
            else{
                if(fourthVert[threadIdx.y + 1].value <= 255){
                    if(offset_X == 0 && offset_Y == 0){
                        cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X] = fourthVert[threadIdx.y + 1];
                    }
                    else{
                        cached_block_result[offset_X + offset_Y * GRID_SIZE_X] = fourthVert[threadIdx.y + 1];
                    }
                    __threadfence();
                }
                else{
                    cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].x = 0;
                    cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].y = 0;
                    cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].value = 0;
                }
            }      
//            printf("val %d x %d y %d\n", cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].value,
//                    cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].x,
//                    cached_block_result[blockIdx.x + blockIdx.y * GRID_SIZE_X].y);
        }
    }
}


__global__ void FindVertMaxAll(unsigned char* image, CustomPoint* result, int cols){
    CustomPoint max = cached_block_result[threadIdx.y * GRID_SIZE_X];
    for(int i = 1; i != GRID_SIZE_X; i++){
        if(cached_block_result[i + threadIdx.y * GRID_SIZE_X].value > max.value){
            max = cached_block_result[i + threadIdx.y * GRID_SIZE_X];
        }
    }
    cache_for_vert_array[threadIdx.y] = max;
    __syncthreads();

    if(threadIdx.y == 0){
        CustomPoint maxI = cache_for_vert_array[0];
        int coor_max_for_delete = 0;
        for(int i = 1; i != GRID_SIZE_Y; i++){
            if(cache_for_vert_array[i].value > maxI.value){
                maxI = cache_for_vert_array[i];
                coor_max_for_delete = i;
            }
        }
        *result = maxI;
        image[maxI.y * cols + maxI.x] = 0;
    }
}


__global__ void FindVertMaxOne(unsigned char* image, CustomPoint* result, int coorForReFind, int cols){
    CustomPoint max = cached_block_result[coorForReFind * GRID_SIZE_X];
    for(int i = 1; i != GRID_SIZE_X; i++){
        if(cached_block_result[i + coorForReFind * GRID_SIZE_X].value > max.value){
            max = cached_block_result[i + coorForReFind * GRID_SIZE_X];
        }
//        printf("%d\n", cached_block_result[i + coorForReFind * GRID_SIZE_X]);
    }
    cache_for_vert_array[coorForReFind] = max;

    CustomPoint maxI = cache_for_vert_array[0];
    for(int i = 1; i != GRID_SIZE_Y; i++){
        if(cache_for_vert_array[i].value > maxI.value){
            maxI = cache_for_vert_array[i];
        }
    }
    *result = maxI;
    image[maxI.y * cols + maxI.x] = 0;
}


int main()
{
    cv::Mat ImageData = cv::imread("E:/Qt/Projects/MaxOnCuda/for_find_max.jpg");
    cv::circle(ImageData, cv::Point(500, 500), 0, cv::Scalar(255,255,255));
    cv::circle(ImageData, cv::Point(501, 501), 0, cv::Scalar(254,254,254));
    cv::circle(ImageData, cv::Point(499, 500), 0, cv::Scalar(254,254,254));
    cv::circle(ImageData, cv::Point(209, 100), 0, cv::Scalar(252,252,252));
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

    std::array<CustomPoint, NUMBER_OF_POINTS> p;
    for(int i=0; i!= NUMBER_OF_POINTS; i++){
        if(i == 0){
            GetMaxGrid<<<gridSize, blockSize>>>(
                        dev_Image,
                        ImageData.rows,
                        ImageData.cols,
                        0,
                        0
                    );
            cudaDeviceSynchronize();

            CustomPoint* dev_result;
            cudaMalloc((void**)&dev_result, sizeof(CustomPoint));

            FindVertMaxAll<<<dim3(1,1), dim3(1,GRID_SIZE_Y)>>>(
                          dev_Image,
                          dev_result,
                          ImageData.cols
                          );

            CustomPoint* result = (CustomPoint*)malloc(sizeof(CustomPoint));
            cudaMemcpy(result, dev_result, sizeof(CustomPoint), cudaMemcpyDeviceToHost);

            p[i] = *result;

            cudaFree(dev_result);
            free(result);
        }
        else{
            //1 найти координаты блока, к которому относится прошлый макс
            //2 найти следующий максимум в этом блоке
            //3 закрасить его, чтобы на следующей итерации не учавствовал(проблема)
            //3 найти новый максимум в строке(но наверное лучше в столбце) глобальной сетки, обновить кэш
            //4 найти новый максимум по новому столбцу

            int blockIdx_x = p[i-1].x / THREAD_DIM;
            int blockIdx_y = p[i-1].y / THREAD_DIM;

            GetMaxGrid<<<gridSize, blockSize>>>(
                        dev_Image,
                        ImageData.rows,
                        ImageData.cols,
                        blockIdx_x,
                        blockIdx_y
                    );
            cudaDeviceSynchronize();

            CustomPoint* dev_result;
            cudaMalloc((void**)&dev_result, sizeof(CustomPoint));

            FindVertMaxOne<<<1, 1>>>(
                          dev_Image,
                          dev_result,
                          blockIdx_y,
                          ImageData.cols
                          );

            CustomPoint* result = (CustomPoint*)malloc(sizeof(CustomPoint));
            cudaMemcpy(result, dev_result, sizeof(CustomPoint), cudaMemcpyDeviceToHost);

            p[i] = *result;

            cudaFree(dev_result);
            free(result);
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    for(int i=0; i!= NUMBER_OF_POINTS; i++){
        std::cout << "result is: value = " << p[i].value << " x = " << p[i].x << "  y = " << p[i].y << "\n";

        cv::Point max{p[i].x, p[i].y};
        cv::circle(ImageData, max, 3, cv::Scalar(0,0,255));
    }

    cv::imshow("f", ImageData);
    cv::waitKey(0);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time of CUDA Work: %3.1f ms\n", elapsedTime);

    cudaEventDestroy( start );
    cudaEventDestroy( stop  );

    cudaFree(dev_Image);
}









