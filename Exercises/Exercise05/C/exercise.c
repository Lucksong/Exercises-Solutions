#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern int output_device_info(cl_device_id);

#define TOL    (0.001)
#define LENGTH (1024000)

const char* KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   __global float* d,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       d[i] = a[i] + b[i] + c[i];                                             \n" \
"}                                                                      \n" \
"\n";

int main(int argc, char** argv){
    cl_int                     err;
    size_t dataSize = sizeof(float) * LENGTH;
    float*      h_a = (float*)malloc(dataSize);
    float*      h_b = (float*)malloc(dataSize);
    float*      h_c = (float*)malloc(dataSize);
    float*      h_d = (float*)malloc(dataSize);
    unsigned int correct;

    size_t global;

    cl_device_id            device_id;
    cl_context              context;
    cl_command_queue        commands;
    cl_program              program;
    cl_kernel               ko_vadd;

    cl_mem      d_a;
    cl_mem      d_b;
    cl_mem      d_c;
    cl_mem      d_d;

    int i=0;
    for(i=0;i<LENGTH;i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
        h_c[i] = rand() / (float)RAND_MAX;
    }

    cl_uint   numPlatforms;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if(numPlatforms==0){
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    for(i=0;i<numPlatforms;i++){
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if(err==CL_SUCCESS){
            break;
        }
    }

    if(device_id==NULL){
        checkError(err, "Getting device");
    }

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    ko_vadd = clCreateKernel(program, "vadd", &err);

    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, dataSize, h_a, &err);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, dataSize, h_b, &err);
    d_c = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, dataSize, h_c, &err);

    d_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &err);

    const int count = LENGTH;

    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(ko_vadd, 3, sizeof(cl_mem), &d_d);
    err |= clSetKernelArg(ko_vadd, 4, sizeof(unsigned int), &count);

    global = count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    err = clEnqueueReadBuffer(commands, d_d, CL_TRUE, 0, sizeof(float)*count, h_d, 0, NULL, NULL);

    correct = 0;
    float tmp;
    for(i=0;i<count;i++){
        tmp = h_a[i] + h_b[i] + h_c[i];
        tmp -= h_d[i];
        if(tmp*tmp < TOL*TOL){
            correct++;
        }
        else{
            printf(" tmp %f h_a %f h_b %f h_c %f h_d %f\n",tmp, h_a[i], h_b[i], h_c[i], h_d[i]);
        }
    }
    printf("D = A+B+C:  %d out of %d results were correct.\n", correct, count);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseMemObject(d_d);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
}