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

extern double wtime();
extern int output_device_info(cl_device_id);

#define TOL (0.001)
#define LENGTH (1024)

const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";

int main(int argc, char** argv){
    int             err;
    float*          h_a = (float*)calloc(LENGTH, sizeof(float));
    float*          h_b = (float*)calloc(LENGTH, sizeof(float));
    float*          h_c = (float*)calloc(LENGTH, sizeof(float));

    unsigned int correct;
    size_t global;

    cl_device_id     device_id;
    cl_context       context;
    cl_command_queue commands;
    cl_program       program;
    cl_kernel        ko_vadd;

    cl_mem           d_a;
    cl_mem           d_b;
    cl_mem           d_c;

    int i=0;
    int count = LENGTH;
    for(int i=0;i<count;i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    cl_uint numPlatforms;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if(numPlatforms == 0){
        printf("Found 0 platforms\n");
        return EXIT_FAILURE;
    }

    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    for(i = 0; i < num)

    return 0;
}