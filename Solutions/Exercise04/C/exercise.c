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
    int       err;
    float*    h_a = (float *)calloc(LENGTH, sizeof(float));
    float*    h_b = (float *)calloc(LENGTH, sizeof(float));
    float*    h_c = (float *)calloc(LENGTH, sizeof(float));
    float*    h_d = (float *)calloc(LENGTH, sizeof(float));
    float*    h_e = (float *)calloc(LENGTH, sizeof(float));
    float*    h_f = (float *)calloc(LENGTH, sizeof(float));
    float*    h_g = (float *)calloc(LENGTH, sizeof(float));

    unsigned int correct;
    size_t global;

    cl_device_id           device_id;
    cl_context             context;
    cl_command_queue       commands;
    cl_program             program;
    cl_kernel              ko_vadd;

    cl_mem                 d_a;
    cl_mem                 d_b;
    cl_mem                 d_c;
    cl_mem                 d_d;
    cl_mem                 d_e;
    cl_mem                 d_f;
    cl_mem                 d_g;

    int i = 0;
    int count = LENGTH;
    for(i = 0; i < count; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
        h_e[i] = rand() / (float)RAND_MAX;
        h_g[i] = rand() / (float)RAND_MAX;
    }

    cl_uint   numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if(numPlatforms == 0){
        printf("Found 0 platforms\n");
        return EXIT_SUCCESS;
    }

    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    for(i = 0; i < numPlatforms; i++){
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if( err==CL_SUCCESS){
            break;
        }
    }

    if(device_id==NULL){
        checkError(err, "Finding a device");
    }

    err = output_device_info(device_id);
    checkError(err, "Printing device output");

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(context, "Creating context");

    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
    checkError(err, "Creating program");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if( err != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_SUCCESS;
    }

    ko_vadd = clCreateKernel(program, "vadd", &err);
    checkError(err, "Creating Kernel");

    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, h_a, &err);
    checkError(err, "Creating buffer d_a");

    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, h_b, &err);
    checkError(err, "Creating buffer d_b");

    d_e = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, h_e, &err);
    checkError(err, "Creating buffer d_e");

    d_g = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, h_g, &err);
    checkError(err, "Creating buffer d_g");

    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, h_c, &err);
    d_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, h_d, &err);
    d_f = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, h_f, &err);

    err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments");

    global = count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_e);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_d);
    err |= clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);

    err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_g);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_d);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_f);
    err |= clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);

    err = clEnqueueReadBuffer(commands, d_f, CL_TRUE, 0, sizeof(float) * count, h_f, 0, NULL, NULL);
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }

    correct = 0;
    float tmp;
    for(i = 0; i < count; i++){
        tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i];
        tmp -= h_f[i];
        if(tmp*tmp < TOL*TOL){
            correct++;
        }
        else{
            printf(" tmp %f h_a %f h_b %f h_c %f \n",tmp, h_a[i], h_b[i], h_c[i]);
        }
    }
    printf("C = A+B:  %d out of %d results were correct.\n", correct, count);

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseMemObject(d_d);
    clReleaseMemObject(d_e);
    clReleaseMemObject(d_f);
    clReleaseMemObject(d_g);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_e);
    free(h_f);
    free(h_g);
    return 0;
}