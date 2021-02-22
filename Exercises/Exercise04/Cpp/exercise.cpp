#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "util.hpp"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif
#include "err_code.h"

#define TOL (0.001)
#define LENGTH (10240000)

int main(void){
    std::vector<float> h_a(LENGTH);
    std::vector<float> h_b(LENGTH);
    std::vector<float> h_c(LENGTH, 0xdeadbeef);
    std::vector<float> h_d(LENGTH, 0xdeadbeef);
    std::vector<float> h_e(LENGTH);
    std::vector<float> h_f(LENGTH, 0xdeadbeef);
    std::vector<float> h_g(LENGTH);

    cl::Buffer d_a;
    cl::Buffer d_b;
    cl::Buffer d_c;
    cl::Buffer d_d;
    cl::Buffer d_e;
    cl::Buffer d_f;
    cl::Buffer d_g;

    int count = LENGTH;
    for(int i=0;i<count;i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
        h_e[i] = rand() / (float)RAND_MAX;
        h_g[i] = rand() / (float)RAND_MAX;
    }

    try{
        cl::Context context(DEVICE);
        cl::Program program(context, util::loadProgram("exercise.cl"), true);
        cl::CommandQueue queue(context);
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");

        d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);
        d_b = cl::Buffer(context, h_b.begin(), h_b.end(), true);
        d_e = cl::Buffer(context, h_e.begin(), h_e.end(), true);
        d_g = cl::Buffer(context, h_g.begin(), h_g.end(), true);

        d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        d_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        d_f = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)
            ),
            d_a,
            d_b,
            d_c,
            count
        );

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)
            ),
            d_e,
            d_c,
            d_d,
            count
        );

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)
            ),
            d_g,
            d_d,
            d_f,
            count
        );
        cl::copy(queue, d_f, h_f.begin(), h_f.end());

        int correct = 0;
        float tmp;
        for(int i=0;i<count;i++){
            tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i];
            tmp -= h_f[i];
            if(tmp*tmp < TOL*TOL) correct++;
            else{
                printf(" tmp %f h_a %f h_b %f h_e %f h_g %f h_f %f\n",tmp, h_a[i], h_b[i], h_e[i], h_g[i], h_f[i]);
            }
        }
        printf("C = A+B+E+G:  %d out of %d results were correct.\n", correct, count);
    }
    catch (cl::Error err){
        std::cout << "Exception\n";
        std::cerr 
            << "ERROR: "
            << err.what()
            << "("
            << err_code(err.err())
           << ")"
           << std::endl;
    }

    return 0;
}