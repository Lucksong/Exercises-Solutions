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

#define TOL    (0.001)
#define LENGTH (10240000)

int main(void){
    std::vector<float> h_a(LENGTH);
    std::vector<float> h_b(LENGTH);
    std::vector<float> h_c(LENGTH);
    std::vector<float> h_d(LENGTH, 0xdeadbeef);

    cl::Buffer d_a;
    cl::Buffer d_b;
    cl::Buffer d_c;
    cl::Buffer d_d;

    int count = LENGTH;
    for(int i = 0; i < count; i++){
        h_a[i] = rand()/(float)RAND_MAX;
        h_b[i] = rand()/(float)RAND_MAX;
        h_c[i] = rand()/(float)RAND_MAX;
    }

    try{
        cl::Context context(DEVICE);
        cl::Program program(context, util::loadProgram("ocl.cl"), true);
        cl::CommandQueue queue(context);
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");
        d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);
        d_b = cl::Buffer(context, h_b.begin(), h_b.end(), true);
        d_c = cl::Buffer(context, h_c.begin(), h_c.end(), true);
        d_d = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*count);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)
            ),
            d_a,
            d_b,
            d_c,
            d_d,
            count
        );

        cl::copy(queue, d_d, h_d.begin(), h_d.end());

        int correct = 0;
        float tmp;
        for(int i=0;i<count;i++){
            tmp = h_a[i] + h_b[i] + h_c[i];
            tmp -= h_d[i];
            if(tmp*tmp<TOL*TOL){
                correct++;
            }
            else{
                printf(" tmp %f h_a %f h_b %f h_c %f h_d %f\n",tmp, h_a[i], h_b[i], h_c[i], h_d[i]);
            }
        }
        printf("D = A+B+C:  %d out of %d results were correct.\n", correct, count);
    }
    catch(cl::Error err){
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