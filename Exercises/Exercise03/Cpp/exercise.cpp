#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "util.hpp"
#include "err_code.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define TOL      (0.001)
#define LENGTH   (102400000)

int main(void){
    std::vector<float> h_a(LENGTH);
    std::vector<float> h_b(LENGTH);
    std::vector<float> h_c(LENGTH, 0xdeadbeef);

    cl::Buffer d_a;
    cl::Buffer d_b;
    cl::Buffer d_c;

    int count = LENGTH;
    for(int i = 0; i < count; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    try{
        cl::Context context(DEVICE);

        cl::Program program(context, util::loadProgram("vadd.cl"), true);

        cl::CommandQueue queue(context);

        auto vadd = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");

        d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
        d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        util::Timer timer;

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

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);

        cl::copy(queue, d_c, begin(h_c), end(h_c));

        int correct = 0;
        float tmp;
        util::Timer cpu_timer;
        for(int i = 0; i < count; i++){
            tmp = h_a[i] + h_b[i];
            tmp -= h_c[i];
            if(tmp*tmp < TOL*TOL){
                correct++;
            }
            else{
                printf("tmp %f h_a %f h_b %f h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
            }
        }
        double cpu_time = static_cast<double>(cpu_timer.getTimeMilliseconds()) / 1000.0;
        printf("vector add to find C = A+B: %d out of %d results were correct with %lf seconds.\n", correct, count, cpu_time);
    }
    catch(cl::Error err){
        std::cout << "Exception\n";
        std::cerr
            << "Error: "
            << err.what()
            << "("
            << err_code(err.err())
            << ")"
            << std::endl;
    }
}