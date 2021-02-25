__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    __global float* d,
    const unsigned int count
){
    int id = get_global_id(0);
    if(id<count){
        d[id] = a[id] + b[id] + c[id];
    }
}