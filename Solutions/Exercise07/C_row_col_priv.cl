__kernel void mmul(
    const int N,
    __global float* A,
    __global float* B,
    __global float* C
)
{
    float tmp = 0;
    float tmpRow[N];
    float tmpCol[N];
    int i = get_global_id(0);
    if(i<N){
        for(int k=0;k<N;i++){
            tmpRow[k] = A[i*N + k];
            tmpCol[k] = A[k*N + i]; 
        }
    }
}