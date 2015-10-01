#include "inttypes.h"
#include "cudaErrorHadling.h"
const int32_t BLOCK_SIZE = 512;

__global__ void sum_kernel(int32_t *a, int32_t *b, int32_t *c, int32_t n){
	int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < n){
	    c[idx] = a[idx] + b[idx];
    }
}

void launchSumKernel(int32_t *aDev, int32_t *bDev, int32_t *cDev, int32_t n){
   dim3 threads = dim3(BLOCK_SIZE);
   dim3 blocks  = dim3((n - 1)/BLOCK_SIZE + 1);
   
   SAFE_KERNEL_CALL( (sum_kernel<<< blocks, threads >>>(aDev, bDev, cDev, n)) );
}
