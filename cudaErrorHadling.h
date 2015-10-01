#ifndef _CUDA_ERR_HNDL_H
#define _CUDA_ERR_HNDL_H
#include <cstdio>
#include <cstdlib>

#define SAFE_CALL( CallInstruction ) { \
    cudaError_t cuerr = CallInstruction; \
    if(cuerr != cudaSuccess) { \
         fprintf(stderr, "CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
         exit(EXIT_FAILURE); \
    } \
}

#define SAFE_KERNEL_CALL( KernelCallInstruction ) { \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        fprintf(stderr, "CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
        exit(EXIT_FAILURE); \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        fprintf(stderr, "CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
        exit(EXIT_FAILURE); \
    } \
}
#endif
