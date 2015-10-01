#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include <limits>
#include "inttypes.h"
#include "cudaErrorHadling.h"

void launchSumKernel(int32_t *a, int32_t *b, int32_t *c, int32_t n);

const char * const printMemorySize(size_t bytes){
    char inches[] = {' ', 'K', 'M', 'G', 'T'};
    double sz = bytes;

    int inch = 0;
    for (; sz > 512 && inch < 5; ++inch){
        sz /= 1024;
    }

    static char ret[64];
    sprintf(ret, "%.2f %cB", sz, inches[inch]);

    return ret;
}

float timer(){
    static clock_t timer = 0;
    if (!timer){
        timer = clock();

        return 0;
    }
    
    clock_t current = clock();
    float ret = ((float)(current - timer))/CLOCKS_PER_SEC;

    timer = current;
    return ret;
}

bool ourRequirementsPassed(const cudaDeviceProp & devProp){
    return devProp.major >= 1;
}

int selectCUDADevice(){
    int deviceCount = 0, suitableDevice = -1;
    cudaDeviceProp devProp;   
    cudaGetDeviceCount( &deviceCount );
    std::cout << "Found "<< deviceCount << " devices: \n";

    for (int device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties ( &devProp, device );

        std::cout << "Device: " << device                                               << std::endl;
        std::cout << "   Compute capability: " << devProp.major << "." << devProp.minor << std::endl;
        std::cout << "   Name: " << devProp.name                                        << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "   Total Global Memory: " << printMemorySize(devProp.totalGlobalMem)               << std::endl;
        std::cout << "   Shared Memory Per Block: " << printMemorySize(devProp.sharedMemPerBlock)        << std::endl;
        std::cout << "   Total Const Memory: " << printMemorySize(devProp.totalConstMem)        << std::endl;
        std::cout << "   L2 Cache size: " << printMemorySize(devProp.l2CacheSize)        << std::endl;
        std::cout << "   Memory bus width: " << printMemorySize(devProp.memoryBusWidth/8)        << std::endl;
        std::cout << "   Memory frequency: " << devProp.memoryClockRate << " kHz"       << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "   Multiprocessors: " << devProp.multiProcessorCount        << std::endl;
        std::cout << "   Clock rate: " << devProp.clockRate << " kHz"       << std::endl;
        std::cout << "   Warp Size: " << devProp.warpSize        << std::endl;
        std::cout << "   Max grid size: " << "(" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", "  << devProp.maxGridSize[2] << ")"      << std::endl;
        std::cout << "   Max block size: " << "(" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", "  << devProp.maxThreadsDim[2] << ")"      << std::endl;
        std::cout << "   Max threads per multiprocessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "   Max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "   Registers per block: " << devProp.regsPerBlock << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::endl;

        if(suitableDevice < 0 && ourRequirementsPassed(devProp)){
            suitableDevice = device;
        }
    }
    return suitableDevice;
}

void initializeRandomArray(int *array, int length){
    static const int32_t MY_INT32_MAX = std::numeric_limits<int32_t>::max();
    for(int i =0; i < length; ++i){
        array[i] = rand() % MY_INT32_MAX;
    }
}

int main(int argc, char *argv[]){
    //------------- Variables -----------
        cudaEvent_t start, stop;
        float timeCPU = 0.0,
              timeGPU = 0.0;

        int n  = 1024;
        int32_t *aHost = NULL, *bHost = NULL, *cHost = NULL;
        int32_t *aDev = NULL , *bDev = NULL , *cDev = NULL, *answer = NULL;
    //-----------------------------------

    //--------- Command line -----------
        if(argc != 2){
            std::cout << "You may define vector size via comandline as: $<program_name> <vector_size>\n";
            std::cout << "Default vector size is: " << n << std::endl;
        }
        else{
            int tmp = atoi(argv[1]);
            if (tmp > 1){
                n = tmp;
            }
        }
    //----------------------------------

    //-------- Select device -----------
        int device = selectCUDADevice();  
        if(device == -1) {
            std::cout << "Can not find suitable device" << "\n";
            return EXIT_FAILURE;
        }
        SAFE_CALL(cudaSetDevice(device));
    //-----------------------------------

    //------- Host memory allocation ------------
        int nb = n*sizeof(int);
        aHost  = (int32_t*)malloc(nb);
        bHost  = (int32_t*)malloc(nb);
        cHost  = (int32_t*)malloc(nb);
        answer = (int32_t*)malloc(nb);
    //-------------------------------------------

    //-------- Initialization arrays by random values ------
        srand(clock());
        initializeRandomArray(aHost, n);
        initializeRandomArray(bHost, n);
    //------------------------------------------------------

    //-------- Calculation on CPU --------------------------
        timer();
        for(int i = 0; i < n; ++i){
            cHost[i] = aHost[i] + bHost[i];
        }
        timeCPU = timer();
        fflush(stdout);
    //------------------------------------------------------

    //----- GPU memory allocation and initialization -------
        SAFE_CALL( cudaMalloc((void**)&aDev, nb) );
        SAFE_CALL( cudaMalloc((void**)&bDev, nb) ) ;
        SAFE_CALL( cudaMalloc((void**)&cDev, nb) ) ;

        SAFE_CALL( cudaMemcpy(aDev, aHost, nb, cudaMemcpyHostToDevice) );
        SAFE_CALL( cudaMemcpy(bDev, bHost, nb, cudaMemcpyHostToDevice) );
    //------------------------------------------------------

    //------ Calculation on GPU --------------
        SAFE_CALL( cudaEventCreate(&start) );
        SAFE_CALL( cudaEventCreate(&stop)  );

        SAFE_CALL( cudaEventRecord(start, 0) );

        launchSumKernel(aDev, bDev, cDev, n);

        SAFE_CALL( cudaEventRecord(stop, 0) );
        SAFE_CALL( cudaEventSynchronize(stop) );
        SAFE_CALL( cudaEventElapsedTime(&timeGPU, start, stop) );
    //--------------------------------------

    printf("processing time on GPU: %4.4f s\n", timeGPU/1000.0);
    printf("processing time on CPU: %4.4f s\n", timeCPU);

    //--------- Compare GPU and CPU results -----------------------------
        SAFE_CALL( cudaMemcpy(answer, cDev, nb, cudaMemcpyDeviceToHost) );

        for(int i = 0; i < n; ++i){
            if(cHost[i] != answer[i]) {
                std::cout << "Incorrect result at [" << i << "]: " << aHost[i] << " + " << bHost[i] << " = " << answer[i] << "\n";
                break;
            }
        }
    //--------------------------------------------------------------------

    SAFE_CALL( cudaFree(aDev) );
    SAFE_CALL( cudaFree(bDev) );
    SAFE_CALL( cudaFree(cDev) );

    free(aHost);
    free(bHost);
    free(cHost);
    free(answer);

    return EXIT_SUCCESS;
}

