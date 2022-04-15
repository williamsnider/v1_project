#include "definitionsInternalCUDAOptim.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    unsigned int numNeurons;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, unsigned int numNeurons) {
    MergedNeuronInitGroup0 group = {spkCnt, spk, V, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 128) {
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
             {
                scalar initVal;
                initVal = (-6.50000000000000000e+01f);
                group->V[lid] = initVal;
            }
            // current source variables
        }
    }
    
    // ------------------------------------------------------------------------
    // Custom update groups
    
    // ------------------------------------------------------------------------
    // Custom WU update groups with dense connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with kernel connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    
}
void initialize() {
    unsigned long long deviceRNGSeed = 0;
     {
        const dim3 threads(32, 1);
        const dim3 grid(4, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
    
}
