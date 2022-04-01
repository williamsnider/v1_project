#include "definitionsInternal.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int* spkQuePtr;
    scalar* V;
    scalar* m;
    scalar* h;
    scalar* n;
    float* inSynInSyn0;
    unsigned int numNeurons;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int* spkQuePtr, scalar* V, scalar* m, scalar* h, scalar* n, float* inSynInSyn0, unsigned int numNeurons) {
    MergedNeuronInitGroup0 group = {spkCnt, spk, spkQuePtr, V, m, h, n, inSynInSyn0, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 230976) {
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                for(unsigned int d = 0; d < 11; d++) {
                    group->spkCnt[d] = 0;
                }
            }
            for(unsigned int d = 0; d < 11; d++) {
                group->spk[(d * group->numNeurons) + lid] = 0;
            }
            if(lid == 0) {
                *group->spkQuePtr = 0;
            }
             {
                scalar initVal;
                initVal = (-6.00000000000000000e+01f);
                group->V[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (5.29323999999999975e-02f);
                group->m[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (3.17676699999999979e-01f);
                group->h[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (5.96120699999999948e-01f);
                group->n[lid] = initVal;
            }
             {
                group->inSynInSyn0[lid] = 0.000000000e+00f;
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
        const dim3 threads(64, 1);
        const dim3 grid(3609, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
    
}
