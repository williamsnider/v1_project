#include "definitionsInternal.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int* spkQuePtr;
    scalar* V;
    scalar* m;
    scalar* h;
    scalar* n;
    float* inSynInSyn0;
    float* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup0
 {
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup1
 {
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int numNeurons) {
    MergedNeuronInitGroup0 group = {spkCnt, spk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0)));
}
__device__ __constant__ MergedNeuronInitGroup1 d_mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int* spkQuePtr, scalar* V, scalar* m, scalar* h, scalar* n, float* inSynInSyn0, float* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronInitGroup1 group = {spkCnt, spk, spkQuePtr, V, m, h, n, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup1, &group, sizeof(MergedNeuronInitGroup1), idx * sizeof(MergedNeuronInitGroup1)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup0 d_mergedSynapseConnectivityInitGroup0[1];
void pushMergedSynapseConnectivityInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseConnectivityInitGroup0 group = {rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup0, &group, sizeof(MergedSynapseConnectivityInitGroup0), idx * sizeof(MergedSynapseConnectivityInitGroup0)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup1 d_mergedSynapseConnectivityInitGroup1[1];
void pushMergedSynapseConnectivityInitGroup1ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseConnectivityInitGroup1 group = {rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup1, &group, sizeof(MergedSynapseConnectivityInitGroup1), idx * sizeof(MergedSynapseConnectivityInitGroup1)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedNeuronInitGroupStartID1[] = {32, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID0[] = {64, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID1[] = {96, };

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 32) {
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
            // current source variables
        }
    }
    // merged1
    if(id >= 32 && id < 64) {
        struct MergedNeuronInitGroup1 *group = &d_mergedNeuronInitGroup1[0]; 
        const unsigned int lid = id - 32;
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
             {
                group->inSynInSyn1[lid] = 0.000000000e+00f;
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
    // merged0
    if(id >= 64 && id < 96) {
        struct MergedSynapseConnectivityInitGroup0 *group = &d_mergedSynapseConnectivityInitGroup0[0]; 
        const unsigned int lid = id - 64;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            // Build sparse connectivity
            while(true) {
                
                do {
                    const unsigned int idx = (lid * group->rowStride) + group->rowLength[lid];
                    group->ind[idx] = (lid + 1)%group->numTrgNeurons;
                    group->rowLength[lid]++;
                }
                while(false);
                break;
                
            }
        }
    }
    // merged1
    if(id >= 96 && id < 128) {
        struct MergedSynapseConnectivityInitGroup1 *group = &d_mergedSynapseConnectivityInitGroup1[0]; 
        const unsigned int lid = id - 96;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            // Build sparse connectivity
            while(true) {
                do {
                    const unsigned int idx = (lid * group->rowStride) + group->rowLength[lid];
                    group->ind[idx] = lid;
                    group->rowLength[lid]++;
                }
                while(false);
                break;
                
            }
        }
    }
    
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
