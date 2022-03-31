#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedPresynapticUpdateGroup0
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    unsigned int* srcSpkQuePtr;
    unsigned int* trgSpkQuePtr;
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedPresynapticUpdateGroup1
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    unsigned int* trgSpkQuePtr;
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
__device__ __constant__ MergedPresynapticUpdateGroup0 d_mergedPresynapticUpdateGroup0[1];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* srcSpkQuePtr, unsigned int* trgSpkQuePtr, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedPresynapticUpdateGroup0 group = {inSyn, srcSpkCnt, srcSpk, srcSpkQuePtr, trgSpkQuePtr, rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup0, &group, sizeof(MergedPresynapticUpdateGroup0), idx * sizeof(MergedPresynapticUpdateGroup0)));
}
__device__ __constant__ MergedPresynapticUpdateGroup1 d_mergedPresynapticUpdateGroup1[1];
void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* trgSpkQuePtr, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedPresynapticUpdateGroup1 group = {inSyn, srcSpkCnt, srcSpk, trgSpkQuePtr, rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup1, &group, sizeof(MergedPresynapticUpdateGroup1), idx * sizeof(MergedPresynapticUpdateGroup1)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID1[] = {32, };
extern "C" __global__ void updatePresynapticKernel(float t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ float shLg[32];
    __shared__ unsigned int shRowLength[32];
    __shared__ unsigned int shSpk[32];
    // merged0
    if(id < 32) {
        struct MergedPresynapticUpdateGroup0 *group = &d_mergedPresynapticUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        const unsigned int preDelaySlot = (*group->srcSpkQuePtr + 1) % 11;
        const unsigned int preDelayOffset = preDelaySlot * group->numSrcNeurons;
        const unsigned int postDelaySlot = *group->trgSpkQuePtr;
        const unsigned int postDelayOffset = postDelaySlot * group->numTrgNeurons;
        if(threadIdx.x < group->numTrgNeurons) {
            shLg[threadIdx.x] = 0;
        }
        __syncthreads();
         {
            const unsigned int numSpikes = group->srcSpkCnt[preDelaySlot];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[preDelayOffset + (r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            shLg[ipost] += (-2.00000000000000011e-01f);
                        }
                    }
                }
            }
        }
        
        __syncthreads();
        if(threadIdx.x < group->numTrgNeurons) {
            atomicAdd(&group->inSyn[threadIdx.x], shLg[threadIdx.x]); 
        }
    }
    // merged1
    if(id >= 32 && id < 64) {
        struct MergedPresynapticUpdateGroup1 *group = &d_mergedPresynapticUpdateGroup1[0]; 
        const unsigned int lid = id - 32;
        const unsigned int postDelaySlot = *group->trgSpkQuePtr;
        const unsigned int postDelayOffset = postDelaySlot * group->numTrgNeurons;
        if(threadIdx.x < group->numTrgNeurons) {
            shLg[threadIdx.x] = 0;
        }
        __syncthreads();
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            shLg[ipost] += (-2.00000000000000011e-01f);
                        }
                    }
                }
            }
        }
        
        __syncthreads();
        if(threadIdx.x < group->numTrgNeurons) {
            atomicAdd(&group->inSyn[threadIdx.x], shLg[threadIdx.x]); 
        }
    }
}
void updateSynapses(float t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(2, 1);
        updatePresynapticKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
