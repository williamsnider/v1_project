#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedNeuronUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int* startSpike;
    unsigned int* endSpike;
    scalar* spikeTimes;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
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
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    unsigned int* spkCnt;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup1
 {
    unsigned int* spkQuePtr;
    unsigned int* spkCnt;
    
}
;
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup0 d_mergedNeuronSpikeQueueUpdateGroup0[1];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt) {
    MergedNeuronSpikeQueueUpdateGroup0 group = {spkCnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup0, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup0), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup0)));
}
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup1 d_mergedNeuronSpikeQueueUpdateGroup1[1];
void pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkQuePtr, unsigned int* spkCnt) {
    MergedNeuronSpikeQueueUpdateGroup1 group = {spkQuePtr, spkCnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup1, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup1), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup1)));
}
__device__ __constant__ MergedNeuronUpdateGroup0 d_mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int* startSpike, unsigned int* endSpike, scalar* spikeTimes, unsigned int numNeurons) {
    MergedNeuronUpdateGroup0 group = {spkCnt, spk, startSpike, endSpike, spikeTimes, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &group, sizeof(MergedNeuronUpdateGroup0), idx * sizeof(MergedNeuronUpdateGroup0)));
}
__device__ __constant__ MergedNeuronUpdateGroup1 d_mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int* spkQuePtr, scalar* V, scalar* m, scalar* h, scalar* n, float* inSynInSyn0, float* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronUpdateGroup1 group = {spkCnt, spk, spkQuePtr, V, m, h, n, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &group, sizeof(MergedNeuronUpdateGroup1), idx * sizeof(MergedNeuronUpdateGroup1)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void pushMergedNeuronUpdate0spikeTimesToDevice(unsigned int idx, scalar* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup0) * (idx)) + offsetof(MergedNeuronUpdateGroup0, spikeTimes)));
}

__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID1[] = {64, };

extern "C" __global__ void neuronSpikeQueueUpdateKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    if(id < 1) {
        struct MergedNeuronSpikeQueueUpdateGroup0 *group = &d_mergedNeuronSpikeQueueUpdateGroup0[id - 0]; 
        group->spkCnt[0] = 0;
    }
    if(id >= 1 && id < 2) {
        struct MergedNeuronSpikeQueueUpdateGroup1 *group = &d_mergedNeuronSpikeQueueUpdateGroup1[id - 1]; 
        *group->spkQuePtr  = (*group->spkQuePtr + 1) % 11;
        group->spkCnt[*group->spkQuePtr] = 0; 
    }
}

extern "C" __global__ void updateNeuronsKernel(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[64];
    __shared__ unsigned int shPosSpk;
    __shared__ unsigned int shSpkCount;
    if (threadIdx.x == 0) {
        shSpkCount = 0;
    }
    
    __syncthreads();
    // merged0
    if(id < 64) {
        struct MergedNeuronUpdateGroup0 *group = &d_mergedNeuronUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        
        if(lid < group->numNeurons) {
            unsigned int lstartSpike = group->startSpike[lid];
            const unsigned int lendSpike = group->endSpike[lid];
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            // test for and register a true spike
            if (lstartSpike != lendSpike && t >= group->spikeTimes[lstartSpike]) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                // spike reset code
                lstartSpike++;
                
            }
            group->startSpike[lid] = lstartSpike;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
    }
    // merged1
    if(id >= 64 && id < 231040) {
        struct MergedNeuronUpdateGroup1 *group = &d_mergedNeuronUpdateGroup1[0]; 
        const unsigned int lid = id - 64;
        const unsigned int readDelayOffset = (((*group->spkQuePtr + 10) % 11) * group->numNeurons);
        const unsigned int writeDelayOffset = (*group->spkQuePtr * group->numNeurons);
        
        if(lid < group->numNeurons) {
            scalar lV = group->V[lid];
            scalar lm = group->m[lid];
            scalar lh = group->h[lid];
            scalar ln = group->n[lid];
            
            float Isyn = 0;
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn * ((-8.00000000000000000e+01f) - lV);
                linSyn*=(9.04837418035959518e-01f);
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn1[lid];
                Isyn += linSyn * ((-8.00000000000000000e+01f) - lV);
                linSyn*=(9.04837418035959518e-01f);
                group->inSynInSyn1[lid] = linSyn;
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike = (lV >= 0.0f);
            // calculate membrane potential
            scalar Imem;
            unsigned int mt;
            scalar mdt= DT/25.0f;
            for (mt=0; mt < 25; mt++) {
               Imem= -(lm*lm*lm*lh*(7.15000000000000036e+00f)*(lV-((5.00000000000000000e+01f)))+
                   ln*ln*ln*ln*(1.42999999999999994e+00f)*(lV-((-9.50000000000000000e+01f)))+
                   (2.67200000000000007e-02f)*(lV-((-6.35630000000000024e+01f)))-Isyn);
               scalar _a;
               if (lV == -52.0f) {
                   _a= 1.28f;
               }
               else {
                   _a= 0.32f*(-52.0f-lV)/(exp((-52.0f-lV)/4.0f)-1.0f);
               }
               scalar _b;
               if (lV == -25.0f) {
                   _b= 1.4f;
               }
               else {
                   _b= 0.28f*(lV+25.0f)/(exp((lV+25.0f)/5.0f)-1.0f);
               }
               lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
               _a= 0.128f*exp((-48.0f-lV)/18.0f);
               _b= 4.0f / (exp((-25.0f-lV)/5.0f)+1.0f);
               lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
               if (lV == -50.0f) {
                   _a= 0.16f;
               }
               else {
                   _a= 0.032f*(-50.0f-lV)/(exp((-50.0f-lV)/5.0f)-1.0f);
               }
               _b= 0.5f*exp((-55.0f-lV)/40.0f);
               ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
               lV+= Imem/(1.42999999999999988e-01f)*mdt;
            }
            
            // test for and register a true spike
            if ((lV >= 0.0f) && !(oldSpike)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
            }
            group->V[lid] = lV;
            group->m[lid] = lm;
            group->h[lid] = lh;
            group->n[lid] = ln;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[*group->spkQuePtr], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[writeDelayOffset + shPosSpk + threadIdx.x] = n;
        }
    }
}
void updateNeurons(float t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(1, 1);
        neuronSpikeQueueUpdateKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(64, 1);
        const dim3 grid(3610, 1);
        updateNeuronsKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
