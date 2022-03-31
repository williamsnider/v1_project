#include "definitionsInternal.h"

#pragma warning(disable: 4297)
extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntPop1;
unsigned int* d_glbSpkCntPop1;
unsigned int* glbSpkPop1;
unsigned int* d_glbSpkPop1;
unsigned int spkQuePtrPop1 = 0;
unsigned int* d_spkQuePtrPop1;
scalar* VPop1;
scalar* d_VPop1;
scalar* mPop1;
scalar* d_mPop1;
scalar* hPop1;
scalar* d_hPop1;
scalar* nPop1;
scalar* d_nPop1;
unsigned int* glbSpkCntStim;
unsigned int* d_glbSpkCntStim;
unsigned int* glbSpkStim;
unsigned int* d_glbSpkStim;
unsigned int* startSpikeStim;
unsigned int* d_startSpikeStim;
unsigned int* endSpikeStim;
unsigned int* d_endSpikeStim;
scalar* spikeTimesStim;
scalar* d_spikeTimesStim;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
float* inSynStimPop1;
float* d_inSynStimPop1;
float* inSynPop1self;
float* d_inSynPop1self;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthPop1self = 1;
unsigned int* rowLengthPop1self;
unsigned int* d_rowLengthPop1self;
uint32_t* indPop1self;
uint32_t* d_indPop1self;
const unsigned int maxRowLengthStimPop1 = 1;
unsigned int* rowLengthStimPop1;
unsigned int* d_rowLengthStimPop1;
uint32_t* indStimPop1;
uint32_t* d_indStimPop1;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------
void allocatespikeTimesStim(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaHostAlloc(&spikeTimesStim, count * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_spikeTimesStim, count * sizeof(scalar)));
    pushMergedNeuronUpdate1spikeTimesToDevice(0, d_spikeTimesStim);
}
void freespikeTimesStim() {
    CHECK_CUDA_ERRORS(cudaFreeHost(spikeTimesStim));
    CHECK_CUDA_ERRORS(cudaFree(d_spikeTimesStim));
}
void pushspikeTimesStimToDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_spikeTimesStim, spikeTimesStim, count * sizeof(scalar), cudaMemcpyHostToDevice));
}
void pullspikeTimesStimFromDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(spikeTimesStim, d_spikeTimesStim, count * sizeof(scalar), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushPop1SpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPop1, glbSpkCntPop1, 11 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPop1, glbSpkPop1, 110 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushPop1CurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPop1 + spkQuePtrPop1, glbSpkCntPop1 + spkQuePtrPop1, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPop1 + (spkQuePtrPop1*10), glbSpkPop1 + (spkQuePtrPop1 * 10), glbSpkCntPop1[spkQuePtrPop1] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVPop1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VPop1, VPop1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVPop1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VPop1, VPop1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushmPop1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mPop1, mPop1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentmPop1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_mPop1, mPop1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushhPop1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_hPop1, hPop1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrenthPop1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_hPop1, hPop1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushnPop1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_nPop1, nPop1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentnPop1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_nPop1, nPop1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushPop1StateToDevice(bool uninitialisedOnly) {
    pushVPop1ToDevice(uninitialisedOnly);
    pushmPop1ToDevice(uninitialisedOnly);
    pushhPop1ToDevice(uninitialisedOnly);
    pushnPop1ToDevice(uninitialisedOnly);
}

void pushStimSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntStim, glbSpkCntStim, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkStim, glbSpkStim, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushStimCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntStim, glbSpkCntStim, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkStim, glbSpkStim, glbSpkCntStim[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushstartSpikeStimToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_startSpikeStim, startSpikeStim, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushCurrentstartSpikeStimToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_startSpikeStim, startSpikeStim, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushendSpikeStimToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_endSpikeStim, endSpikeStim, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushCurrentendSpikeStimToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_endSpikeStim, endSpikeStim, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushStimStateToDevice(bool uninitialisedOnly) {
    pushstartSpikeStimToDevice(uninitialisedOnly);
    pushendSpikeStimToDevice(uninitialisedOnly);
}

void pushPop1selfConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthPop1self, rowLengthPop1self, 10 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indPop1self, indPop1self, 10 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushStimPop1ConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthStimPop1, rowLengthStimPop1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indStimPop1, indStimPop1, 1 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushinSynPop1selfToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynPop1self, inSynPop1self, 10 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushPop1selfStateToDevice(bool uninitialisedOnly) {
    pushinSynPop1selfToDevice(uninitialisedOnly);
}

void pushinSynStimPop1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynStimPop1, inSynStimPop1, 10 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushStimPop1StateToDevice(bool uninitialisedOnly) {
    pushinSynStimPop1ToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullPop1SpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPop1, d_glbSpkCntPop1, 11 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkPop1, d_glbSpkPop1, 110 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullPop1CurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPop1 + spkQuePtrPop1, d_glbSpkCntPop1 + spkQuePtrPop1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkPop1 + (spkQuePtrPop1 * 10), d_glbSpkPop1 + (spkQuePtrPop1 * 10), glbSpkCntPop1[spkQuePtrPop1] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VPop1, d_VPop1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VPop1, d_VPop1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullmPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mPop1, d_mPop1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentmPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mPop1, d_mPop1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullhPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(hPop1, d_hPop1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrenthPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(hPop1, d_hPop1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullnPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(nPop1, d_nPop1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentnPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(nPop1, d_nPop1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullPop1StateFromDevice() {
    pullVPop1FromDevice();
    pullmPop1FromDevice();
    pullhPop1FromDevice();
    pullnPop1FromDevice();
}

void pullStimSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntStim, d_glbSpkCntStim, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkStim, d_glbSpkStim, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullStimCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntStim, d_glbSpkCntStim, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkStim, d_glbSpkStim, glbSpkCntStim[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullstartSpikeStimFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(startSpikeStim, d_startSpikeStim, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullCurrentstartSpikeStimFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(startSpikeStim, d_startSpikeStim, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullendSpikeStimFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(endSpikeStim, d_endSpikeStim, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullCurrentendSpikeStimFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(endSpikeStim, d_endSpikeStim, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullStimStateFromDevice() {
    pullstartSpikeStimFromDevice();
    pullendSpikeStimFromDevice();
}

void pullPop1selfConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthPop1self, d_rowLengthPop1self, 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indPop1self, d_indPop1self, 10 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullStimPop1ConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthStimPop1, d_rowLengthStimPop1, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indStimPop1, d_indStimPop1, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullinSynPop1selfFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynPop1self, d_inSynPop1self, 10 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullPop1selfStateFromDevice() {
    pullinSynPop1selfFromDevice();
}

void pullinSynStimPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynStimPop1, d_inSynStimPop1, 10 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullStimPop1StateFromDevice() {
    pullinSynStimPop1FromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getPop1CurrentSpikes(unsigned int batch) {
    return (glbSpkPop1 + (spkQuePtrPop1 * 10));
}

unsigned int& getPop1CurrentSpikeCount(unsigned int batch) {
    return glbSpkCntPop1[spkQuePtrPop1];
}

scalar* getCurrentVPop1(unsigned int batch) {
    return VPop1;
}

scalar* getCurrentmPop1(unsigned int batch) {
    return mPop1;
}

scalar* getCurrenthPop1(unsigned int batch) {
    return hPop1;
}

scalar* getCurrentnPop1(unsigned int batch) {
    return nPop1;
}

unsigned int* getStimCurrentSpikes(unsigned int batch) {
    return (glbSpkStim);
}

unsigned int& getStimCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntStim[0];
}

unsigned int* getCurrentstartSpikeStim(unsigned int batch) {
    return startSpikeStim;
}

unsigned int* getCurrentendSpikeStim(unsigned int batch) {
    return endSpikeStim;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushPop1StateToDevice(uninitialisedOnly);
    pushStimStateToDevice(uninitialisedOnly);
    pushPop1selfStateToDevice(uninitialisedOnly);
    pushStimPop1StateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushPop1selfConnectivityToDevice(uninitialisedOnly);
    pushStimPop1ConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullPop1StateFromDevice();
    pullStimStateFromDevice();
    pullPop1selfStateFromDevice();
    pullStimPop1StateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullPop1CurrentSpikesFromDevice();
    pullStimCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateMem() {
    int deviceID;
    CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, "0000:09:00.0"));
    CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));
    
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntPop1, 11 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntPop1, 11 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkPop1, 110 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkPop1, 110 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_spkQuePtrPop1, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&VPop1, 10 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_VPop1, 10 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&mPop1, 10 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mPop1, 10 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&hPop1, 10 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_hPop1, 10 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&nPop1, 10 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_nPop1, 10 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntStim, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntStim, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkStim, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkStim, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&startSpikeStim, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_startSpikeStim, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&endSpikeStim, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_endSpikeStim, 1 * sizeof(unsigned int)));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynStimPop1, 10 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynStimPop1, 10 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynPop1self, 10 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynPop1self, 10 * sizeof(float)));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthPop1self, 10 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthPop1self, 10 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indPop1self, 10 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indPop1self, 10 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthStimPop1, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthStimPop1, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indStimPop1, 1 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indStimPop1, 1 * sizeof(uint32_t)));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntStim, d_glbSpkStim, 1);
    pushMergedNeuronInitGroup1ToDevice(0, d_glbSpkCntPop1, d_glbSpkPop1, d_spkQuePtrPop1, d_VPop1, d_mPop1, d_hPop1, d_nPop1, d_inSynStimPop1, d_inSynPop1self, 10);
    pushMergedSynapseConnectivityInitGroup0ToDevice(0, d_rowLengthPop1self, d_indPop1self, 1, 10, 10);
    pushMergedSynapseConnectivityInitGroup1ToDevice(0, d_rowLengthStimPop1, d_indStimPop1, 1, 1, 10);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntPop1, d_glbSpkPop1, d_spkQuePtrPop1, d_VPop1, d_mPop1, d_hPop1, d_nPop1, d_inSynStimPop1, d_inSynPop1self, 10);
    pushMergedNeuronUpdateGroup1ToDevice(0, d_glbSpkCntStim, d_glbSpkStim, d_startSpikeStim, d_endSpikeStim, d_spikeTimesStim, 1);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynPop1self, d_glbSpkCntPop1, d_glbSpkPop1, d_spkQuePtrPop1, d_spkQuePtrPop1, d_rowLengthPop1self, d_indPop1self, 1, 10, 10);
    pushMergedPresynapticUpdateGroup1ToDevice(0, d_inSynStimPop1, d_glbSpkCntStim, d_glbSpkStim, d_spkQuePtrPop1, d_rowLengthStimPop1, d_indStimPop1, 1, 1, 10);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_spkQuePtrPop1, d_glbSpkCntPop1);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(0, d_glbSpkCntStim);
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_spkQuePtrPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(VPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_VPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(mPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_mPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(hPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_hPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(nPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_nPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntStim));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntStim));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkStim));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkStim));
    CHECK_CUDA_ERRORS(cudaFreeHost(startSpikeStim));
    CHECK_CUDA_ERRORS(cudaFree(d_startSpikeStim));
    CHECK_CUDA_ERRORS(cudaFreeHost(endSpikeStim));
    CHECK_CUDA_ERRORS(cudaFree(d_endSpikeStim));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynStimPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynStimPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynPop1self));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynPop1self));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthPop1self));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthPop1self));
    CHECK_CUDA_ERRORS(cudaFreeHost(indPop1self));
    CHECK_CUDA_ERRORS(cudaFree(d_indPop1self));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthStimPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthStimPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(indStimPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_indStimPop1));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
}

size_t getFreeDeviceMemBytes() {
    size_t free;
    size_t total;
    CHECK_CUDA_ERRORS(cudaMemGetInfo(&free, &total));
    return free;
}

void stepTime() {
    updateSynapses(t);
    spkQuePtrPop1 = (spkQuePtrPop1 + 1) % 11;
    updateNeurons(t); 
    iT++;
    t = iT*DT;
}

