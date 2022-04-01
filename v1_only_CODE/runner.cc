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
unsigned int* glbSpkCntv1_pop;
unsigned int* d_glbSpkCntv1_pop;
unsigned int* glbSpkv1_pop;
unsigned int* d_glbSpkv1_pop;
unsigned int spkQuePtrv1_pop = 0;
unsigned int* d_spkQuePtrv1_pop;
scalar* Vv1_pop;
scalar* d_Vv1_pop;
scalar* mv1_pop;
scalar* d_mv1_pop;
scalar* hv1_pop;
scalar* d_hv1_pop;
scalar* nv1_pop;
scalar* d_nv1_pop;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
float* inSynStimPop1;
float* d_inSynStimPop1;
float* inSynv1_pop_synapses;
float* d_inSynv1_pop_synapses;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthStimPop1 = 1;
unsigned int* rowLengthStimPop1;
unsigned int* d_rowLengthStimPop1;
uint32_t* indStimPop1;
uint32_t* d_indStimPop1;
const unsigned int maxRowLengthv1_pop_synapses = 817;
unsigned int* rowLengthv1_pop_synapses;
unsigned int* d_rowLengthv1_pop_synapses;
uint32_t* indv1_pop_synapses;
uint32_t* d_indv1_pop_synapses;

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
    pushMergedNeuronUpdate0spikeTimesToDevice(0, d_spikeTimesStim);
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

void pushv1_popSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntv1_pop, glbSpkCntv1_pop, 11 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkv1_pop, glbSpkv1_pop, 2540164 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushv1_popCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntv1_pop + spkQuePtrv1_pop, glbSpkCntv1_pop + spkQuePtrv1_pop, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkv1_pop + (spkQuePtrv1_pop*230924), glbSpkv1_pop + (spkQuePtrv1_pop * 230924), glbSpkCntv1_pop[spkQuePtrv1_pop] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVv1_popToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_Vv1_pop, Vv1_pop, 230924 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVv1_popToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vv1_pop, Vv1_pop, 230924 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushmv1_popToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mv1_pop, mv1_pop, 230924 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentmv1_popToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_mv1_pop, mv1_pop, 230924 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushhv1_popToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_hv1_pop, hv1_pop, 230924 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrenthv1_popToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_hv1_pop, hv1_pop, 230924 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushnv1_popToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_nv1_pop, nv1_pop, 230924 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentnv1_popToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_nv1_pop, nv1_pop, 230924 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushv1_popStateToDevice(bool uninitialisedOnly) {
    pushVv1_popToDevice(uninitialisedOnly);
    pushmv1_popToDevice(uninitialisedOnly);
    pushhv1_popToDevice(uninitialisedOnly);
    pushnv1_popToDevice(uninitialisedOnly);
}

void pushStimPop1ConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthStimPop1, rowLengthStimPop1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indStimPop1, indStimPop1, 1 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushv1_pop_synapsesConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthv1_pop_synapses, rowLengthv1_pop_synapses, 230924 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_indv1_pop_synapses, indv1_pop_synapses, 188664908 * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void pushinSynStimPop1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynStimPop1, inSynStimPop1, 230924 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushStimPop1StateToDevice(bool uninitialisedOnly) {
    pushinSynStimPop1ToDevice(uninitialisedOnly);
}

void pushinSynv1_pop_synapsesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynv1_pop_synapses, inSynv1_pop_synapses, 230924 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushv1_pop_synapsesStateToDevice(bool uninitialisedOnly) {
    pushinSynv1_pop_synapsesToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
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

void pullv1_popSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntv1_pop, d_glbSpkCntv1_pop, 11 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkv1_pop, d_glbSpkv1_pop, 2540164 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullv1_popCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntv1_pop + spkQuePtrv1_pop, d_glbSpkCntv1_pop + spkQuePtrv1_pop, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkv1_pop + (spkQuePtrv1_pop * 230924), d_glbSpkv1_pop + (spkQuePtrv1_pop * 230924), glbSpkCntv1_pop[spkQuePtrv1_pop] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vv1_pop, d_Vv1_pop, 230924 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vv1_pop, d_Vv1_pop, 230924 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullmv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mv1_pop, d_mv1_pop, 230924 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentmv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mv1_pop, d_mv1_pop, 230924 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullhv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(hv1_pop, d_hv1_pop, 230924 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrenthv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(hv1_pop, d_hv1_pop, 230924 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullnv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(nv1_pop, d_nv1_pop, 230924 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentnv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(nv1_pop, d_nv1_pop, 230924 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullv1_popStateFromDevice() {
    pullVv1_popFromDevice();
    pullmv1_popFromDevice();
    pullhv1_popFromDevice();
    pullnv1_popFromDevice();
}

void pullStimPop1ConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthStimPop1, d_rowLengthStimPop1, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indStimPop1, d_indStimPop1, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullv1_pop_synapsesConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthv1_pop_synapses, d_rowLengthv1_pop_synapses, 230924 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indv1_pop_synapses, d_indv1_pop_synapses, 188664908 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullinSynStimPop1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynStimPop1, d_inSynStimPop1, 230924 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullStimPop1StateFromDevice() {
    pullinSynStimPop1FromDevice();
}

void pullinSynv1_pop_synapsesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynv1_pop_synapses, d_inSynv1_pop_synapses, 230924 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullv1_pop_synapsesStateFromDevice() {
    pullinSynv1_pop_synapsesFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
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

unsigned int* getv1_popCurrentSpikes(unsigned int batch) {
    return (glbSpkv1_pop + (spkQuePtrv1_pop * 230924));
}

unsigned int& getv1_popCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntv1_pop[spkQuePtrv1_pop];
}

scalar* getCurrentVv1_pop(unsigned int batch) {
    return Vv1_pop;
}

scalar* getCurrentmv1_pop(unsigned int batch) {
    return mv1_pop;
}

scalar* getCurrenthv1_pop(unsigned int batch) {
    return hv1_pop;
}

scalar* getCurrentnv1_pop(unsigned int batch) {
    return nv1_pop;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushStimStateToDevice(uninitialisedOnly);
    pushv1_popStateToDevice(uninitialisedOnly);
    pushStimPop1StateToDevice(uninitialisedOnly);
    pushv1_pop_synapsesStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushStimPop1ConnectivityToDevice(uninitialisedOnly);
    pushv1_pop_synapsesConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullStimStateFromDevice();
    pullv1_popStateFromDevice();
    pullStimPop1StateFromDevice();
    pullv1_pop_synapsesStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullStimCurrentSpikesFromDevice();
    pullv1_popCurrentSpikesFromDevice();
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
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntStim, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntStim, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkStim, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkStim, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&startSpikeStim, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_startSpikeStim, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&endSpikeStim, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_endSpikeStim, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntv1_pop, 11 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntv1_pop, 11 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkv1_pop, 2540164 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkv1_pop, 2540164 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_spkQuePtrv1_pop, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vv1_pop, 230924 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vv1_pop, 230924 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&mv1_pop, 230924 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mv1_pop, 230924 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&hv1_pop, 230924 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_hv1_pop, 230924 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&nv1_pop, 230924 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_nv1_pop, 230924 * sizeof(scalar)));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynStimPop1, 230924 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynStimPop1, 230924 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynv1_pop_synapses, 230924 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynv1_pop_synapses, 230924 * sizeof(float)));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthStimPop1, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthStimPop1, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indStimPop1, 1 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indStimPop1, 1 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthv1_pop_synapses, 230924 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthv1_pop_synapses, 230924 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indv1_pop_synapses, 188664908 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indv1_pop_synapses, 188664908 * sizeof(uint32_t)));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntv1_pop, d_glbSpkv1_pop, d_spkQuePtrv1_pop, d_Vv1_pop, d_mv1_pop, d_hv1_pop, d_nv1_pop, d_inSynStimPop1, d_inSynv1_pop_synapses, 230924);
    pushMergedNeuronInitGroup1ToDevice(0, d_glbSpkCntStim, d_glbSpkStim, 1);
    pushMergedSynapseConnectivityInitGroup0ToDevice(0, d_rowLengthStimPop1, d_indStimPop1, 1, 1, 230924);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntStim, d_glbSpkStim, d_startSpikeStim, d_endSpikeStim, d_spikeTimesStim, 1);
    pushMergedNeuronUpdateGroup1ToDevice(0, d_glbSpkCntv1_pop, d_glbSpkv1_pop, d_spkQuePtrv1_pop, d_Vv1_pop, d_mv1_pop, d_hv1_pop, d_nv1_pop, d_inSynStimPop1, d_inSynv1_pop_synapses, 230924);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynStimPop1, d_glbSpkCntStim, d_glbSpkStim, d_spkQuePtrv1_pop, d_rowLengthStimPop1, d_indStimPop1, 1, 1, 230924);
    pushMergedPresynapticUpdateGroup1ToDevice(0, d_inSynv1_pop_synapses, d_glbSpkCntv1_pop, d_glbSpkv1_pop, d_spkQuePtrv1_pop, d_spkQuePtrv1_pop, d_rowLengthv1_pop_synapses, d_indv1_pop_synapses, 817, 230924, 230924);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_glbSpkCntStim);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(0, d_spkQuePtrv1_pop, d_glbSpkCntv1_pop);
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
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntStim));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntStim));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkStim));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkStim));
    CHECK_CUDA_ERRORS(cudaFreeHost(startSpikeStim));
    CHECK_CUDA_ERRORS(cudaFree(d_startSpikeStim));
    CHECK_CUDA_ERRORS(cudaFreeHost(endSpikeStim));
    CHECK_CUDA_ERRORS(cudaFree(d_endSpikeStim));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntv1_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntv1_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkv1_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkv1_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_spkQuePtrv1_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vv1_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_Vv1_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(mv1_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_mv1_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(hv1_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_hv1_pop));
    CHECK_CUDA_ERRORS(cudaFreeHost(nv1_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_nv1_pop));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynStimPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynStimPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynv1_pop_synapses));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynv1_pop_synapses));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthStimPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthStimPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(indStimPop1));
    CHECK_CUDA_ERRORS(cudaFree(d_indStimPop1));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthv1_pop_synapses));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthv1_pop_synapses));
    CHECK_CUDA_ERRORS(cudaFreeHost(indv1_pop_synapses));
    CHECK_CUDA_ERRORS(cudaFree(d_indv1_pop_synapses));
    
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
    spkQuePtrv1_pop = (spkQuePtrv1_pop + 1) % 11;
    updateNeurons(t); 
    iT++;
    t = iT*DT;
}

