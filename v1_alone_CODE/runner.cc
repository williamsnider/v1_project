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
float* inSynv1_pop_self;
float* d_inSynv1_pop_self;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthv1_pop_self = 2;
unsigned int* rowLengthv1_pop_self;
unsigned int* d_rowLengthv1_pop_self;
uint32_t* indv1_pop_self;
uint32_t* d_indv1_pop_self;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
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

void pushv1_pop_selfConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthv1_pop_self, rowLengthv1_pop_self, 230924 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_indv1_pop_self, indv1_pop_self, 461848 * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void pushinSynv1_pop_selfToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynv1_pop_self, inSynv1_pop_self, 230924 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushv1_pop_selfStateToDevice(bool uninitialisedOnly) {
    pushinSynv1_pop_selfToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
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

void pullv1_pop_selfConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthv1_pop_self, d_rowLengthv1_pop_self, 230924 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indv1_pop_self, d_indv1_pop_self, 461848 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullinSynv1_pop_selfFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynv1_pop_self, d_inSynv1_pop_self, 230924 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullv1_pop_selfStateFromDevice() {
    pullinSynv1_pop_selfFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
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
    pushv1_popStateToDevice(uninitialisedOnly);
    pushv1_pop_selfStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushv1_pop_selfConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullv1_popStateFromDevice();
    pullv1_pop_selfStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
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
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynv1_pop_self, 230924 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynv1_pop_self, 230924 * sizeof(float)));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthv1_pop_self, 230924 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthv1_pop_self, 230924 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indv1_pop_self, 461848 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indv1_pop_self, 461848 * sizeof(uint32_t)));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntv1_pop, d_glbSpkv1_pop, d_spkQuePtrv1_pop, d_Vv1_pop, d_mv1_pop, d_hv1_pop, d_nv1_pop, d_inSynv1_pop_self, 230924);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntv1_pop, d_glbSpkv1_pop, d_spkQuePtrv1_pop, d_Vv1_pop, d_mv1_pop, d_hv1_pop, d_nv1_pop, d_inSynv1_pop_self, 230924);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynv1_pop_self, d_glbSpkCntv1_pop, d_glbSpkv1_pop, d_spkQuePtrv1_pop, d_spkQuePtrv1_pop, d_rowLengthv1_pop_self, d_indv1_pop_self, 2, 230924, 230924);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_spkQuePtrv1_pop, d_glbSpkCntv1_pop);
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
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynv1_pop_self));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynv1_pop_self));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthv1_pop_self));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthv1_pop_self));
    CHECK_CUDA_ERRORS(cudaFreeHost(indv1_pop_self));
    CHECK_CUDA_ERRORS(cudaFree(d_indv1_pop_self));
    
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

