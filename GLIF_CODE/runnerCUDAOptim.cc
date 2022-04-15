#include "definitionsInternalCUDAOptim.h"

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
scalar* Vv1_pop;
scalar* d_Vv1_pop;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

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
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntv1_pop, glbSpkCntv1_pop, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkv1_pop, glbSpkv1_pop, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushv1_popCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntv1_pop, glbSpkCntv1_pop, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkv1_pop, glbSpkv1_pop, glbSpkCntv1_pop[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVv1_popToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_Vv1_pop, Vv1_pop, 100 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVv1_popToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vv1_pop, Vv1_pop, 100 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushv1_popStateToDevice(bool uninitialisedOnly) {
    pushVv1_popToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullv1_popSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntv1_pop, d_glbSpkCntv1_pop, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkv1_pop, d_glbSpkv1_pop, 100 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullv1_popCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntv1_pop, d_glbSpkCntv1_pop, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkv1_pop, d_glbSpkv1_pop, glbSpkCntv1_pop[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vv1_pop, d_Vv1_pop, 100 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVv1_popFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vv1_pop, d_Vv1_pop, 100 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullv1_popStateFromDevice() {
    pullVv1_popFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getv1_popCurrentSpikes(unsigned int batch) {
    return (glbSpkv1_pop);
}

unsigned int& getv1_popCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntv1_pop[0];
}

scalar* getCurrentVv1_pop(unsigned int batch) {
    return Vv1_pop;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushv1_popStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullv1_popStateFromDevice();
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
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntv1_pop, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntv1_pop, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkv1_pop, 100 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkv1_pop, 100 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vv1_pop, 100 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vv1_pop, 100 * sizeof(scalar)));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntv1_pop, d_glbSpkv1_pop, d_Vv1_pop, 100);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntv1_pop, d_glbSpkv1_pop, d_Vv1_pop, 100);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_glbSpkCntv1_pop);
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
    CHECK_CUDA_ERRORS(cudaFreeHost(Vv1_pop));
    CHECK_CUDA_ERRORS(cudaFree(d_Vv1_pop));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
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
    updateNeurons(t); 
    iT++;
    t = iT*DT;
}

