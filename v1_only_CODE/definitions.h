#pragma once
#ifdef BUILDING_GENERATED_CODE
#define EXPORT_VAR __declspec(dllexport) extern
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_VAR __declspec(dllimport) extern
#define EXPORT_FUNC __declspec(dllimport)
#endif
// Standard C++ includes
#include <random>
#include <string>
#include <stdexcept>

// Standard C includes
#include <cassert>
#include <cstdint>
#define DT 1.00000000000000006e-01f
typedef float scalar;
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f

#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double initTime;
EXPORT_VAR double initSparseTime;
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_Stim glbSpkCntStim[0]
#define spike_Stim glbSpkStim
#define glbSpkShiftStim 0

EXPORT_VAR unsigned int* glbSpkCntStim;
EXPORT_VAR unsigned int* d_glbSpkCntStim;
EXPORT_VAR unsigned int* glbSpkStim;
EXPORT_VAR unsigned int* d_glbSpkStim;
EXPORT_VAR unsigned int* startSpikeStim;
EXPORT_VAR unsigned int* d_startSpikeStim;
EXPORT_VAR unsigned int* endSpikeStim;
EXPORT_VAR unsigned int* d_endSpikeStim;
EXPORT_VAR scalar* spikeTimesStim;
EXPORT_VAR scalar* d_spikeTimesStim;
#define spikeCount_v1_pop glbSpkCntv1_pop[spkQuePtrv1_pop]
#define spike_v1_pop (glbSpkv1_pop + (spkQuePtrv1_pop * 230924))
#define glbSpkShiftv1_pop spkQuePtrv1_pop*230924

EXPORT_VAR unsigned int* glbSpkCntv1_pop;
EXPORT_VAR unsigned int* d_glbSpkCntv1_pop;
EXPORT_VAR unsigned int* glbSpkv1_pop;
EXPORT_VAR unsigned int* d_glbSpkv1_pop;
EXPORT_VAR unsigned int spkQuePtrv1_pop;
EXPORT_VAR unsigned int* d_spkQuePtrv1_pop;
EXPORT_VAR scalar* Vv1_pop;
EXPORT_VAR scalar* d_Vv1_pop;
EXPORT_VAR scalar* mv1_pop;
EXPORT_VAR scalar* d_mv1_pop;
EXPORT_VAR scalar* hv1_pop;
EXPORT_VAR scalar* d_hv1_pop;
EXPORT_VAR scalar* nv1_pop;
EXPORT_VAR scalar* d_nv1_pop;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynStimPop1;
EXPORT_VAR float* d_inSynStimPop1;
EXPORT_VAR float* inSynv1_pop_synapses;
EXPORT_VAR float* d_inSynv1_pop_synapses;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthStimPop1;
EXPORT_VAR unsigned int* rowLengthStimPop1;
EXPORT_VAR unsigned int* d_rowLengthStimPop1;
EXPORT_VAR uint32_t* indStimPop1;
EXPORT_VAR uint32_t* d_indStimPop1;
EXPORT_VAR const unsigned int maxRowLengthv1_pop_synapses;
EXPORT_VAR unsigned int* rowLengthv1_pop_synapses;
EXPORT_VAR unsigned int* d_rowLengthv1_pop_synapses;
EXPORT_VAR uint32_t* indv1_pop_synapses;
EXPORT_VAR uint32_t* d_indv1_pop_synapses;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

EXPORT_FUNC void pushStimSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimSpikesFromDevice();
EXPORT_FUNC void pushStimCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getStimCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getStimCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushstartSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullstartSpikeStimFromDevice();
EXPORT_FUNC void pushCurrentstartSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentstartSpikeStimFromDevice();
EXPORT_FUNC unsigned int* getCurrentstartSpikeStim(unsigned int batch = 0); 
EXPORT_FUNC void pushendSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullendSpikeStimFromDevice();
EXPORT_FUNC void pushCurrentendSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentendSpikeStimFromDevice();
EXPORT_FUNC unsigned int* getCurrentendSpikeStim(unsigned int batch = 0); 
EXPORT_FUNC void pushStimStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimStateFromDevice();
EXPORT_FUNC void allocatespikeTimesStim(unsigned int count);
EXPORT_FUNC void freespikeTimesStim();
EXPORT_FUNC void pushspikeTimesStimToDevice(unsigned int count);
EXPORT_FUNC void pullspikeTimesStimFromDevice(unsigned int count);
EXPORT_FUNC void pushv1_popSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullv1_popSpikesFromDevice();
EXPORT_FUNC void pushv1_popCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullv1_popCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getv1_popCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getv1_popCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVv1_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVv1_popFromDevice();
EXPORT_FUNC void pushCurrentVv1_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVv1_popFromDevice();
EXPORT_FUNC scalar* getCurrentVv1_pop(unsigned int batch = 0); 
EXPORT_FUNC void pushmv1_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmv1_popFromDevice();
EXPORT_FUNC void pushCurrentmv1_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentmv1_popFromDevice();
EXPORT_FUNC scalar* getCurrentmv1_pop(unsigned int batch = 0); 
EXPORT_FUNC void pushhv1_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullhv1_popFromDevice();
EXPORT_FUNC void pushCurrenthv1_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrenthv1_popFromDevice();
EXPORT_FUNC scalar* getCurrenthv1_pop(unsigned int batch = 0); 
EXPORT_FUNC void pushnv1_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnv1_popFromDevice();
EXPORT_FUNC void pushCurrentnv1_popToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentnv1_popFromDevice();
EXPORT_FUNC scalar* getCurrentnv1_pop(unsigned int batch = 0); 
EXPORT_FUNC void pushv1_popStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullv1_popStateFromDevice();
EXPORT_FUNC void pushStimPop1ConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimPop1ConnectivityFromDevice();
EXPORT_FUNC void pushv1_pop_synapsesConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullv1_pop_synapsesConnectivityFromDevice();
EXPORT_FUNC void pushinSynStimPop1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynStimPop1FromDevice();
EXPORT_FUNC void pushStimPop1StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimPop1StateFromDevice();
EXPORT_FUNC void pushinSynv1_pop_synapsesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynv1_pop_synapsesFromDevice();
EXPORT_FUNC void pushv1_pop_synapsesStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullv1_pop_synapsesStateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC size_t getFreeDeviceMemBytes();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t); 
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"
