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
#define spikeCount_v1_pop glbSpkCntv1_pop[0]
#define spike_v1_pop glbSpkv1_pop
#define glbSpkShiftv1_pop 0

EXPORT_VAR unsigned int* glbSpkCntv1_pop;
EXPORT_VAR unsigned int* d_glbSpkCntv1_pop;
EXPORT_VAR unsigned int* glbSpkv1_pop;
EXPORT_VAR unsigned int* d_glbSpkv1_pop;
EXPORT_VAR scalar* Vv1_pop;
EXPORT_VAR scalar* d_Vv1_pop;

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
EXPORT_FUNC void pushv1_popStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullv1_popStateFromDevice();
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