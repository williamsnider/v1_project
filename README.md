# v1_project
Code for creating a GPU-accelerated model of the mouse primary visual cortex

## Current Progress
- Constructed prototype model "v1_only" (see v1_only.ipynb) with the ~230,000 nodes from the GLIF dataset (did not include LGN or background neurons)
- Added all synaptic connections between each of these v1 neurons (~70,000,000 connections) (did not )
- Added an initial stimulus to excite node 0
- Ran a quick simulation (2000 timesteps) and observed runaway/unrealistic excitation in the model 



## TODO
1. Improve synaptic connections of "v1_only" model. At present the parameters are quickly-and-dirtily copied from the PyGeNN tutorial. Using the parameters in the GLIF Network dataset will improve the model and likely solve the runaway excitation
2. Add in LGN neuron nodes and synaptic connections
3. Add in background neuron nodes and synaptic connections 
4. Simulate model using diverse stimuli (flashes, natural movies, looming stimulus) and compare to results in Billeh et al. 2020 
5. Run simulation in real-time with input from a webcam