# Google Summer of Code 2022


## A GPU-accelerated Model of the Mouse Primary Visual Cortex


### Project Outline

- 6/13 - 6/26 
    - ~~Implement GLIF neuron type at five levels of complexity~~
    - Add these models to PyGeNN’s standard cell models
- 6/27 - 7/10 
    - Use Billeh LGN models to convert videos to spikes offline
    - Insert spikes into model using SpikeSourceArray
- 7/11 - 7/24
    - Implement BKG neuron
    - Test model by inputting video stimuli and analyzing the resulting activity across V1 network
- 7/25 - 8/07
    - Validate model by comparing to BMTK/NEST simulation and in vivo experimental data
    - Benchmark model’s performance against BMTK/NEST
- 8/08 - 8/21 
    - Develop webcam application that simulates V1 activity in real-time in response to webcam stream
- 8/22 - 9/04
    - Complete preceding tasks if there were any unpredicted delays (this period serves as a two-week buffer)
    - Construct GPU-accelerated implementation of filternet LGN model
- 9/04 - 9/12
    - Finish documentation of project
    - Publish repository containing the model and the complete steps to reproduce it
