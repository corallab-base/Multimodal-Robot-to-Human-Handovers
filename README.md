# Multimodal Human-Intent Modeling for Contextual Robot-to-Human Handovers of Arbitrary Objects

Lucas Chen∗, Guna Avula∗, Hanwen Ren, Zixing Wang and Ahmed H. Qureshi

This code release contains the software for our pipeline as well as analysis data. 

## Installation

`conda create -f env.yml`

`python -m spacy download en_core_web_sm`

Our pipeline is structured with a portable device running the realtime portion (sensor readings, fast computations, UR5 interfacing) and 
a larger compute machine handling the ![GLIP](https://github.com/microsoft/GLIP) model inference and ![CoGrasp](https://github.com/corallab-base/CoGrasp).

The server code can be found as a docker image ![here]

## Running
```python
./realmain.sh
```

## Troubleshooting

For some mobile devices (laptops, embedded processors) PyTorch may appear to have run out of memory.
```
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:50"
```

The UR library on the device side occasionally reports `RuntimeError: Timeout connecting to UR dashboard server.`
In this case, roggle the UR5 Wired Preset off and on

It may also say `RuntimeError: ur_rtde: Failed to start control script, before timeout of 5 seconds`

Wait 3 seconds, then try again. Make sure there are no errors on the tablet. This may occur due to estop or safety states.
