# Multimodal Human-Intent Modeling for Contextual Robot-to-Human Handovers of Arbitrary Objects

Lucas Chen∗, Guna Avula∗, Hanwen Ren, Zixing Wang and Ahmed H. Qureshi

This code release contains the software for our pipeline as well as analysis data. 

## Installation

`conda create -f env.yml`

`python -m spacy download en_core_web_sm`

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
