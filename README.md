

## Installation

`conda create -f env.yml`

`python -m spacy download en_core_web_sm`

## Running
```python
./realmain.sh
```

Add route to the remote camera (Get this issue when there are multiple network devices connected)
```
sudo ip route add 192.168.1.149 dev wlp0s20f3
```

sshpass -p corallab-151 ssh -t corallab@192.168.1.149 

## Common Issues:
CUDA out of memory?
```
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:50"
```

RuntimeError: Timeout connecting to UR dashboard server.
```
Toggle the UR5 Wired Preset off and on
``

RuntimeError: One of the RTDE input registers are already in use! Currently you must disable the EtherNet/IP adapter, PROFINET or any MODBUS unit configured on the robot. This might change in the future.
```
I don't know
```

RuntimeError: ur_rtde: Failed to start control script, before timeout of 5 seconds
```
Wait 3 seconds, then try again. Make sure there are no errors on the tablet
```