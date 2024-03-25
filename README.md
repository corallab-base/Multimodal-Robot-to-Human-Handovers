

## Installation
`python -m spacy download en_core_web_sm`

## Running
```python
python main.py --real True --gaze False --prompt "Give me the red cup"
```

## Tips:
```
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:50"
```