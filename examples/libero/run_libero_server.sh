#!/bin/bash
which python

ENV_NAME=LIBERO
port=10000
cuda_device_id=5

CUDA_VISIBLE_DEVICES=$cuda_device_id python scripts/serve_policy.py \
    --env $ENV_NAME \
    --port $port