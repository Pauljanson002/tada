#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python -m apps.run --multirun +experiment=dg_native_resol_low_view
CUDA_VISIBLE_DEVICES=0 python -m apps.run --multirun +experiment=dg_native_resol