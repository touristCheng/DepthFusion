#!/usr/bin/env bash

root_path="/media/zexiang/ExtraData/codes/ucsnet/tanks_results"

root_path="/cephfs/shuocheng/tanks_results"

python fuse.py --root_path $root_path --save_path "./points" \
                --dist_thresh 0.001 --prob_thresh 0.6 --num_consist 10 --device "cuda"
