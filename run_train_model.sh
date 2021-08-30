#!/bin/bash
# coding=utf-8
# Copyright 2021 South China University of Technology and 
# Engineering Research Ceter of Minstry of Education on Human Body Perception.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2021.08.30

# Train model with torch.distributed.launch by using torch.distributed.launch


# 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29500 train_model.py --model_type=GPT --lr=5.5e-3 --pretrained
