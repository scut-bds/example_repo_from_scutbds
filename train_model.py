# Copyright (c) 2020-present, South China University of Technology.
# All rights reserved.
# CPED Dataset Baseline Dialog Generation Model
# File: train_model.py
# Used for CPED dataset
# Author: Chen Yirong
# Date: 2021.01.04
# Through this python code file, You can train the following model on CPED Dataset:
# ['seq2seq', 'transformer', 'CDialGPT', 'EDACDialGPT', 'EDASDialGPT', 'REDialGPT', 'RDADialGPT','GPT_PER']
# by set the "--model_type"!
# Reference: 
# [1] https://github.com/thu-coai/CDial-GPT
# [2] https://github.com/huggingface/transfer-learning-conv-ai
# [3] https://github.com/huggingface/transformers


import os
import json
import math
import torch
import logging
import random
import numpy as np
from pprint import pformat
from argparse import ArgumentParser
from torch.optim.lr_scheduler import LambdaLR, CyclicLR, OneCycleLR
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTConfig, GPT2LMHeadModel, GPT2Config,
                          WEIGHTS_NAME, CONFIG_NAME, AdamW, BertTokenizer)
# import baseline model
from baseline.seq2seq import Seq2Seq
from baseline.transformer import Transformer
from baseline.EDACDialGPT import (EDACGPTLMHeadModel, EDACGPTConfig)
from baseline.EDASDialGPT import (EDASGPTLMHeadModel, EDASGPTConfig)
from baseline.REDialGPT import REGPTLMHeadModel
from baseline.RDADialGPT import RDAGPTLMHeadModel
from baseline.GPT_PER import GPTPERLMHeadModel
from baseline.GPT_EMO_PER import GPTEMOPERLMHeadModel
from baseline.GPT_EMO_PER_DA import GPTEMOPERDALMHeadModel


from utils.util import (build_dataloaders, build_edadial_dataloaders, build_edasdial_dataloaders, build_redial_dataloaders, build_cped_dataloaders,
                        build_rdadial_dataloaders, SENTIMENT_TOKENS, EMOTION_TOKENS, BASEEMOTION_TOKENS, DA_TOKENS,
                        DA_TO_TOKENS, SENTIMENT_TO_TOKENS, EMOTION_TO_TOKENS, BASEEMOTION_TO_TOKENS)

logger = logging.getLogger(__file__)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2021)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def train():
    parser = ArgumentParser()
    parser.add_argument('--gpt2', action='store_true', help="use gpt2")
    parser.add_argument("--model_type", type=str, default="seq2seq", choices=['seq2seq', 'transformer', 'CDialGPT', 'EDACDialGPT', 'EDASDialGPT', 'REDialGPT', 'RDADialGPT','GPT_PER','GPT_EMO_PER','GPT_EMO_PER_DA'], help="Type of Model")
    parser.add_argument("--model_checkpoint", type=str, default="./config/seq2seq", help="Path or URL of the model")
    parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    parser.add_argument('--pretrained', action='store_true', help="If False train from scratch")
    parser.add_argument("--data_path", type=str, default="/home/CPED",
                        help="Path or url of the dataset. ")
    parser.add_argument("--train_path", type=str, default="/home/CPED/train_split.csv",
                        help="Path of the train dataset for CPED dataset. ")
    parser.add_argument("--valid_path", type=str, default="/home/CPED/valid_split.csv",
                        help="Path of the valid dataset for CPED dataset. ")
    parser.add_argument("--test_path", type=str, default="/home/CPED/test_split.csv",
                        help="Path of the test dataset for CPED dataset. ")
    parser.add_argument("--dataset_cache", type=str, default="dataset_cache",
                        help="Path or url of the dataset cache")
    parser.add_argument('--log_file', '-log_file', type=str, default="./log", help="Output logs to a file under this path")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=70, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=2, help="Batch size for testing")
    parser.add_argument("--max_history", type=int, default=15, help="Number of previous exchanges to keep in history")
    parser.add_argument("--emotion_type", type=str, default="Emotion", choices=['Sentiment', 'BaseEmotion', 'Emotion'], help="Type of Emotion")
    parser.add_argument("--da_type", type=str, default="DA", choices=['DA', 'BaseDA'], help="Type of DA")
    parser.add_argument('--with_emotion', action='store_true', help="use emotion as token_type")
    parser.add_argument('--with_da', action='store_true', help="use da as token_type")
    parser.add_argument('--with_current_speaker', action='store_true', help="use current speaker as control signal")
    parser.add_argument('--with_current_persona', action='store_true', help="use current persona as control signal")
    parser.add_argument('--with_current_emotion', action='store_true', help="use current emotion as control signal")
    parser.add_argument('--with_current_da', action='store_true', help="use current da as control signal")
    parser.add_argument("--scheduler", type=str, default="noam", choices=['noam', 'linear', 'cyclic', '1cycle','fixedlr'], help="method of optim")
    parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process.
    # logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = BertTokenizer
    logger.info("Model Type is "+args.model_type)
    if args.model_type == 'EDACDialGPT':
        model_class = EDACGPTLMHeadModel
        config_class = EDACGPTConfig
    elif args.model_type == 'EDASDialGPT':
        model_class = EDASGPTLMHeadModel
        config_class = EDASGPTConfig
    elif args.model_type == 'CDialGPT':
        model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
        config_class = OpenAIGPTConfig if not args.gpt2 else GPT2Config
    elif args.model_type == 'REDialGPT':
         model_class = REGPTLMHeadModel
         config_class = OpenAIGPTConfig
    elif args.model_type == 'RDADialGPT':
         model_class = RDAGPTLMHeadModel
         config_class = OpenAIGPTConfig
    elif args.model_type == 'GPT_PER':
         model_class = GPTPERLMHeadModel
         config_class = OpenAIGPTConfig
    elif args.model_type == 'GPT_EMO_PER':
         model_class = GPTEMOPERLMHeadModel
         config_class = OpenAIGPTConfig
    elif args.model_type == 'GPT_EMO_PER_DA':
         model_class = GPTEMOPERDALMHeadModel
         config_class = OpenAIGPTConfig
    elif args.model_type == 'transformer':
        model_class = Transformer
    elif args.model_type == 'seq2seq':
        model_class = Seq2Seq
    # You can add your model here use elif
    else:
        logger.error("You set the wrong model_type:"+args.model_type)
        logger.error("Please set the --model_type from the choices: "+"[\'seq2seq\', \'transformer\', \'CDialGPT\', \'EDACDialGPT\', \'EDASDialGPT\', \'REDialGPT\', \'RDADialGPT\',\'GPT_PER\']")
        return

    if args.pretrained: # for pretrained model
        if args.model_type == 'transformer' or args.model_type == 'seq2seq':
            tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True)
            config = json.load(open(os.path.join(args.model_checkpoint, "config.json"), "r"))
            model = model_class(config)
            PATH = os.path.join(args.model_checkpoint,"pytorch_model.bin")
            model.load_state_dict(torch.load(PATH))
        else:
            tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True)
            model = model_class.from_pretrained(args.model_checkpoint)

    else:
        tokenizer = tokenizer_class(os.path.join(args.model_checkpoint, "vocab.txt"), do_lower_case=True)
        if args.model_type == 'transformer' or args.model_type == 'seq2seq':
            config = json.load(open(os.path.join(args.model_checkpoint, "config.json"), "r"))
            config["n_embd"] = args.n_emd
            if args.model_type == 'seq2seq': 
                config["hidden_size"] = args.n_emd
            model = model_class(config)
        else:
            config = config_class.from_json_file(os.path.join(args.model_checkpoint, CONFIG_NAME))
            model = model_class(config)
    if args.model_type == 'EDASDialGPT' or args.model_type == 'EDACDialGPT' or args.model_type == 'CDialGPT':
        logger.info("Add new tokens...")
    # SENTIMENT_TOKENS, EMOTION_TOKENS, BASEEMOTION_TOKENS, DA_TOKENS
    special_tokens_dict = {}
    special_tokens_list = []
    if args.model_type == 'EDASDialGPT' or args.model_type == 'EDACDialGPT':
        if args.with_emotion:
            if args.emotion_type == "Sentiment":
                #tokenizer.add_tokens(SENTIMENT_TOKENS)
                special_tokens_list.extend(SENTIMENT_TOKENS) 
                #tokenizer.add_special_tokens(SENTIMENT_TO_TOKENS)
            elif args.emotion_type == "BaseEmotion":
                #tokenizer.add_tokens(BASEEMOTION_TOKENS)
                special_tokens_list.extend(BASEEMOTION_TOKENS)
                #tokenizer.add_special_tokens(BASEEMOTION_TO_TOKENS)
            else:
                #tokenizer.add_tokens(EMOTION_TOKENS)
                special_tokens_list.extend(EMOTION_TOKENS)
                #tokenizer.add_special_tokens(EMOTION_TO_TOKENS)

        if args.with_da:
            #tokenizer.add_tokens(DA_TOKENS)
            #tokenizer.add_special_tokens(DA_TO_TOKENS)
            special_tokens_list.extend(DA_TOKENS)
        special_tokens_dict["additional_special_tokens"]=special_tokens_list
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))   # resize tokens embedding network
    if args.model_type == 'CDialGPT': # for EDialGPT and DADialGPT
        if args.with_emotion:
            if args.emotion_type == "Sentiment":
                special_tokens_list.extend(SENTIMENT_TOKENS)
            elif args.emotion_type == "BaseEmotion":
                special_tokens_list.extend(BASEEMOTION_TOKENS)
            else:
                special_tokens_list.extend(EMOTION_TOKENS)
        elif args.with_da:
            special_tokens_list.extend(DA_TOKENS)
        special_tokens_dict["additional_special_tokens"]=special_tokens_list
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))   # resize tokens embedding network        

    model.to(args.device)
    if args.model_type == 'transformer' or args.model_type == 'seq2seq':
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # ignore pad_id
    optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)

    logger.info("Prepare datasets")
    if args.model_type == 'EDASDialGPT':
        loader_class = build_edasdial_dataloaders
    elif args.model_type == 'EDACDialGPT':
        loader_class = build_edadial_dataloaders
    elif args.model_type == 'REDialGPT':
        loader_class = build_redial_dataloaders
    elif args.model_type == 'RDADialGPT':
        loader_class = build_rdadial_dataloaders
    elif args.model_type == 'GPT_PER' or args.model_type == 'GPT_EMO_PER' or args.model_type == 'GPT_EMO_PER_DA':
        loader_class = build_cped_dataloaders
    else:
        loader_class = build_dataloaders

    train_loader, val_loader, train_sampler, valid_sampler = loader_class(args, tokenizer, logger)
    test_loader, test_sampler = loader_class(args, tokenizer, logger, load_test=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        if args.model_type == 'transformer' or args.model_type == 'seq2seq':
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                        find_unused_parameters=True)   
            # Add "find_unused_parameters=True" to avoid the following error
            # ERROR:ignite.engine.engine.Engine:Current run is terminating due to exception: 
            # Expected to have finished reduction in the prior iteration before starting a new one. 
            # This error indicates that your module has parameters that were not used in producing loss.
            #  You can enable unused parameter detection by (1) passing the keyword argument 
            # `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`;
        else:
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        if args.model_type == 'EDASDialGPT' or args.model_type == 'EDACDialGPT':
            #input_ids, token_type_ids, emotion_ids, da_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, token_type_ids, emotion_ids, da_ids, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)
        
        if args.model_type == 'CDialGPT':
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)

        if args.model_type == 'REDialGPT':
            input_ids, token_type_ids, current_speaker_id, current_emotion_id, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)

        if args.model_type == 'RDADialGPT':
            input_ids, token_type_ids, current_speaker_id, current_da_id, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
        
        if args.model_type == 'GPT_PER' or args.model_type == 'GPT_EMO_PER' or args.model_type == 'GPT_EMO_PER_DA':
            input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)


        if args.model_type == 'transformer':
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
            context_ids = [input_ids[i][lm_labels[i] == -1] for i in range(input_ids.size(0))]
            context_ids = [context_ids[i][context_ids[i] != 1][: -1] for i in range(input_ids.size(0))]
            context_ids = pad_sequence(context_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            token_type_ids = token_type_ids[:, :context_ids.size(-1)]
            response_ids = [lm_labels[i][lm_labels[i] != -1] for i in range(input_ids.size(0))]

            label_ids = pad_sequence(response_ids, batch_first=True, padding_value=-1)
            label_ids = torch.cat(
                [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), label_ids],
                dim=-1)


            response_ids = pad_sequence(response_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            response_ids = torch.cat(
                [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), response_ids],
                dim=-1)
            lm_logits = model(context_ids, token_type_ids, response_ids[:, :-1])
            lm_logits = lm_logits.contiguous().view(-1, lm_logits.size(-1))
            loss = criterion(lm_logits, label_ids[:, 1:].contiguous().view(-1)) / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            if engine.state.iteration % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            return loss.item(), optimizer.param_groups[0]['lr']

        if args.model_type == 'seq2seq': # seq2seq
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
            context_ids = [input_ids[i][lm_labels[i] == -1] for i in range(input_ids.size(0))]
            context_ids = [context_ids[i][context_ids[i] != 1][: -1] for i in range(input_ids.size(0))]
            context_ids = pad_sequence(context_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            token_type_ids = token_type_ids[:, :context_ids.size(-1)]
            response_ids = [lm_labels[i][lm_labels[i] != -1] for i in range(input_ids.size(0))]

            label_ids = pad_sequence(response_ids, batch_first=True, padding_value=-1)
            label_ids = torch.cat(
                [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), label_ids],
                dim=-1)
            


            response_ids = pad_sequence(response_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            response_ids = torch.cat(
                [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), response_ids],
                dim=-1)
            lm_logits = model(context_ids, token_type_ids, response_ids[:, :-1])
            lm_logits = lm_logits.contiguous().view(-1, lm_logits.size(-1))
            loss = criterion(lm_logits, label_ids[:, 1:].contiguous().view(-1)) / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            if engine.state.iteration % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            return loss.item(), optimizer.param_groups[0]['lr']

        #print("input_ids, token_type_ids, lm_labels:",input_ids, token_type_ids, lm_labels)
        
        if args.model_type == 'EDASDialGPT' or args.model_type == 'EDACDialGPT':
            (lm_loss), *_ = model(input_ids, labels=lm_labels, token_type_ids=token_type_ids, emotion_ids=emotion_ids, da_ids=da_ids)
        elif args.model_type == 'CDialGPT':
            (lm_loss), *_ = model(input_ids, labels=lm_labels, token_type_ids=token_type_ids)
        elif args.model_type == 'REDialGPT':
            (lm_loss), *_ = model(input_ids, labels=lm_labels, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_emotion_id=current_emotion_id)
        elif args.model_type == 'RDADialGPT':
            (lm_loss), *_ = model(input_ids, labels=lm_labels, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_da_id=current_da_id)
        elif args.model_type == 'GPT_PER':
            (lm_loss), *_ = model(input_ids=input_ids, labels=lm_labels, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_persona_ids=current_persona_ids)
        elif args.model_type == 'GPT_EMO_PER':
            (lm_loss), *_ = model(input_ids=input_ids, labels=lm_labels, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_emotion_id=current_emotion_id, current_persona_ids=current_persona_ids)
        elif args.model_type == 'GPT_EMO_PER_DA':
            (lm_loss), *_ = model(input_ids=input_ids, labels=lm_labels, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_emotion_id=current_emotion_id, current_persona_ids=current_persona_ids, current_da_id=current_da_id)

        loss = lm_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item(), optimizer.param_groups[0]['lr']

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            if args.model_type == 'EDASDialGPT' or args.model_type == 'EDACDialGPT':
                #input_ids, token_type_ids, emotion_ids, da_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                input_ids, token_type_ids, emotion_ids, da_ids, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids, emotion_ids=emotion_ids, da_ids=da_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'CDialGPT':
                input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'REDialGPT':
                input_ids, token_type_ids, current_speaker_id, current_emotion_id, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_emotion_id=current_emotion_id)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'RDADialGPT':
                input_ids, token_type_ids, current_speaker_id, current_da_id, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_da_id=current_da_id)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'GPT_PER':
                input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids=input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_persona_ids=current_persona_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'GPT_EMO_PER':
                input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids=input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_emotion_id=current_emotion_id, current_persona_ids=current_persona_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'GPT_EMO_PER_DA':
                input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids=input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_emotion_id=current_emotion_id, current_persona_ids=current_persona_ids, current_da_id=current_da_id)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'transformer':
                input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                context_ids = [input_ids[i][lm_labels[i] == -1] for i in range(input_ids.size(0))]
                context_ids = [context_ids[i][context_ids[i] != 1][: -1] for i in range(input_ids.size(0))]
                context_ids = pad_sequence(context_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                token_type_ids = token_type_ids[:, :context_ids.size(-1)]
                response_ids = [lm_labels[i][lm_labels[i] != -1] for i in range(input_ids.size(0))]

                label_ids = pad_sequence(response_ids, batch_first=True, padding_value=-1)
                label_ids = torch.cat(
                    [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), label_ids],
                    dim=-1)

                response_ids = pad_sequence(response_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                response_ids = torch.cat(
                    [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), response_ids],
                    dim=-1)
                lm_logits = model(context_ids, token_type_ids, response_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = label_ids[..., 1:].contiguous().view(-1) 
                         
            if args.model_type == 'seq2seq':
                input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                context_ids = [input_ids[i][lm_labels[i] == -1] for i in range(input_ids.size(0))]
                context_ids = [context_ids[i][context_ids[i] != 1][: -1] for i in range(input_ids.size(0))]
                context_ids = pad_sequence(context_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                token_type_ids = token_type_ids[:, :context_ids.size(-1)]
                response_ids = [lm_labels[i][lm_labels[i] != -1] for i in range(input_ids.size(0))]

                label_ids = pad_sequence(response_ids, batch_first=True, padding_value=-1)
                label_ids = torch.cat(
                    [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), label_ids],
                    dim=-1)
                
                response_ids = pad_sequence(response_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                response_ids = torch.cat(
                    [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), response_ids],
                    dim=-1)
                lm_logits = model(context_ids, token_type_ids, response_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = label_ids[..., 1:].contiguous().view(-1)

            return lm_logits_flat_shifted, lm_labels_flat_shifted

    evaluator = Engine(inference)



    # Evaluation function and testor (testor output is the input of the metrics)
    def test(engine, batch):
        model.eval()
        with torch.no_grad():
            if args.model_type == 'EDASDialGPT' or args.model_type == 'EDACDialGPT':
                #input_ids, token_type_ids, emotion_ids, da_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                input_ids, token_type_ids, emotion_ids, da_ids, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids, emotion_ids=emotion_ids, da_ids=da_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'CDialGPT':
                input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'REDialGPT':
                input_ids, token_type_ids, current_speaker_id, current_emotion_id, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_emotion_id=current_emotion_id)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'RDADialGPT':
                input_ids, token_type_ids, current_speaker_id, current_da_id, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_da_id=current_da_id)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'GPT_PER':
                input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids=input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_persona_ids=current_persona_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'GPT_EMO_PER':
                input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids=input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_emotion_id=current_emotion_id, current_persona_ids=current_persona_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'GPT_EMO_PER_DA':
                input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)
                lm_logits, *_ = model(input_ids=input_ids, token_type_ids=token_type_ids, current_speaker_id=current_speaker_id, current_emotion_id=current_emotion_id, current_persona_ids=current_persona_ids, current_da_id=current_da_id)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            if args.model_type == 'transformer':
                input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                context_ids = [input_ids[i][lm_labels[i] == -1] for i in range(input_ids.size(0))]
                context_ids = [context_ids[i][context_ids[i] != 1][: -1] for i in range(input_ids.size(0))]
                context_ids = pad_sequence(context_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                token_type_ids = token_type_ids[:, :context_ids.size(-1)]
                response_ids = [lm_labels[i][lm_labels[i] != -1] for i in range(input_ids.size(0))]
                
                label_ids = pad_sequence(response_ids, batch_first=True, padding_value=-1)
                label_ids = torch.cat(
                    [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), label_ids],
                    dim=-1)

                response_ids = pad_sequence(response_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                response_ids = torch.cat(
                    [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), response_ids],
                    dim=-1)
                lm_logits = model(context_ids, token_type_ids, response_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = label_ids[..., 1:].contiguous().view(-1) 
                         
            if args.model_type == 'seq2seq':
                input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
                context_ids = [input_ids[i][lm_labels[i] == -1] for i in range(input_ids.size(0))]
                context_ids = [context_ids[i][context_ids[i] != 1][: -1] for i in range(input_ids.size(0))]
                context_ids = pad_sequence(context_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                token_type_ids = token_type_ids[:, :context_ids.size(-1)]
                response_ids = [lm_labels[i][lm_labels[i] != -1] for i in range(input_ids.size(0))]

                label_ids = pad_sequence(response_ids, batch_first=True, padding_value=-1)
                label_ids = torch.cat(
                    [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), label_ids],
                    dim=-1)

                response_ids = pad_sequence(response_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                response_ids = torch.cat(
                    [torch.tensor([[tokenizer.cls_token_id]] * input_ids.size(0), device=args.device), response_ids],
                    dim=-1)
                lm_logits = model(context_ids, token_type_ids, response_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = label_ids[..., 1:].contiguous().view(-1)

            return lm_logits_flat_shifted, lm_labels_flat_shifted

    testor = Engine(test)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: testor.run(test_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
        trainer.add_event_handler(Events.COMPLETED, lambda _: testor.run(test_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))
        trainer.add_event_handler(Events.COMPLETED, lambda _: testor.run(test_loader))

    # Evaluation during training
    @trainer.on(Events.ITERATION_STARTED)
    def log_iterations(engine):
        # if engine.state.iteration % max(int(0.1 * len(train_loader)), 1) == 0:
        if engine.state.iteration % args.valid_steps == 0:
            evaluator.run(val_loader)
            testor.run(test_loader)

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))
        testor.add_event_handler(Events.EPOCH_STARTED, lambda engine: test_sampler.set_epoch(engine.state.epoch))

    # noam decrease the learning rate
    # model_size = model.config.n_embd
    model_size = args.n_emd
    noam_lambda = lambda step: (
            model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
    noam_scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda, last_epoch=args.from_step)
    scheduler = LRScheduler(noam_scheduler)
    if args.scheduler == "linear":
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    if args.scheduler == "cyclic":
        scheduler = LRScheduler(CyclicLR(optimizer, args.lr/100, args.lr, step_size_up=500, step_size_down=2500, mode='triangular2', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1))
    if args.scheduler == "1cycle":
        scheduler = LRScheduler(OneCycleLR(optimizer, args.lr, total_steps=args.n_epochs, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=False, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1))
    if args.scheduler == "fixedlr":
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), args.lr)])

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lr")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
        metric.attach(testor, name)

    # On the main process: add progress bar, tensorboard, checkpoints
    # And save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True, mininterval=2)
        pbar.attach(trainer, metric_names=["loss", "lr"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        testor.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Test: %s" % pformat(testor.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=None, comment='_'+args.model_type)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                                                              another_engine=trainer),
                         event_name=Events.EPOCH_COMPLETED)

        tb_logger.attach(testor, log_handler=OutputHandler(tag="test", metric_names=list(metrics.keys()),
                                                              another_engine=trainer),
                         event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)
        # save model after evaluation
        testor.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
        if args.model_type == 'transformer' or args.model_type == 'seq2seq':
            json.dump(getattr(model, 'module', model).config, open(os.path.join(tb_logger.writer.logdir, CONFIG_NAME), "w"))
        else:
            getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
        #tokenizer.save_vocabulary(tb_logger.writer.logdir)
        # save the new tokens vacab
        tokenizer.save_pretrained(tb_logger.writer.logdir)
        with open(tb_logger.writer.logdir + "/training_args.json",'w',encoding='utf-8') as json_file:
            json.dump(pformat(args),json_file,ensure_ascii=False)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint
    # (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1],
                  os.path.join(tb_logger.writer.logdir,
                               WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
