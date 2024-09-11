import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformers
from peft import LoraConfig, TaskType, get_peft_model, IA3Config
import logging
from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model

import pytorch_lightning as pl
from torchmetrics import AUROC

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers.logging.set_verbosity_error()

token = "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"

OPERA_CT_TARGET_MODULES = ["qkv", "proj"]
OPERA_CE_TARGET_MODULES = ['conv', 'fc', 'linear']
target_module_dict = {"operaCT": OPERA_CT_TARGET_MODULES, "operaCE": OPERA_CE_TARGET_MODULES}
LLM_TARGET_MODULES = ["q_proj", "v_proj"]
LLM_TARGET_MODULES_ALLPROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]

class FlattenHead(nn.Module):
    def __init__(self, nf, out_dim, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, out_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x, no_fc=False):
        x = self.flatten(x)
        if no_fc:
            return x
        x = self.linear(x)
        x = self.dropout(x)
        return x


class RespLLM(nn.Module):

    def __init__(self, configs):
        super(RespLLM, self).__init__()

        self.loss = nn.CrossEntropyLoss()
        self.n_cls = configs.n_cls
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        # self.patch_len = configs.patch_len
        # self.stride = configs.
        self.audio_peft = configs.audio_peft
        self.d_audio = configs.enc_dim
        self.patch_nums = configs.patch_nums
        self.head_nf = self.d_ff * self.patch_nums

        self.llm_peft = configs.llm_peft
        self.llm_lora_rank = configs.llm_lora_rank
        self.llm_lora_alpha = configs.llm_lora_alpha
        self.llm_lora_dropout = configs.llm_lora_dropout

        self.use_audio = configs.use_audio

        if configs.llm_model == 'llama':
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b') # 13.5G
            
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'llama2':
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b') # 13.5G
            model_id = "meta-llama/Llama-2-7b-chat-hf"
            # model_id = 'meta-llama/Llama-2-7b'
            self.llama_config = LlamaConfig.from_pretrained(model_id, token=token)
            self.tokenizer = LlamaTokenizer.from_pretrained(model_id, token=token)
            self.llm_model = LlamaModel.from_pretrained(model_id, token=token, config=self.llama_config)
        elif configs.llm_model == 'medalpaca':
            self.llama_config = LlamaConfig.from_pretrained("medalpaca/medalpaca-7b") # 13.5G
            
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == "OpenBioLLM":
            model_id = "aaditya/OpenBioLLM-Llama3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModel.from_pretrained(model_id)
            # self.llama_config = LlamaConfig.from_pretrained(model_id) # 13.5G
            
            # try:
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=True,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # except EnvironmentError:  # downloads model from HF is not already done
            #     print("Local model files not found. Attempting to download...")
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # try:
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=True
            #     )
            # except EnvironmentError:  # downloads the tokenizer from HF if not already done
            #     print("Local tokenizer files not found. Atempting to download them..")
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False
            #     )
        elif configs.llm_model == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModel.from_pretrained(model_id)
            # self.llama_config = LlamaConfig.from_pretrained(model_id) # 13.5G
            
            # try:
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # except EnvironmentError:  # downloads model from HF is not already done
            #     print("Local model files not found. Attempting to download...")
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # try:
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False
            #     )
            # except EnvironmentError:  # downloads the tokenizer from HF if not already done
            #     print("Local tokenizer files not found. Atempting to download them..")
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False
            #     )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            # self.gpt2_config.num_hidden_layers = configs.llm_layers
            # self.gpt2_config.output_attentions = True
            # self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            # self.bert_config.num_hidden_layers = configs.llm_layers
            # self.bert_config.output_attentions = True
            # self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'mistral':
            model_id = "mistralai/Mistral-7B-v0.1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModel.from_pretrained(model_id, token = token)
        elif configs.llm_model == 'phi':
            model_id = "microsoft/Phi-3.5-mini-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModel.from_pretrained(model_id, token = token)
        elif configs.llm_model == "gemma2B":
            # model_id = "google/gemma-2-2b-it"
            model_id = "google/gemma-2-2b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModel.from_pretrained(model_id, token = token)
        elif configs.llm_model == "gemma9B":
            # model_id = "google/gemma-2-9b-it"
            model_id = "google/gemma-2-9b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModel.from_pretrained(model_id, token = token)
        else:
            raise NotImplementedError('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        
        if self.llm_peft == "lora":
            self.peft_config = LoraConfig(
                r=self.llm_lora_rank, 
                lora_alpha=self.llm_lora_alpha, 
                lora_dropout=self.llm_lora_dropout,
                # target_modules=LLM_TARGET_MODULES_ALLPROJ,
            )
            if configs.llm_lora_allproj:
                self.peft_config = LoraConfig(
                    r=self.llm_lora_rank, 
                    lora_alpha=self.llm_lora_alpha, 
                    lora_dropout=self.llm_lora_dropout,
                    target_modules=LLM_TARGET_MODULES_ALLPROJ,
                )
            try:
                self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            except ValueError:
                print(self.llm_model)
                if configs.llm_model == "phi":
                    self.peft_config = LoraConfig(
                        r=self.llm_lora_rank, 
                        lora_alpha=self.llm_lora_alpha, 
                        lora_dropout=self.llm_lora_dropout,
                        target_modules=["qkv_proj"]
                    )
                else:
                    self.peft_config = LoraConfig(
                        r=self.llm_lora_rank, 
                        lora_alpha=self.llm_lora_alpha, 
                        lora_dropout=self.llm_lora_dropout,
                        target_modules=LLM_TARGET_MODULES
                    )
                self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            self.llm_model.print_trainable_parameters()
            print('LoRA Training LLM')
        elif self.llm_peft == "frozen":
            for param in self.llm_model.parameters():
                param.requires_grad = False
        else:
            return NotImplementedError("LLM fine-tuning mode undefined")
        
        if configs.audio_encoder == "operaCT":
            self.audio_encoder = initialize_pretrained_model(configs.audio_encoder).encoder

        if self.audio_peft == "frozen":
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print("freeze audio encoder")
        elif self.audio_peft == "full":
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = True
            self.audio_encoder.train()
            print("full model fine-tune audio encoder")
        else:
            # peft
            if self.audio_peft == "lora":
                peft_config = LoraConfig(
                    # task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=configs.audio_lora_rank, lora_alpha=32, lora_dropout=0.1,
                    target_modules=target_module_dict[configs.audio_encoder]
                )
            elif self.audio_peft == "IA3":
                peft_config = IA3Config(
                    target_modules=target_module_dict[configs.audio_encoder],
                    feedforward_modules=['proj']
                )
            else:
                return NotImplementedError("audio fine-tuning mode undefined")
            self.audio_encoder = get_peft_model(self.audio_encoder, peft_config)
            self.audio_encoder.print_trainable_parameters()
            
        

        if configs.aligner == "projection":
            self.aligner = nn.Linear(self.d_audio, self.d_llm)
        else:
            return NotImplementedError("aligner module undefined")
        
        self.head_dropout = configs.head_dropout
        self.output_projection = FlattenHead(self.head_nf, self.n_cls, head_dropout=self.head_dropout)

        self.print_trainable()

    def reinitialize_clf(self, n_cls):
        self.output_projection = FlattenHead(self.head_nf, n_cls, head_dropout=self.head_dropout)

    def print_trainable(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("total trainable parameters:", trainable_params)

    def reset_trainable(self):
        if self.llm_peft == "lora":
            for name, param in self.audio_encoder.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.llm_peft == "frozen":
            for param in self.llm_model.parameters():
                param.requires_grad = False
        
        for param in self.aligner.parameters():
            param.requires_grad = True
        
        if self.audio_peft == "frozen":
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        elif self.audio_peft == "full":
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
        
        for param in self.output_projection.parameters():
            param.requires_grad = True
        self.print_trainable()

    def forward(self, x_spectrogram, x_prompt, x_context, no_fc=False):

        if self.patch_nums == 1:
            x_enc = self.audio_encoder(x_spectrogram)
            # print(x_enc.shape)
            enc_out = self.aligner(x_enc)
            enc_out = enc_out.unsqueeze(dim=1)
        elif self.patch_nums == 64:
            x_enc = self.audio_encoder.forward_window(x_spectrogram)
            # print(x_enc.shape)
            enc_out = self.aligner(x_enc)
        else:
            raise NotImplementedError

        prompt = self.tokenizer(x_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        context = self.tokenizer(x_context, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        context_embeddings = self.llm_model.get_input_embeddings()(context.to(x_enc.device))  # (batch, prompt_token, dim)

        # print(prompt_embeddings.shape, enc_out.shape)

        if self.use_audio:
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings], dim=1)

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # print(dec_out.shape)
        dec_out = dec_out[:, :, :self.d_ff]
        # print(dec_out.shape)

        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # print(dec_out.shape)

        dec_out = self.output_projection(dec_out[:, :, -self.patch_nums:], no_fc=no_fc)
        # print(dec_out.shape)
        return dec_out
