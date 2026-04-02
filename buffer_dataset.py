# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""
import json
from typing import List, Union
import datasets
from fractions import Fraction
import verl.utils.torch_functional as verl_F
from verl import DataProto
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


def compute_response_mask(response_ids, attention_mask):
    response_length = response_ids.size(1)
    return attention_mask[:, -response_length:]

class BufferDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 4096)
        truncation = config.get("truncation", "error")

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_response_length = config.get("max_response_length", 3072)
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []


        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            try:
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            except Exception as e:
                print(f"Error reading parquet file {parquet_file}: {str(e)}")
                json_file = parquet_file.replace('.parquet', '.json')
                dataframe = pd.read_json(json_file, orient='records')
                dataframe = datasets.Dataset.from_pandas(dataframe)
            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            # if self.tokenizer.chat_template is not None:
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )
            # else:
            # self.dataframe = self.dataframe.filter(
            #     lambda doc: len(tokenizer(doc[prompt_key][0]['content']).input_ids) <= self.max_prompt_length,
            #     num_proc=self.num_workers,
            #     desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            # )
            
            print(f"filter dataset len: {len(self.dataframe)}")
        # self.prompts = self.dataframe[self.prompt_key]
        # for key in self.prompt_dict_keys:
        #     # type(x): pandas.core.series.Series
        #     # type(x[0]): numpy.ndarray
        #     # type(x[0][0]): dict
        #     try:
        #         self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
        #     except Exception:
        #         print(f"self.prompts={self.prompts}")
        #         raise
        # self.prompts = self.prompts.tolist()
        # self.responses = self.dataframe[self.response_key]
        # for key in self.response_dict_keys:
        #     try:
        #         self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
        #     except Exception:
        #         print(f"self.responses={self.responses}")
        #         raise
        # self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        row_dict: dict = self.dataframe[item]
        prompt = row_dict[self.prompt_key]
        response = row_dict[self.response_key]

        # apply chat template
        # prompt_chat = [{"role": "user", "content": prompt}]
        # string
        # if isinstance(prompt, str):
        #     prompt_chat_str = prompt
        # else:
        #     prompt_chat_str = prompt[0]['content']
        prompt_chat_str = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token
        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output.pop("input_ids")
        prompt_attention_mask = prompt_ids_output.pop("attention_mask")

        prompt_ids, prompt_attention_mask = verl_F.postprocess_data(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        prompt_ids = prompt_ids[0]
        prompt_attention_mask = prompt_attention_mask[0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output.pop("input_ids")
        response_attention_mask = response_ids_output.pop("attention_mask")

        response_length = response_ids.shape[1]

        response_ids, response_attention_mask = verl_F.postprocess_data(
            input_ids=response_ids,
            attention_mask=response_attention_mask,
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation=self.truncation,
        )

        response_ids = response_ids[0]
        response_attention_mask = response_attention_mask[0]

        prompt_length = prompt_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        # sequence_length = input_ids.shape[0]
        # if sequence_length < self.max_length:
        #     padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
        #     padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

        #     input_ids = torch.cat((input_ids, padded_input_ids))
        #     attention_mask = torch.cat((attention_mask, padded_attention_mask))
        # elif sequence_length > self.max_length:
        #     input_ids = input_ids[: self.max_length]
        #     attention_mask = attention_mask[: self.max_length]

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0
        index = row_dict.get("extra_info", {}).get("index", 0)

        return {
            "prompts": prompt_ids,
            "index": index,
            "responses": response_ids,
            "response_mask": loss_mask[-response_ids.shape[0]:],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

class BufferedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.buffer = []
        self.dataloader_iter = iter(self.dataloader)

    def start_new_epoch(self):
        """Reset for new epoch"""
        self.dataloader_iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.start_new_epoch()
            return next(self.dataloader_iter)

    def __len__(self):
        return len(self.dataloader)

    def add_to_buffer(self, samples):
        if len(self.buffer) == 0:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        return len(self.buffer)