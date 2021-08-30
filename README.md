# Scheduled Sampling Based on Decoding Steps for Neural Machine Translation (EMNLP-2021 main conference)

## Contents

* [Overview](#overview)
* [Background](#background)
* [Quick to Use](#quick-to-use)
* [Further Usage](#further-usage)
* [Experiments](#experiments)
* [Citation](#citation)
* [Contact](#contact)



## Overview

We propose to conduct [scheduled sampling](https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf) based on decoding steps instead of the original training steps. We observe that our proposal can more realistically simulate the distribution of real translation errors, thus better bridging the gap between training and inference. The paper[a link] has been accepted to the main conference of EMNLP-2021.


## Background

<p align="center">
  <img src="https://github.com/Adaxry/ss_on_decoding_steps/blob/main/figures/ss_for_transformer.png" alt="fastText" width="600"/>
</p>

We conduct scheduled sampling for the Transformer with a [two-pass decoder](https://aclanthology.org/P19-2049/). An example of pseudo code is as follows:    
```python
# first-pass: the same as the standard Transformer decoder
first_decoder_outputs = decoder(first_decoder_inputs)

# sampling tokens between model predicitions and ground-truth tokens
second_decoder_inputs = sampling_function(first_decoder_outputs, first_decoder_inputs)

# second-pass: computing the decoder again with the above sampled tokens
second_decoder_outputs = decoder(second_decoder_inputs)

```

## Quick to Use

Our approaches are suitable for most autoregressive-based tasks, Please feel free to try the following pseudo codes when conducting scheduled sampling:


```python
import torch

def sampling_function(first_decoder_outputs, first_decoder_inputs, max_seq_len, tgt_lengths)
    '''
    conduct scheduled sampling based on the index of decoded tokens 
    param first_decoder_outputs: [batch_size, seq_len, hidden_size], model prediections 
    param first_decoder_inputs: [batch_size, seq_len, hidden_size], ground-truth target tokens
    param max_seq_len: scalar, the max lengh of target sequence
    param tgt_lengths: [batch_size], the lenghs of target sequences in a mini-batch
    '''

    # indexs of decoding steps
    t = torch.range(0, max_seq_len-1)

    # differenct sampling strategy based on decoding steps
    if sampling_strategy == "exponential":
        threshold_table = exp_radix ** t  
    elif sampling_strategy == "sigmoid":
        threshold_table = sigmoid_k / (sigmoid_k + torch.exp(t / sigmoid_k ))
    elif sampling_strategy == "linear":        
        threshold_table = torch.max(epsilon, 1 - t / max_seq_len)
    else:
        ValuraiseeError("Unknown sampling_strategy %s" % sampling_strategy)

    # convert threshold_table to [batch_size, seq_len]
    threshold_table = threshold_table.unsqueeze_(0).repeat(max_seq_len, 1).tril()
    thresholds = threshold_table[tgt_lengths].view(-1, max_seq_len)
    thresholds = current_thresholds[:, :seq_len]

    # conduct sampling based on the above thresholds
    random_select_seed = torch.rand([batch_size, seq_len]) 
    second_decoder_inputs = torch.where(random_select_seed < thresholds, first_decoder_inputs, first_decoder_outputs)

    return second_decoder_inputs
    
```
## Further Usage

Error accumulation is a common phenomenon in NLP tasks. Whenever you want to simulate the accumulation of errors, our method may come in handy. For examples:

+ [Target Denoising](http://www.statmt.org/wmt20/pdf/2020.wmt-1.24.pdf)

```python
# sampling tokens between noisy target tokens and ground-truth tokens
decoder_inputs = sampling_function(noisy_decoder_inputs, golden_decoder_inputs, max_seq_len, tgt_lengths)

# computing the decoder with the above sampled tokens
decoder_outputs = decoder(decoder_inputs)

```

+ [Multi-turn Dialogue](https://arxiv.org/abs/1506.08909)

```python
# sampling utterences from model predictions and ground-truth utterences
contexts = sampling_function(predicted_utterences, golden_utterences, max_turns, current_turns)

model_predictions = dialogue_model(contexts, target_inputs)
```


## Experiments
We provide scripts to reproduce the results in this paper([NMT]() and [text summarization]())


## Citation
Please cite this paper if you find this repo useful.
```
@inproceedings{liu_ss_decoding_2021,
    title = "Scheduled Sampling Based on Decoding Steps for Neural Machine Translation",
    author = "Liu, Yijin  and
      Meng, Fandong  and
      Chen, Yufeng  and
      Xu, Jinan  and
      Zhou, Jie",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2021",
    address = "Online"
}
```

## Contact
Please feel free to contact us (yijinliu@tencent.com) for any further questions.


