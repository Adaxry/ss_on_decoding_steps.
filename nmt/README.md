

## NMT

### Requirements
+ Tensorflow version >= 1.12
+ Python version >= 3.5


### Datasets
+ [WMT14 EN-DE](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh)
+ [WMT14 EN-FR](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2fr.sh)
+ [WMT19 ZH-EN](http://www.statmt.org/wmt19/translation-task.html) 
+ The WMT14 EN-DE and EN-FR can be obtained from the above scripts. The WMT19 ZH-EN dataset may be slightly different with others due to different Chinese word segmentation tools. Our pre-processed data is available at this [link](https://drive.google.com/file/d/1LvUPsIZ_xRwuB1vHlvi1COeZEOxfbYy0/view?usp=sharing)

### Pre-training

```
code_dir=./THUMT
data_dir=path2yourdata
work_dir=./
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export PYTHONPATH=$work_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

signature=pretrained_model_name

output_dir=$work_dir/train/$signature
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $output_dir \
  --input $data_dir/train.src $data_dir/train.trg \
  --vocabulary $data_dir/dict.src.txt $data_dir/dict.trg.txt \
  --parameters=device_list=[0,1,2,3,4,5,6,7],eval_steps=90000000,train_steps=100000,batch_size=4096,max_length=128,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,num_encoder_layers=6,num_decoder_layers=6,layer_preprocess=none,layer_postprocess=layer_norm,update_cycle=1,hidden_size=512,filter_size=2048,num_heads=8,label_smoothing=0.1,warmup_steps=4000,learning_rate=1.0,save_checkpoint_steps=5000,keep_checkpoint_max=200,position_info_type=absolute,shared_embedding_and_softmax_weights=True,shared_source_target_embedding=True
```


### Finetuning

+ 'Sampling Based on Decoding Steps'

```
code_dir=./THUMT-decoding_step_sampling_absolute
work_dir=./
data_dir=path2yourdata

export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

signature=finetune_model_name

output_dir=$work_dir/train/$signature
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $output_dir \
  --input $data_dir/train.src $data_dir/train.trg \
  --vocabulary $data_dir/dict.src.txt $data_dir/dict.trg.txt \
  --checkpoint $work_dir/train/pretrained_model_name
  --parameters=device_list=[0,1,2,3,4,5,6,7],eval_steps=90000000,train_steps=300000,batch_size=4096,max_length=128,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,num_encoder_layers=6,num_decoder_layers=6,layer_preprocess=none,layer_postprocess=layer_norm,update_cycle=1,hidden_size=512,filter_size=2048,num_heads=8,label_smoothing=0.1,warmup_steps=4000,learning_rate=1.0,save_checkpoint_steps=5000,keep_checkpoint_max=200,position_info_type=absolute,shared_embedding_and_softmax_weights=True,shared_source_target_embedding=True,mle_rate=0,zero_step=False,timestep_scheduled_sampling_strategy=exp,timestep_exp_epsilon=0.99
```


+ 'Sampling Based on Both Training Steps and Decoding Steps'

```
code_dir=./THUMT-schedule_composite
work_dir=./
data_dir=path2yourdata

export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

signature=finetune_model_name

output_dir=$work_dir/train/$signature
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $output_dir \
  --input $data_dir/train.src $data_dir/train.trg \
  --vocabulary $data_dir/dict.src.txt $data_dir/dict.trg.txt \
  --checkpoint $work_dir/train/pretrained_model_name
  --parameters=device_list=[0,1,2,3,4,5,6,7],eval_steps=90000000,train_steps=300000,batch_size=4096,max_length=128,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,num_encoder_layers=6,num_decoder_layers=6,layer_preprocess=none,layer_postprocess=layer_norm,update_cycle=1,hidden_size=512,filter_size=2048,num_heads=8,label_smoothing=0.1,warmup_steps=4000,learning_rate=1.0,save_checkpoint_steps=5000,keep_checkpoint_max=200,position_info_type=absolute,shared_embedding_and_softmax_weights=True,shared_source_target_embedding=True,mle_rate=0,zero_step=False,trainstep_scheduled_sampling_strategy=sigmoid,timestep_scheduled_sampling_strategy=exp,timestep_exp_epsilon=0.99,trainstep_sigmoid_k=20000
```


### Inference and Evaluation



```
code_dir=./THUMT
work_dir=./
data_dir=path2yourdata
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
signature=finetune_model_name

output_dir=$work_dir/results/$signature
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

test_names='your_test_name' # e.g., newstest2014

for idx in `seq 5000 5000 300000`; do
  echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/train/$signature/checkpoint
  python $work_dir/$code_dir/thumt/bin/translator.py \
    --models transformer \
    --input $data_dir/test.src \
    --output $output_dir/"$test_names".out.$idx \
    --vocabulary $data_dir/dict.src.txt $data_dir/dict.trg.txt \
    --checkpoints $work_dir/train/$signature \
    --parameters=device_list=[0],decode_alpha=0.6,decode_batch_size=128
  echo evaluating with checkpoint-$idx
  cd $output_dir
  # uncomment one of the bellow three lines according to your datasets
  # sh ./evaluation/ende/eval_test14.sh "$test_names".out.$idx  # eval for WMT14-ENDE
  # sh ./evaluation/enfr/eval_test14.sh "$test_names".out.$idx  # eval for WMT14-ENFR
  # sh ./evaluation/zhen/eval_test19.sh "$test_names".out.$idx  # eval for WMT19-ZHEN
  cat $output_dir/"$test_names".out.$idx.delbpe.atat.eval
done
```




