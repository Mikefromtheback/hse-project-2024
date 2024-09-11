# Incorrect code

```
(f'python3 /kaggle/usr/lib/prior/prior.py '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=1 '
           f'--num_train_epochs=100 '
           f'--mixed_precision="fp16" '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```
1.0 checkpoint  
prompt = orc with sword  
infer_steps=125
<p float="left">
  <img src="/character/images/1-1.jpg" width="100" />
  <img src="/character/images/1-2.jpg" width="100" /> 
  <img src="/character/images/1-3.jpg" width="100" />
  <img src="/character/images/1-4.jpg" width="100" />
  <img src="/character/images/1-5.jpg" width="100" />
</p>

0.1 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/2-1.jpg" width="100" />
  <img src="/character/images/2-2.jpg" width="100" /> 
  <img src="/character/images/2-3.jpg" width="100" />
  <img src="/character/images/2-4.jpg" width="100" />
  <img src="/character/images/2-5.jpg" width="100" />
</p>
0.5 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/3-1.jpg" width="100" />
  <img src="/character/images/3-2.jpg" width="100" /> 
  <img src="/character/images/3-3.jpg" width="100" />
  <img src="/character/images/3-4.jpg" width="100" />
  <img src="/character/images/3-5.jpg" width="100" />
</p>

```
(f'python3 /kaggle/usr/lib/prior/prior.py '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=1 '
           f'--num_train_epochs=100 '
           f'--mixed_precision="fp16" '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=64 '
          )
```
1.0 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/4-1.jpg" width="100" />
  <img src="/character/images/4-2.jpg" width="100" /> 
  <img src="/character/images/4-3.jpg" width="100" />
  <img src="/character/images/4-4.jpg" width="100" />
  <img src="/character/images/4-5.jpg" width="100" />
</p>
0.1 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/5-1.jpg" width="100" />
  <img src="/character/images/5-2.jpg" width="100" /> 
  <img src="/character/images/5-3.jpg" width="100" />
  <img src="/character/images/5-4.jpg" width="100" />
  <img src="/character/images/5-5.jpg" width="100" />
</p>
0.6 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/6-1.jpg" width="100" />
  <img src="/character/images/6-2.jpg" width="100" /> 
  <img src="/character/images/6-3.jpg" width="100" />
  <img src="/character/images/6-4.jpg" width="100" />
  <img src="/character/images/6-5.jpg" width="100" />
</p>

```
(f'python3 /kaggle/usr/lib/prior/prior.py '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=2 '
           f'--num_train_epochs=100 '
           f'--mixed_precision="fp16" '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```
1.0 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/7-1.jpg" width="100" />
  <img src="/character/images/7-2.jpg" width="100" /> 
  <img src="/character/images/7-3.jpg" width="100" />
  <img src="/character/images/7-4.jpg" width="100" />
  <img src="/character/images/7-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=100 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```
1.0 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/8-1.jpg" width="100" />
  <img src="/character/images/8-2.jpg" width="100" /> 
  <img src="/character/images/8-3.jpg" width="100" />
  <img src="/character/images/8-4.jpg" width="100" />
  <img src="/character/images/8-5.jpg" width="100" />
</p>

```
(
           f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=300 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```

1.0 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/9-1.jpg" width="100" />
  <img src="/character/images/9-2.jpg" width="100" /> 
  <img src="/character/images/9-3.jpg" width="100" />
  <img src="/character/images/9-4.jpg" width="100" />
  <img src="/character/images/9-5.jpg" width="100" />
</p>

```
(
           f'accelerate launch '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=100 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```
1.0 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/10-1.jpg" width="100" />
  <img src="/character/images/10-2.jpg" width="100" /> 
  <img src="/character/images/10-3.jpg" width="100" />
  <img src="/character/images/10-4.jpg" width="100" />
  <img src="/character/images/10-5.jpg" width="100" />
</p>

```
(
           f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=300 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```

0.2 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/11-1.jpg" width="100" />
  <img src="/character/images/11-2.jpg" width="100" /> 
  <img src="/character/images/11-3.jpg" width="100" />
  <img src="/character/images/11-4.jpg" width="100" />
  <img src="/character/images/11-5.jpg" width="100" />
</p>
0.4 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/12-1.jpg" width="100" />
  <img src="/character/images/12-2.jpg" width="100" /> 
  <img src="/character/images/12-3.jpg" width="100" />
  <img src="/character/images/12-4.jpg" width="100" />
  <img src="/character/images/12-5.jpg" width="100" />
</p>
0.6 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/13-1.jpg" width="100" />
  <img src="/character/images/13-2.jpg" width="100" /> 
  <img src="/character/images/13-3.jpg" width="100" />
  <img src="/character/images/13-4.jpg" width="100" />
  <img src="/character/images/13-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=400 '
           f'--learning_rate=5e-5 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 
```
dead

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=5e-5 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```
1.0 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/14-1.jpg" width="100" />
  <img src="/character/images/14-2.jpg" width="100" /> 
  <img src="/character/images/14-3.jpg" width="100" />
  <img src="/character/images/14-4.jpg" width="100" />
  <img src="/character/images/14-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=5e-5 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="constant" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```
1.0 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/15-1.jpg" width="100" />
  <img src="/character/images/15-2.jpg" width="100" /> 
  <img src="/character/images/15-3.jpg" width="100" />
  <img src="/character/images/15-4.jpg" width="100" />
  <img src="/character/images/15-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=1 '
           f'--num_train_epochs=100 '
           f'--learning_rate=1e-2 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="constant" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```
1.0 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/16-1.jpg" width="100" />
  <img src="/character/images/16-2.jpg" width="100" /> 
  <img src="/character/images/16-3.jpg" width="100" />
  <img src="/character/images/16-4.jpg" width="100" />
  <img src="/character/images/16-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/decoder/decoder.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=100 '
           f'--learning_rate=5e-5 '
           f'--gradient_checkpointing '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 ' 
          )
```
1.0 checkpoint <br> 
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/17-1.jpg" width="100" />
  <img src="/character/images/17-2.jpg" width="100" /> 
  <img src="/character/images/17-3.jpg" width="100" />
  <img src="/character/images/17-4.jpg" width="100" />
  <img src="/character/images/17-5.jpg" width="100" />
</p>

```
nothing
```
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/18-1.jpg" width="100" />
  <img src="/character/images/18-2.jpg" width="100" /> 
  <img src="/character/images/18-3.jpg" width="100" />
  <img src="/character/images/18-4.jpg" width="100" />
  <img src="/character/images/18-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=300 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
          )
```
1.0 checkpoint <br>
prompt = orc with sword <br>
infer_steps=125
<p float="left">
  <img src="/character/images/19-1.jpg" width="100" />
  <img src="/character/images/19-2.jpg" width="100" /> 
  <img src="/character/images/19-3.jpg" width="100" />
  <img src="/character/images/19-4.jpg" width="100" />
  <img src="/character/images/19-5.jpg" width="100" />
</p>

# Outdated code
```
(f'accelerate launch  --multi_gpu '
           f'diffusers/examples/kandinsky2_2_train/tune_decoder_lora.py '
           f'--train_images_paths_csv=captions.csv '
           f'--image_resolution=512 '
           f'--train_batch_size=2 '
           f'--gradient_accumulation_steps=1 '
           f'--gradient_checkpointing '
           f'--mixed_precision="fp16"  '
           f'--max_train_steps=5000 '
           f'--lr=1e-05 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="constant" '
           f'--lr_warmup_steps=0 '
           f'--output_dir=decoder_lora_saves '
           f'--rank=4 '
           f'--checkpointing_steps=1000')
           
(f'accelerate launch  --multi_gpu '
           f'diffusers/examples/kandinsky2_2_train/tune_prior_lora.py '
           f'--train_images_paths_csv=captions.csv '
           f'--train_batch_size=2 '
           f'--gradient_accumulation_steps=1 '
           f'--gradient_checkpointing '
           f'--mixed_precision="fp16"  '
           f'--max_train_steps=5000 '
           f'--lr=1e-05 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="constant" '
           f'--lr_warmup_steps=0 '
           f'--output_dir=decoder_prior_saves '
           f'--rank=4 '
           f'--checkpointing_steps=1000')
```
0.8 checkpoint <br>
prompt = orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/20-1.jpg" width="100" />
  <img src="/character/images/20-2.jpg" width="100" /> 
  <img src="/character/images/20-3.jpg" width="100" />
  <img src="/character/images/20-4.jpg" width="100" />
  <img src="/character/images/20-5.jpg" width="100" />
</p>

0.8 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/21-1.jpg" width="100" />
  <img src="/character/images/21-2.jpg" width="100" /> 
  <img src="/character/images/21-3.jpg" width="100" />
  <img src="/character/images/21-4.jpg" width="100" />
  <img src="/character/images/21-5.jpg" width="100" />
</p>

0.8 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=25
<p float="left">
  <img src="/character/images/22-1.jpg" width="100" />
  <img src="/character/images/22-2.jpg" width="100" /> 
  <img src="/character/images/22-3.jpg" width="100" />
  <img src="/character/images/22-4.jpg" width="100" />
  <img src="/character/images/22-5.jpg" width="100" />
</p>

0.2 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/23-1.jpg" width="100" />
  <img src="/character/images/23-2.jpg" width="100" /> 
  <img src="/character/images/23-3.jpg" width="100" />
  <img src="/character/images/23-4.jpg" width="100" />
  <img src="/character/images/23-5.jpg" width="100" />
</p>

# Correct code
```
 f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=100 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr 
           
 f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/decoder/decoder.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=100 '
           f'--learning_rate=5e-5 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
```
1.0 checkpoint <br>
prompt = orc with sword <br>
infer_steps=100
<p float="left">
  <img src="/character/images/24-1.jpg" width="100" />
  <img src="/character/images/24-2.jpg" width="100" /> 
  <img src="/character/images/24-3.jpg" width="100" />
  <img src="/character/images/24-4.jpg" width="100" />
  <img src="/character/images/24-5.jpg" width="100" />
</p>

1.0 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=100
<p float="left">
  <img src="/character/images/25-1.jpg" width="100" />
  <img src="/character/images/25-2.jpg" width="100" /> 
  <img src="/character/images/25-3.jpg" width="100" />
  <img src="/character/images/25-4.jpg" width="100" />
  <img src="/character/images/25-5.jpg" width="100" />
</p>

0.5 checkpoint <br>
prompt = orc with sword <br>
infer_steps=100
<p float="left">
  <img src="/character/images/26-1.jpg" width="100" />
  <img src="/character/images/26-2.jpg" width="100" /> 
  <img src="/character/images/26-3.jpg" width="100" />
  <img src="/character/images/26-4.jpg" width="100" />
  <img src="/character/images/26-5.jpg" width="100" />
</p>

0.5 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=100
<p float="left">
  <img src="/character/images/27-1.jpg" width="100" />
  <img src="/character/images/27-2.jpg" width="100" /> 
  <img src="/character/images/27-3.jpg" width="100" />
  <img src="/character/images/27-4.jpg" width="100" />
  <img src="/character/images/27-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=32'
)
```
1.0 checkpoint <br>
prompt = orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/28-1.jpg" width="100" />
  <img src="/character/images/28-2.jpg" width="100" /> 
  <img src="/character/images/28-3.jpg" width="100" />
  <img src="/character/images/28-4.jpg" width="100" />
  <img src="/character/images/28-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=32'
)
```
1.0 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/29-1.jpg" width="100" />
  <img src="/character/images/29-2.jpg" width="100" /> 
  <img src="/character/images/29-3.jpg" width="100" />
  <img src="/character/images/29-4.jpg" width="100" />
  <img src="/character/images/29-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=32'
)
```
0.5 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/30-1.jpg" width="100" />
  <img src="/character/images/30-2.jpg" width="100" /> 
  <img src="/character/images/30-3.jpg" width="100" />
  <img src="/character/images/30-4.jpg" width="100" />
  <img src="/character/images/30-5.jpg" width="100" />
</p>

0.75 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/31-1.jpg" width="100" />
  <img src="/character/images/31-2.jpg" width="100" /> 
  <img src="/character/images/31-3.jpg" width="100" />
  <img src="/character/images/31-4.jpg" width="100" />
  <img src="/character/images/31-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8'
)
```

1.0 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/32-1.jpg" width="100" />
  <img src="/character/images/32-2.jpg" width="100" /> 
  <img src="/character/images/32-3.jpg" width="100" />
  <img src="/character/images/32-4.jpg" width="100" />
  <img src="/character/images/32-5.jpg" width="100" />
</p>

0.5 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/33-1.jpg" width="100" />
  <img src="/character/images/33-2.jpg" width="100" /> 
  <img src="/character/images/33-3.jpg" width="100" />
  <img src="/character/images/33-4.jpg" width="100" />
  <img src="/character/images/33-5.jpg" width="100" />
</p>

0.75 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/34-1.jpg" width="100" />
  <img src="/character/images/34-2.jpg" width="100" /> 
  <img src="/character/images/34-3.jpg" width="100" />
  <img src="/character/images/34-4.jpg" width="100" />
  <img src="/character/images/34-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=16'
)
```
1.0 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/35-1.jpg" width="100" />
  <img src="/character/images/35-2.jpg" width="100" /> 
  <img src="/character/images/35-3.jpg" width="100" />
  <img src="/character/images/35-4.jpg" width="100" />
  <img src="/character/images/35-5.jpg" width="100" />
</p>

0.5 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/36-1.jpg" width="100" />
  <img src="/character/images/36-2.jpg" width="100" /> 
  <img src="/character/images/36-3.jpg" width="100" />
  <img src="/character/images/36-4.jpg" width="100" />
  <img src="/character/images/36-5.jpg" width="100" />
</p>

0.75 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/37-1.jpg" width="100" />
  <img src="/character/images/37-2.jpg" width="100" /> 
  <img src="/character/images/37-3.jpg" width="100" />
  <img src="/character/images/37-4.jpg" width="100" />
  <img src="/character/images/37-5.jpg" width="100" />
</p>

0.25 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/38-1.jpg" width="100" />
  <img src="/character/images/38-2.jpg" width="100" /> 
  <img src="/character/images/38-3.jpg" width="100" />
  <img src="/character/images/38-4.jpg" width="100" />
  <img src="/character/images/38-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8'
)
```
0.25 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/39-1.jpg" width="100" />
  <img src="/character/images/39-2.jpg" width="100" /> 
  <img src="/character/images/39-3.jpg" width="100" />
  <img src="/character/images/39-4.jpg" width="100" />
  <img src="/character/images/39-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/prior/prior.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=1e-4 '
           f'--max_grad_norm=1 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=4'
)
```
1.0 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/40-1.jpg" width="100" />
  <img src="/character/images/40-2.jpg" width="100" /> 
  <img src="/character/images/40-3.jpg" width="100" />
  <img src="/character/images/40-4.jpg" width="100" />
  <img src="/character/images/40-5.jpg" width="100" />
</p>
0.5 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/41-1.jpg" width="100" />
  <img src="/character/images/41-2.jpg" width="100" /> 
  <img src="/character/images/41-3.jpg" width="100" />
  <img src="/character/images/41-4.jpg" width="100" />
  <img src="/character/images/41-5.jpg" width="100" />
</p>
0.75 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/42-1.jpg" width="100" />
  <img src="/character/images/42-2.jpg" width="100" /> 
  <img src="/character/images/42-3.jpg" width="100" />
  <img src="/character/images/42-4.jpg" width="100" />
  <img src="/character/images/42-5.jpg" width="100" />
</p>
0.25 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/43-1.jpg" width="100" />
  <img src="/character/images/43-2.jpg" width="100" /> 
  <img src="/character/images/43-3.jpg" width="100" />
  <img src="/character/images/43-4.jpg" width="100" />
  <img src="/character/images/43-5.jpg" width="100" />
</p>

```
saved prior 100 epoch rank 8

f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/decoder/decoder.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=5e-5 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 '
```

1.0 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/44-1.jpg" width="100" />
  <img src="/character/images/44-2.jpg" width="100" /> 
  <img src="/character/images/44-3.jpg" width="100" />
  <img src="/character/images/44-4.jpg" width="100" />
  <img src="/character/images/44-5.jpg" width="100" />
</p>
0.5 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/45-1.jpg" width="100" />
  <img src="/character/images/45-2.jpg" width="100" /> 
  <img src="/character/images/45-3.jpg" width="100" />
  <img src="/character/images/45-4.jpg" width="100" />
  <img src="/character/images/45-5.jpg" width="100" />
</p>

0.25 checkpoint <br>
prompt = D&D character, orc with sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/46-1.jpg" width="100" />
  <img src="/character/images/46-2.jpg" width="100" /> 
  <img src="/character/images/46-3.jpg" width="100" />
  <img src="/character/images/46-4.jpg" width="100" />
  <img src="/character/images/46-5.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/decoder/decoder.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=200 '
           f'--learning_rate=5e-5 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=8 ' 
```
0.25 checkpoint <br>
prompt = D&D character, orc a man with a sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/47-1.jpg" width="100" />
  <img src="/character/images/47-2.jpg" width="100" /> 
  <img src="/character/images/47-3.jpg" width="100" />
  <img src="/character/images/47-4.jpg" width="100" />
  <img src="/character/images/47-5.jpg" width="100" />
</p>

0.5 checkpoint <br>
prompt = D&D character, orc a man with a sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/48-1.jpg" width="100" />
  <img src="/character/images/48-2.jpg" width="100" /> 
  <img src="/character/images/48-3.jpg" width="100" />
  <img src="/character/images/48-4.jpg" width="100" />
  <img src="/character/images/48-5.jpg" width="100" />
</p>

0.75 checkpoint <br>
prompt = D&D character, orc a man with a sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/49-1.jpg" width="100" />
  <img src="/character/images/49-2.jpg" width="100" /> 
  <img src="/character/images/49-3.jpg" width="100" />
  <img src="/character/images/49-4.jpg" width="100" />
  <img src="/character/images/49-5.jpg" width="100" />
</p>

with any of the checkpoints above saved prior 8 rank <br>
prompt = D&D character, orc a man with a sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/50-1.jpg" width="100" />
</p>

```
f'accelerate launch  --multi_gpu '
           f'/kaggle/usr/lib/decoder/decoder.py '
           f'--mixed_precision="fp16" '
           f'--train_data_dir="/kaggle/working/images" '
           f'--train_batch_size=3 '
           f'--num_train_epochs=100 '
           f'--learning_rate=5e-5 '
           f'--lr_scheduler="cosine_with_restarts" '
           f'--snr_gamma=5 '
           f'--use_8bit_adam '
           f'--scale_lr '
           f'--rank=32 ' 
```

1 checkpoint <br>
prompt = D&D character, orc a man with a sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/51-1.jpg" width="100" />
  <img src="/character/images/51-2.jpg" width="100" /> 
  <img src="/character/images/51-3.jpg" width="100" />
  <img src="/character/images/51-4.jpg" width="100" />
  <img src="/character/images/51-5.jpg" width="100" />
</p>

0.5 checkpoint <br>
prompt = D&D character, orc a man with a sword <br>
infer_steps=300
<p float="left">
  <img src="/character/images/52-1.jpg" width="100" />
  <img src="/character/images/52-2.jpg" width="100" /> 
  <img src="/character/images/52-3.jpg" width="100" />
  <img src="/character/images/52-4.jpg" width="100" />
  <img src="/character/images/52-5.jpg" width="100" />
</p>