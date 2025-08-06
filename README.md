# S3BIR

### Environment Setup
```bash
conda create -n s3bir python=3.12
conda activate s3bir
pip install -r requirements.txt
```

### Download Pretrained Weights
Download the pretrained weights from the [official DINOv2 repository](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)

### Training

```bash
python -m experiments.LN_prompt --exp_name=exp_dinov2_flickr --n_prompts=3 --LN_lr=1e-6 --prompt_lr=1e-4 --batch_size=32 --workers=16 --fg --txt_train ./dir/flickr25k/train/ --txt_test ./dir/flickr25k/val_flickr15k/ --gpu_id 0 --epochs 100 --encoder dinov2

python -m experiments.LN_prompt --exp_name=exp_clip_flickr --n_prompts=3 --LN_lr=1e-6 --prompt_lr=1e-4 --batch_size=32 --workers=16 --fg --txt_train ./dir/flickr25k/train/ --txt_test ./dir/flickr25k/val_flickr15k/ --gpu_id 0 --epochs 100 --encoder clip
```

### Evaluation

```bash
python SBIR.py --model ./saved_models/exp_clip_flickr/last.ckpt --output_file exp_clip_flickr --image_file ./dir/flickr15k/dataset.txt --sketch_file ./dir/flickr15k/query_class.txt --encoder clip --exp_name=exp_clip_flickr

python generate_images.py --model ./saved_models/exp_clip_flickr/last.ckpt --output_file exp_clip_flickr --image_file ./dir/flickr15k/dataset.txt --sketch_file ./dir/flickr15k/query_class.txt --encoder clip --exp_name=exp_clip_flickr

python3 mAP.py
```

Then, you can calculate the mAP using the provided `mAP.py`.

### Dataset example

You can download the [Flickr15k](https://drive.google.com/file/d/1iP2r6mMlP6NaCWjlfaFAlUwWVg9GT0-y/view?usp=sharing) and [Flickr25k](https://drive.google.com/file/d/1TXdbdbUxt3Rw5rYbA1SPZP0Ci2puVpgS/view?usp=sharing) and use the files in the `dir` folder as an example.