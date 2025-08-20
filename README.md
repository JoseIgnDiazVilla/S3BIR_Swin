# S3BIR

[![Paper (ScienceDirect)](https://img.shields.io/badge/Paper-ScienceDirect-blue)](https://www.sciencedirect.com/science/article/pii/S0167865525001527)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace-orange)](https://chstr-s3bir.hf.space/)
[![Dataset](https://img.shields.io/badge/Dataset-CVLab--UANDES-green)](https://deepcvl.ai/datasets.html)
[![Models](https://img.shields.io/badge/Models-GoogleDrive-yellow)](https://drive.google.com/drive/folders/1T1qp9o3AQG0aZl9R-JZEMLsKJiJbT916?usp=sharing)


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

You can download the datasets from [CVLab-UANDES](https://deepcvl.ai/datasets.html).

Pre-trained models are available on [Google Drive](https://drive.google.com/drive/folders/1T1qp9o3AQG0aZl9R-JZEMLsKJiJbT916?usp=sharing).

### Demo
You can try the demo on Hugging Face.
Check it out here: [Hugging Face Demo](https://chstr-s3bir.hf.space/)

## 📖 Citation

If you find this work useful, please cite our paper:
```bibtex
@article{SAAVEDRA202594,
  title   = {Achieving high performance on sketch-based image retrieval without real sketches for training},
  journal = {Pattern Recognition Letters},
  volume  = {193},
  pages   = {94-100},
  year    = {2025},
  issn    = {0167-8655},
  doi     = {https://doi.org/10.1016/j.patrec.2025.04.018},
  url     = {https://www.sciencedirect.com/science/article/pii/S0167865525001527},
  author  = {Jose M. Saavedra and Christopher Stears and Waldo Campos}
}
```