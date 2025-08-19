import argparse

parser = argparse.ArgumentParser(description='S3BIR CLIP')

parser.add_argument('--exp_name', type=str, default='clip_v1', help='Experiment name for logging and saving models')

# --------------------
# DataLoader Options
# --------------------

parser.add_argument('--data_dir', type=str, default='/media/chr/F43A32C43A3283A0/dataset_tesis/Sketchy/') #/home/chr/Sketch_LVM/dataset/Sketchy #/media/chr/F43A32C43A3283A0/dataset_tesis/doc_explore/dataset 
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# If fg is in the argument, the dataloader will be fine-grained
parser.add_argument('--fg', action='store_true', default=False)
parser.add_argument('--pidinet', action='store_true', default=False)
parser.add_argument('--txt_train', type=str, default='./dir/ecommerce/pairs/') 
parser.add_argument('--txt_test', type=str, default='./dir/ecommerce/')

# ----------------------
# Training Params
# ----------------------

# parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=12)
parser.add_argument('--model_type', type=str, default='one_encoder', choices=['one_encoder', 'two_encoder'])
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for training')
parser.add_argument('--encoder', type=str, default='clip', choices=['clip', 'dinov2', 'dinov3'])

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# ----------------------
# SBIR Parameters
# ----------------------
parser.add_argument('--model', type=str, default="/sketchy_models/epoch\=04-mAP\=0.73.ckpt", help='Path of the model to evaluate')
parser.add_argument('--output_file', type=str, default="example_results_model_1", help='Path to save the retrieval results')
parser.add_argument('--image_file', type=str, default="./dir/images.txt")
parser.add_argument('--sketch_file', type=str, default="./dir/sketches.txt")
opts = parser.parse_args()
