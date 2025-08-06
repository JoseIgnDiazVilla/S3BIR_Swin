import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

from torch.utils.data import Dataset

# Will depend on the dataset
unseen_classes = [
    # "knife",
    # "rabbit",
    # "ray",
    # "geyser",
    # "sword",
    # "songbird",
    # "ape",
    # "ant",
    # "crab",
    # "giraffe",
    # "snail",
    # "bell",
    # "bee",
    # "saxophone",
    # "rocket",
    # "apple",
    # "pig",
    # "squirrel",
    # "umbrella",
    # "beetle",
    # "airplane",
    # "bicycle",
    # "hotdog",
    # "snake",
    # "wine_bottle"
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig
        self.mode = mode

        if self.opts.pidinet:
            self.transform_sk = transforms.Compose([
                transforms.Resize((opts.max_size, opts.max_size)),
                transforms.functional.invert(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.all_sketches_path = []
        self.all_photos_path = {}
        self.all_pair = []
        self.all_pair_by_category = {}
        self.files = []
        self.n2c = {}

        if self.opts.fg:
            print("El archivo a leer es fine-grained")
            # The dataloader will be fine-grained
            if self.opts.txt_train != '' and mode == 'train':
                # get files ends with _train in the directory ./dir
                for i in os.listdir(self.opts.txt_train):
                    if i.endswith('_train.txt'):
                        self.files.append(os.path.join(self.opts.txt_train, i))
                
                with open(self.files[0], 'r') as f:
                    c = 0
                    for line in f.readlines():
                        c += 1
                        line = line.strip()
                        if line == '':
                            continue
                        sketch_path, image_path, category = line.split('\t')

                        name_of_category = sketch_path.split(os.path.sep)[-2]
                        self.n2c[name_of_category] = category

                        if category not in self.all_pair_by_category:
                            self.all_pair_by_category[category] = []
                        
                        self.all_pair_by_category[category].append((sketch_path, image_path))
                        self.all_pair.append((sketch_path, image_path, category))
                    self.all_categories = list(self.all_pair_by_category.keys())

            elif self.opts.txt_test != '' and mode == 'val':
                # get files ends with _test in the directory ./dir
                for i in os.listdir(self.opts.txt_test):
                    if i.endswith('_test.txt'):
                        self.files.append(os.path.join(self.opts.txt_test, i))
                
                with open(self.files[0], 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line == '':
                            continue
                        sketch_path, image_path, category = line.split('\t')

                        name_of_category = sketch_path.split(os.path.sep)[-2]
                        self.n2c[name_of_category] = category
                        
                        if category not in self.all_pair_by_category:
                            self.all_pair_by_category[category] = []
                        
                        self.all_pair_by_category[category].append((sketch_path, image_path))
                        self.all_pair.append((sketch_path, image_path, category))
                    self.all_categories = list(self.all_pair_by_category.keys())
        else:
            if self.opts.txt_train != '' and mode == 'train':
                # get files ends with _train in the directory ./dir
                for i in os.listdir(self.opts.txt_train):
                    if i.endswith('_train.txt'):
                        self.files.append(os.path.join(self.opts.txt_train, i))

                sketch, photo = self.get_files(self.files)
                
                with open(sketch, 'r') as f:
                    print(f"El archivo {sketch} se guardará en self.all_sketches_path")
                    for line in f.readlines():
                        line = line.strip()
                        if line == '':
                            continue
                        full_file_path, category = line.split('\t')
                        self.all_sketches_path.append(full_file_path)

                with open(photo, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line == '':
                            continue
                        full_file_path, category = line.split('\t')

                        name_of_category = full_file_path.split(os.path.sep)[-2]
                        self.n2c[name_of_category] = category

                        if category not in self.all_photos_path:
                            self.all_photos_path[category] = []
                        self.all_photos_path[category].append(full_file_path)
                    self.all_categories = list(self.all_photos_path.keys())
            
            elif self.opts.txt_test != '' and mode == 'val':
                # get files ends with _test in the directory ./dir
                for i in os.listdir(self.opts.txt_test):
                    if i.endswith('_test.txt'):
                        self.files.append(os.path.join(self.opts.txt_test, i))

                sketch, photo = self.get_files(self.files)
                
                with open(sketch, 'r') as f:
                    print(f"El archivo {sketch} se guardará en self.all_sketches_path")
                    for line in f.readlines():
                        line = line.strip()
                        if line == '':
                            continue
                        full_file_path, category = line.split('\t')
                        self.all_sketches_path.append((full_file_path, category))

                with open(photo, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line == '':
                            continue
                        full_file_path, category = line.split('\t')

                        name_of_category = full_file_path.split(os.path.sep)[-2]
                        self.n2c[name_of_category] = category

                        if category not in self.all_photos_path:
                            self.all_photos_path[category] = []
                        self.all_photos_path[category].append(full_file_path)
                    self.all_categories = list(self.all_photos_path.keys())
            else:
                self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
                if '.ipynb_checkpoints' in self.all_categories:
                    self.all_categories.remove('.ipynb_checkpoints')
                    
                if self.opts.data_split > 0:
                    np.random.shuffle(self.all_categories)
                    if used_cat is None:
                        self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
                    else:
                        self.all_categories = list(set(self.all_categories) - set(used_cat))
                else:
                    if mode == 'train':
                        self.all_categories = list(set(self.all_categories) - set(unseen_classes))
                    else:
                        self.all_categories = unseen_classes
                for category in self.all_categories:
                    self.all_sketches_path.extend(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png')))
                    self.all_photos_path[category] = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpg'))

    def get_files(self, files):
        for file in self.files:
            if 'sketch' in file:
                sketch = file
            else:
                photo = file
        return sketch, photo

    def __len__(self):
        if self.opts.fg and self.mode == 'train':
            return len(self.all_pair)
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        if self.opts.fg and self.mode == 'train':
            sk_path, img_path, category = self.all_pair[index]
            filename = os.path.basename(sk_path)

            neg_classes = self.all_categories.copy()
            neg_classes.remove(category)
            choise_neg = np.random.choice(neg_classes)
        else:
            if self.opts.txt_train != '':
                filepath, category = self.all_sketches_path[index]
                category = filepath.split(os.path.sep)[-2] if category == '' else category
                try:
                    category = self.n2c[category]
                except KeyError:
                    category = ''
                    category = filepath.split(os.path.sep)[-2] if category == '' else category
                    category = self.n2c[category]
                filename = os.path.basename(filepath)
            else:
                filepath = self.all_sketches_path[index]                
                category = filepath.split(os.path.sep)[-2]
                filename = os.path.basename(filepath)
        
            neg_classes = self.all_categories.copy()
            neg_classes.remove(category)

            sk_path  = filepath
            img_path = np.random.choice(self.all_photos_path[category])
            choise_neg = np.random.choice(neg_classes)

        if self.opts.fg and self.mode == 'train':
            index = np.random.choice(len(self.all_pair_by_category[choise_neg]))
            _, neg_path = self.all_pair_by_category[choise_neg][index]
        else:
            neg_path = np.random.choice(self.all_photos_path[choise_neg])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor  = self.transform(sk_data) # if not self.opts.pidinet else self.transform_sk(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms
    
    @staticmethod
    def data_transform_pidinet(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.functional.invert(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return dataset_transforms
    
class Ecommerce(Dataset):
    """
    Clase para cargar datasets con la estructura:
    
    path_imagen \t label
    """
    def __init__(self, file, max_size=None, transform=None, opts=None):
        self.data = []
        self.transform = transform
        self.opts = opts

        # Leer el archivo de imágenes y cargar los datos
        with open(file, 'r') as f_images:
            lines = f_images.readlines()
            for line in lines:
                image_path, label = line.strip().split('\t')
                self.data.append((image_path, int(label)))

        if max_size is not None:
            self.data = self.data[max_size:max_size + max_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path, data_label = self.data[idx]
        data = ImageOps.pad(Image.open(data_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))

        if self.transform:
            data = self.transform(data)

        return data, data_label, data_path
    
    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms
