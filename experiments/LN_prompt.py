import os
import glob
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval_fg import Sketchy
from experiments.options import opts

import copy

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)
    
    train_opts = copy.deepcopy(opts)
    val_opts = copy.deepcopy(opts)

    train_dataset = Sketchy(train_opts, dataset_transforms, mode='train', return_orig=False)
    print ('train_dataset:', len(train_dataset))
    # En el caso de no contar con un archivo de validacion con pares (fg)
    val_opts.fg = False
    val_dataset = Sketchy(val_opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)
    print ('val_dataset:', len(val_dataset))

    print(train_dataset.opts)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='mAP', # 
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{mAP:.3f}-{top10:.2f}",
        mode='max', # min
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'none_.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    trainer = Trainer(
        accelerator='gpu',
        devices=[opts.gpu_id], # Se puede seleccionar en que GPU entrenar (depende de la version de pytorch-lightning)
        min_epochs=1, max_epochs=opts.epochs,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
    )

    if ckpt_path is None:
        model = Model()
    else:
        print ('resuming training from %s'%ckpt_path)
        model = Model().load_from_checkpoint(ckpt_path)

    print ('beginning training...good luck...')
    trainer.fit(model, train_loader, val_loader)