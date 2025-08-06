import os

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.model_LN_prompt import Model
from src.dataset_retrieval_fg import Ecommerce
from experiments.options import opts

from tqdm import tqdm


if __name__ == '__main__':
    dataset_transforms = Ecommerce.data_transform(opts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_file = opts.image_file 
    sketch_file = opts.sketch_file 

    image_dataset = Ecommerce(image_file, opts=opts, transform=dataset_transforms)
    sketches_dataset = Ecommerce(sketch_file, opts=opts, transform=dataset_transforms)

    image_loader = DataLoader(dataset=image_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=False)
    sketches_loader = DataLoader(dataset=sketches_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=False)

    path_models = 'saved_models/%s'%opts.exp_name
    if not os.path.exists(path_models):
        os.makedirs(path_models)

    # Creamos la carpeta para guardar los resultados
    path_results = 'results/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Cargar el modelo
    try:
        model = Model().to(device)
        model_checkpoint = torch.load(opts.model) 
        model.load_state_dict(model_checkpoint['state_dict'])
        model.eval()                                                
    except:
        print("No se pudo cargar el modelo. Intenta nuevamente cambiando el argumento --model_type")
        exit()
    
    # Función de distancia
    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1) 

    # Listas para almacenar los embeddings y las etiquetas correspondientes
    sketch_embeddings = []
    sketch_labels = []
    sketch_paths = []
    top_k = len(image_dataset)
    archivo_salida = path_results + opts.output_file #"original_model_flickr15"

    with torch.no_grad():
        # Genera los embeddings para los sketches
        print("Generando embeddings para los sketches")
        for (sketch, s_label, path) in tqdm(sketches_loader):
            sketch_feat = model(sketch.to(device), dtype='sketch')
            sketch_embeddings.append(sketch_feat) # [batch_size, 512]
            sketch_labels.append(s_label) # [batch_size]
            sketch_paths.append(path)

        sketch_embeddings = torch.cat(sketch_embeddings, dim=0) # [batch_size, 512] -> [n_sketches, 512]
        sketch_labels = torch.cat(sketch_labels, dim=0) # [batch_size] -> [n_sketches]
        sketch_paths = [path for paths in sketch_paths for path in paths]

    distance_values = []
    labels_images = []
    images_path = []

    with torch.no_grad():
        print("Generando rankings")
        for (image, label, path) in tqdm(image_loader):
            image_feat = model(image.to(device), dtype='image') # [batch_size, 512]
            
            # [n_sketches, 1, 512] - [1, batch_size, 512] -> [n_sketches, batch_size]
            distance = -1 * distance_fn(sketch_embeddings.unsqueeze(1), image_feat.unsqueeze(0))
            # Es decir, para cada sketch se obtiene la distancia con cada imagen del batch

            distance_values.append(distance)
            labels_images.append(label.unsqueeze(0))
            images_path.append(path)

        # Convierte la lista de tuplas en una lista de cadenas
        images_path = [path for paths in images_path for path in paths]
        all_images_path = images_path

        # Convierte las listas en tensores
        all_query_distance = torch.cat(distance_values, dim=1).to(device) # [n_sketches, total_images]
        all_labels_images = torch.cat(labels_images, dim=1).to(device) # [1, total_images]

        # Obtiene los valores máximos (mas cercanos a 0) y sus índices
        max_values, max_indices = torch.topk(all_query_distance, top_k, dim=1, largest=True, sorted=True) # [n_sketches, top_k]

        # Obtiene los labels de las imágenes con los valores más altos
        max_labels = all_labels_images[0][max_indices]

        top_images_path = []
        for i in range(len(max_indices)):
            tmp = []
            for j in range(10):
                tmp.append(all_images_path[max_indices[i][j]])
            top_images_path.append(tmp)

        # Abrir el archivo en modo escritura
        with open(archivo_salida + ".csv", "w") as f:
            for i in range(len(sketch_labels)):
                sketch_label = sketch_labels[i]
                max_label_row = max_labels[i]
                sketch_label_str = str(sketch_label.item())
                max_label_row_str = ", ".join(map(str, max_label_row.tolist()))
                f.write(f"{sketch_label_str}, {max_label_row_str}\n")

        # Guardar los paths de las imágenes
        with open(archivo_salida + "_images.txt", "w") as f:
            for i in range(len(sketch_paths)):
                sketch_label = sketch_labels[i]
                sketch_path = sketch_paths[i]
                top_images_path_row = top_images_path[i]
                sketch_label_str = str(sketch_path)
                top_images_path_row_str = ", ".join(map(str, top_images_path_row))
                f.write(f"{sketch_label}, {sketch_label_str}, {top_images_path_row_str}\n")

        print("Valores guardados en", archivo_salida)
    print(f"Se utilizó el modelo {opts.model} para generar los rankings de las imágenes de {opts.image_file} con los sketches de {opts.sketch_file}")