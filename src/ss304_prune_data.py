from ss304_dataset import ss304Dataset
import os, shutil, random, json
import ss304_globals

DATA_PATH = os.path.join(ss304_globals.DATA_DIR, 'ss304')
DATA_PATH_REDUCED = os.path.join(ss304_globals.DATA_DIR, 'ss304_reduced')

def make_reduced_set(data_type, num_per_class=5):

    dataset = ss304Dataset(data_type=data_type)
    img_dir = os.path.join(DATA_PATH_REDUCED, data_type)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    data_dict = {}
    classes = []
    for i in range(dataset.num_classes):
        classes.append([d for d in dataset.data.values() if d['label'] == i])

    for i in range(dataset.num_classes):
        random.shuffle(classes[i])
        classes[i] = classes[i][:num_per_class]
    
    for img_class in classes:
        for img in img_class:
            src_path = img['image'].replace('/', '\\')
            path_tok = img['image'].split('\\')
            img_path = path_tok[-1]

            dest_path = os.path.join(DATA_PATH_REDUCED, data_type, img_path.replace('/', '\\'))
            dest_dir = dest_path.split('\\')[:-1]
            dest_dir = '\\'.join(dest_dir)
            print(dest_dir)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            print('Copying {} to {}'.format(src_path, dest_path))
            shutil.copy(src_path, dest_path)
            
            img_label = img['label']
            data_dict[img_path] = img_label
    
    json_path = os.path.join(img_dir, data_type + '.json')
    # print(data_dict)
    with open(json_path, 'w') as f:
        json.dump(data_dict, f)

    

if __name__=='__main__':
    #test_set = ss304Dataset(data_type='test')
    #train_set = ss304Dataset(data_type='train')
    #valid_set = ss304Dataset(data_type='valid')
    make_reduced_set('test', num_per_class=10)
    make_reduced_set('valid', num_per_class=10)
    make_reduced_set('train', num_per_class=10)