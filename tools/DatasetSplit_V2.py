from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
import os
import shutil

# Assuming 'annotations' is a list of dictionaries with 'image_id', 'class_id', and 'bbox'
# You will need to replace this with your actual method to extract annotations
#annotations = [
#    {'image_id': 'image_1.jpg', 'class_id': 0, 'bbox': [...]},
    # ... other annotations ...
#]
# Get images and labels
data_folder = input('Give path to folder where images and labels are located: ')
# Count files in the defined folder
print('Counting number of label files in folder...')
file_id = 0
annotations = []
for filename in os.listdir(f'{data_folder}/labels'):
    f = os.path.join(f'{data_folder}/labels', filename)
    f.replace('\\', '/')
    # checking if it is a file
    file_added = False
    if os.path.isfile(f):
        if filename.endswith('.txt'):
            file_id += 1
            annotation_dic = {}
            file = open(f, "r")
            for x in file:
                #print(x)
                annotation_dic = {}
                if x != '':
                    line_data = x.split(' ')
                    annotation_dic['image_id'] = filename.replace('.txt','.jpg')
                    annotation_dic['class_id'] = line_data[0]
                    annotation_dic['bbox'] = [line_data[1],line_data[2],line_data[3],line_data[4].replace('\n','')]
                    annotations.append(annotation_dic)
            file.close()

print(f'{file_id} label files found and processed.')

# Extract unique classes
unique_classes = set([ann['class_id'] for ann in annotations])

# Create a bounding box ID to image dict and an array for multilabel stratification
bbox_to_image = {}
multilabels = []
print(f'Mapping out unique and multilabel annotations')
for unique_class in unique_classes:
    class_bboxes = [ann for ann in annotations if ann['class_id'] == unique_class]
    for bbox in class_bboxes:
        bbox_id = f"{bbox['image_id']}_{bbox['bbox']}"
        bbox_to_image[bbox_id] = bbox['image_id']
        multilabels.append([int(unique_class == bbox['class_id']) for unique_class in unique_classes])

multilabels_copy = multilabels
multilabels = np.array(multilabels)

# Assuming bbox_to_image and multilabels have been appropriately defined as per the earlier script
bbox_ids = list(bbox_to_image.keys())

# Perform multilabel stratified shuffle split
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
'''
for train_index, test_index in msss.split(multilabels, multilabels):
    train_validate_bboxes = [list(bbox_to_image.keys())[index] for index in train_index]
    test_bboxes = [list(bbox_to_image.keys())[index] for index in test_index]
'''
print(f'Creating split for training and test sets')
# Get the indices for train and test splits, making sure they associate correctly with bbox ids
for train_index, test_index in msss.split(multilabels, multilabels):
    train_validate_bbox_ids = []
    train_validate_bbox_ids_indx = []
    test_bbox_ids = []
    test_bbox_ids_indx = []
    for i in train_index:
        if i-1 < len(bbox_ids):
            train_validate_bbox_ids.append(bbox_ids[i-1])
            train_validate_bbox_ids_indx.append(i)
    #train_validate_bbox_ids = [bbox_ids[index-1] for index in train_index]
    for i in test_index:
        if i-1 < len(bbox_ids):
            test_bbox_ids.append(bbox_ids[i-1])
            test_bbox_ids_indx.append(i)
    #test_bbox_ids = [bbox_ids[index-1] for index in test_index]

# Further split the train_validate set into train and validate
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)  # 0.2 * 0.9 = 0.18 ~ 20% of total
train_validate_multilabels = []
for i in train_validate_bbox_ids_indx:
    train_validate_multilabels.append(multilabels_copy[i])
train_validate_multilabels = np.array(train_validate_multilabels)
print(f'Creating split for evaluation')
#for train_index, validate_index in msss.split(np.array(train_validate_bbox_ids), np.array(train_validate_bbox_ids)):
for train_index, validate_index in msss.split(train_validate_multilabels, train_validate_multilabels):
    train_bboxes = []
    validate_bboxes = []
    for i in train_index:
        if i - 1 < len(train_validate_bbox_ids_indx):
            train_bboxes.append(train_validate_bbox_ids[i-1])
    #train_bboxes = [train_validate_bbox_ids[index] for index in train_index]
    for i in validate_index:
        if i - 1 < len(train_validate_bbox_ids_indx):
            validate_bboxes.append(train_validate_bbox_ids[i - 1])
    #validate_bboxes = [train_validate_bbox_ids[index] for index in validate_index]



# Convert bbox IDs back to image lists without duplicates
train_images = list({bbox_to_image[bbox_id] for bbox_id in train_bboxes})
validate_images = list({bbox_to_image[bbox_id] for bbox_id in validate_bboxes})
test_images = list({bbox_to_image[bbox_id] for bbox_id in test_bbox_ids})

print(f'Moving files into correct folders')
# Now, transfer the image files and annotations into the appropriate dataset directories
# You will need to modify the paths and methods according to your file structure
copied_files_list = []
for dataset, images in [('train', train_images), ('validate', validate_images), ('eval', test_images)]:
    os.makedirs(f"{data_folder}/{dataset}", exist_ok=True)
    os.makedirs(f"{data_folder}/{dataset}/labels", exist_ok=True)
    os.makedirs(f"{data_folder}/{dataset}/images", exist_ok=True)
    for image in images:
        if copied_files_list.count(image) == 0:
            print(f'Moving {image} to folder {data_folder}/{dataset}/images/')
            shutil.move(f"{data_folder}/images/{image}", f"{data_folder}/{dataset}/images/{image}")
            copied_files_list.append(image)
            print(f'Moving annotation for image {image} to folder {data_folder}/{dataset}/annotations/')
            shutil.move(f"{data_folder}/labels/{image.replace('.jpg','.txt')}", f"{data_folder}/{dataset}/labels/{image.replace('.jpg','.txt')}")

        # Corresponding annotations need to be moved as well
        # Implement the logic to move annotations as per your annotation file structure
print(f'All done!')
# Don't forget to install 'iterative-stratification' package to use MultilabelStratifiedShuffleSplit
# pip install iterative-stratification