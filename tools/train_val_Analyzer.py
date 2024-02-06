import os

path = input("Give dataset folder path:")
classes_file = path + "/classes.names"
labels_dir = path + "/labels"
train_path = path + "/train.txt"
val_path = path + "/validate.txt"


def read_file(path):
    #read classes file and store it in a list
    f = open(path, "r")
    file_data = []
    #print(f.read())
    for line in f:
        line_to_list = line.replace("\n", "")
        file_data.append(line_to_list)
    f.close()
    return file_data

def read_annotation(files, classes):
    annotations = []
    for i in classes:
        annotations.append(0)
    for file in files:
        file = file.replace('images','labels')
        file = file.replace('jpg', 'txt')
        # checking if it is a file
        if os.path.isfile(file) and file.endswith(".txt"):
            # print(f)
            r_file = open(file, "r")
            # print(file.read())
            for line in r_file:
                annotation_number = line.split(" ")[0]
                annotations[int(annotation_number)] += 1
            r_file.close()

    return annotations

classes = read_file(classes_file)
train_files = read_file(train_path)
val_files = read_file(val_path)
train_annotations = read_annotation(train_files,classes)
val_annotations = read_annotation(val_files,classes)
totals = (sum(train_annotations),sum(val_annotations))
print(f'Train vs val annotations total count: {totals[0]} / {totals[1]}')

for i in range(len(classes)):
    train_percents = round(train_annotations[i]/totals[0]*100,2)
    val_percents = round(val_annotations[i]/totals[1]*100,2)

    print(f'{classes[i]} - {str(train_annotations[i])}-{train_percents}% <--> {str(val_annotations[i])}-{val_percents}%')
