#imports
import os
import math
import shutil
import numpy as np
import random
import cv2
from utils import progress_bar

# Variables
diff_tolerance = 10 # %
one_line_label_files = {}
# Give path to dataset to balance

path = input("Give dataset folder path:")
path = path.replace("\\", "/")
classes_file = path + "/classes.txt"
labels_dir = path + "/labels"
images_dir = path + "/images"
datasetinfo_path = path + "/datasetInfo.txt"
datasetinfo_data = {}
classes = []
classes_reduction = []
max_factor = 230 # factor
target_balance_for_each_class = {} #class, target-annotations, factor
unbalance_warn = False
# Check if datasetInfo.txt file exists
is_file = os.path.isfile(datasetinfo_path)

def img_add_noise_blur(img):
    def sp_noise(image,prob):
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def blur(image, size):
        # Averaging
        # You can change the kernel size as you want
        avging = cv2.blur(image, (size, size))
        return avging

    #Recommended values between 0.01 and 1
    noise_val = random.uniform(0.01, 0.08)
    #Recommended values between 2 and 25
    blur_val = random.randint(2, 15)

    #Create new image with noise
    noise_img = sp_noise(img, noise_val)
    img = noise_img

    blur_img = blur(img,blur_val)
    img = blur_img
    return img

# If not, run utils/datasetAnalyser.py to create it
if not is_file:
    print(f'datasetInfo.txt not found -> running datasetAnalyzer')
    path = path
    classes_file = path + "/classes.txt"
    labels_dir = path + "/labels"
    classes = []
    annotations = []
    #read classes file and store it in a list
    f = open(classes_file, "r")
    #print(f.read())
    for line in f:
        line_to_list = line.replace("\n", "")
        classes.append(line_to_list)
        annotations.append(0)
    f.close()
    #print(classes)

    #read label files and store number of labels it in a list
    for filename in os.listdir(labels_dir):
        f = os.path.join(labels_dir, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.endswith(".txt"):
            #print(f)
            file = open(f, "r")
            #print(file.read())
            for line in file:
                annotation_number = line.split(" ")[0]
                annotations[int(annotation_number)] += 1
            file.close()

    #print(classes)
    #print(annotations)

    f = open(path+"/datasetInfo" + ".txt", "w")

    print("Number of annotations in dataset by class:\n")

    for i in range(len(classes)):
        print(classes[i] + "-"+str(annotations[i]))
        f.write(classes[i] + "-" + str(annotations[i]) + "\n")
    f.close()
    print(f"\nTotal number of annotations: {sum(annotations)}")
else:
    # Read data from datasetInfo
    f = open(path+"/datasetInfo" + ".txt", "r")
    for line in f:
        annotation, number = line.split("-")
        datasetinfo_data[annotation] = int(number)
        classes.append(annotation)
    f.close()
    # calculate number of files
    files_to_process = 0
    for filename in os.listdir(labels_dir):
        f = os.path.join(labels_dir, filename)
        if os.path.isfile(f) and filename.endswith(".txt"):
            files_to_process += 1
    # Analyze how to achieve optimal balance
    # Calculate how many labels have only one line
#read label files and store number of labels it in a list
    files_processed = 0
    print(f'Analyzing label files')
    for filename in os.listdir(labels_dir):
        percents = round(files_processed / files_to_process * 100, 2)
        percents = str(percents)
        prog_bar = progress_bar.print_progress_bar(files_to_process, files_processed) #total lines, current line
        print(f'[-] Progress:|{prog_bar}| {percents}% done' + '\r', end='')
        labels_in_file = 0
        class_in_file = ""
        f = os.path.join(labels_dir, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.endswith(".txt"):
            #print(f)
            file = open(f, "r")
            #print(file.read())
            for line in file:
                if line != "":
                    labels_in_file += 1
                    class_in_file = line.split(" ")[0]
            file.close()
        if labels_in_file == 1:
            one_line_label_files[filename] = int(class_in_file)
        files_processed += 1
    #print(one_line_label_files)
    # Possible amount of reduction on each class
    #print('Possible maximum reduction on each class:')
    classes_count = 0
    for i in classes:
        count = list(one_line_label_files.values()).count(classes_count)
        #print(f'{i} - {count}')
        classes_reduction.append(count)
        classes_count += 1


    # Determine two biggest classes in datasetInfo
    print(f'Determining two largest classes in dataset and target balance value')
    first_class = ["",0,0]
    sec_class =["",0,0]
    indx = 0
    for x in datasetinfo_data:
        if datasetinfo_data[x] > first_class[1]:
            sec_class[0] = first_class[0]
            sec_class[1] = first_class[1]
            sec_class[2] = first_class[2]
            first_class[0] = x
            first_class[1] = datasetinfo_data[x]
            first_class[2] = indx
        elif datasetinfo_data[x] > sec_class[1]:
            sec_class[0] = x
            sec_class[1] = datasetinfo_data[x]
            sec_class[2] = indx
        indx += 1
    first_class_max_reduction = datasetinfo_data[first_class[0]]-classes_reduction[first_class[2]]
    second_class_max_reduction = datasetinfo_data[sec_class[0]]-classes_reduction[sec_class[2]]
    if first_class_max_reduction > second_class_max_reduction:
        target_balance = first_class_max_reduction, first_class[0]
    else:
        target_balance = second_class_max_reduction, sec_class[0]

    print(f'Target balance for all classes is {target_balance}')
    # Calculate how close could get with current dataset
    indx = 0

    for x in classes:
        class_factor = 1
        target_annotation = 0
        if datasetinfo_data[x] > target_balance[0]:

            target_annotation = datasetinfo_data[x]-target_balance[0]
            target_annotation = datasetinfo_data[x] - target_annotation
            target_balance_for_each_class[x] = int(target_annotation), -1
            diff_target = round(((target_annotation/target_balance[0])*100)-100,2)
        else:
            #class_factor = round(target_balance[0]/datasetinfo_data[x],0)
            target_class_annotation = target_balance[0]-datasetinfo_data[x]
            target_annotation = datasetinfo_data[x]+target_class_annotation
            diff_target = round(((target_annotation/target_balance[0])*100)-100,2)
        target_balance_for_each_class[x] = int(target_annotation), class_factor, diff_target
        if diff_target > diff_tolerance or diff_target > diff_tolerance*(-1):
            unbalance_warn = True
        indx += 1

    print(f'Target balances for each class:')
    for x in target_balance_for_each_class:
        print(f'{x} - target value: {target_balance_for_each_class[x][0]} -> difference from target balance: '
              f'{target_balance_for_each_class[x][2]} %')
    if unbalance_warn:
        print('WARNING -> Some of the classes are still unbalanced!')
        accept_input = input("Do you want to continue with balancing with current settings? [yes/no] ")
    else:
        accept_input = input("Do you want to continue with balancing with current settings? [yes/no] ")

    if accept_input == "no":
        exit()

    elif accept_input != "yes":
        print("Unknown command, exiting...")
        exit()
    else:
        # Create new dataset folder
        local_path = os.getcwd()
        source_folder_name = path.split('/')[-1]
        new_folder_name = path.replace(source_folder_name,source_folder_name+'_balanced')
        # Check if folder exists
        #logs_path_there = os.path.exists(new_folder_name)
        #if not logs_path_there:
            #print(f'Creating new folder {new_folder_name}')
            #os.mkdir(new_folder_name)

        #copy images and labels from original folder
        # Source and destination paths
        print(f'Copying files to new folder {new_folder_name}')
        source_folder = path
        destination_folder = new_folder_name
        destination_folder_exists = os.path.exists(destination_folder)
        if destination_folder_exists:
            print(f'WARNING! Destination folder already exists!')
            overwrite = input('Do you want to overwrite existing files? [yes/no] ')
            if overwrite == 'no':
                exit()
            elif overwrite != 'yes':
                print('Unknown command, exiting...')
                exit()
            else:
                print('Removing old files')
                shutil.rmtree(destination_folder)
        # Copy the entire folder and its contents
        shutil.copytree(source_folder, destination_folder)
        # Go through classes and determine if files should be added or deleted
        print(f'Starting class balancing')
        index = 0
        for x in datasetinfo_data:
            print(f'Balancing class - {x}')
            if datasetinfo_data[x] > target_balance[0]:
                print(f'Removing excess files from class')
                files_to_remove = datasetinfo_data[x]-target_balance[0]
                files_removed = 0
                for y in one_line_label_files:
                    if one_line_label_files[y] == index and files_removed < files_to_remove:
                        # File path
                        label_file_path = new_folder_name+'/labels/'+y
                        img_file_path = new_folder_name+'/images/'+y.replace('txt','jpg')
                        # Delete the file
                        os.remove(label_file_path)
                        os.remove(img_file_path)
                        files_removed += 1
                print(f'{files_removed} - files removed')
            else:
                files_to_add = target_balance_for_each_class[x][0]-datasetinfo_data[x]
                files_added = 0
                print(f'Adding {files_to_add} files and labels to class')
                while files_added < files_to_add:
                    percents = round(files_added / files_to_add * 100, 2)
                    percents = str(percents)
                    prog_bar = progress_bar.print_progress_bar(files_to_add, files_added) #total lines, current line
                    print(f'[{files_added}/{files_to_add}] Progress:|{prog_bar}| {percents}% done '+'\r', end='')
                    for y in one_line_label_files:
                        use_image = bool(random.getrandbits(1))
                        if use_image and one_line_label_files[y] == index:
                            new_file_name = y.split(".txt")
                            new_file_name = str(new_file_name[0]) + str(files_added) + "." + str(new_file_name[1])
                            # File path
                            orig_image_path = str(images_dir+'/'+y[:-3])
                            orig_image_path = orig_image_path+'jpg'
                            img = cv2.imread(orig_image_path)
                            modded_img = img_add_noise_blur(img)
                            img_file_path = new_folder_name+'/images/'+new_file_name+'jpg'
                            cv2.imwrite(img_file_path, modded_img)

                            label_file_path = img_file_path.replace('images','labels/')
                            label_file_path = label_file_path[:-3]
                            label_file_path = label_file_path+'txt'
                            #label_file_path = label_file_path.replace("jpg", "txt")
                            # copy label file
                            shutil.copyfile(labels_dir+'/'+y, label_file_path)
                            files_added += 1
                        else:
                            pass
                print(f'\n{files_added} - files added')
            index += 1

        print(f'Running datasetAnalyzer on balanced dataset')
        path = new_folder_name
        classes_file = path + "/classes.txt"
        labels_dir = path + "/labels"
        classes = []
        annotations = []
        #read classes file and store it in a list
        f = open(classes_file, "r")
        #print(f.read())
        for line in f:
            line_to_list = line.replace("\n", "")
            classes.append(line_to_list)
            annotations.append(0)
        f.close()

