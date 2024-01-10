import os

'''
This code analyzes a dataset by counting the number of annotations for each class. 
It prompts the user to provide the path to the dataset folder, reads the classes from a "classes.txt" file, 
and initializes a list to store the number of annotations for each class. 
It then reads the label files from a "labels" directory and updates the annotation count for each class. 
Finally, it writes the results to a "datasetInfo.txt" file, displaying the number of annotations for each 
class and the total number of annotations in the dataset.
'''

# Prompt user for dataset folder path
path = input("Give dataset folder path:")

# Define paths for classes file and labels directory
classes_file = path + "/classes.names"
labels_dir = path + "/labels"

# Initialize lists for classes and annotations
classes = []
annotations = []

# Read classes file and store classes in a list
f = open(classes_file, "r")
for line in f:
    line_to_list = line.replace("\n", "")
    classes.append(line_to_list)
    annotations.append(0)
f.close()

# Read label files and update annotation count for each class
for filename in os.listdir(labels_dir):
    f = os.path.join(labels_dir, filename)
    if os.path.isfile(f) and filename.endswith(".txt"):
        file = open(f, "r")
        for line in file:
            annotation_number = line.split(" ")[0]
            annotations[int(annotation_number)] += 1
        file.close()

# Write results to datasetInfo.txt file
f = open(path + "/datasetInfo.txt", "w")
print("Number of annotations in dataset by class:\n")

for i in range(len(classes)):
    print(classes[i] + "-" + str(annotations[i]))
    f.write(classes[i] + "-" + str(annotations[i]) + "\n")
f.close()

# Print total number of annotations and file path of datasetInfo.txt
print(f"\nTotal number of annotations: {sum(annotations)}")
print(f'Results written into {path + "/datasetInfo.txt"}\n')
