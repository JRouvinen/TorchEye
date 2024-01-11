import json
#import paramiko
#import time
#import os
import time
import glob
import os
import datetime

import cv2
import numpy
import numpy as np
import torch
from PIL import Image

from detect import detect_image, detect_images
from models import load_model
from utils.parse_config import parse_autodetect_config, parse_hyp_config
from utils.utils import load_classes, rescale_boxes
from profilehooks import profile


#from utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info

def get_files_in_dir(directory):
    # Get a list of all daily sub-directories
    daily_dirs = glob.glob(directory + "/*/")
    if len(daily_dirs) > 0:
        # Sort the directories by name in decreasing order and pick the first one
        latest_daily_dir = sorted(daily_dirs, reverse=True)[0]

        files = set()
        for dir_path, dir_names, file_names in os.walk(latest_daily_dir):
            for file_name in file_names:
                file_path = os.path.join(dir_path, file_name)
                files.add(file_path)
        return files
    else:
        files = set()
        return files

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def _write_json(image_path, detections, img_size, output_path, classes):

    folder_path = output_path  # folder path
    data = []
    img = np.array(Image.open(image_path))
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)

    for x1, y1, x2, y2, conf, cls_pred in detections:
        #print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")
        data.append({
            'image_path': image_path,
            'detections': classes[int(cls_pred)],
            'confidence': round(conf.item(),4),

        })
    return data


def monitor_local_folder(directory, interval,classes, model_path,gpu, weights_path,img_size,conf_thres,nms_thres,output, hyp):
    #Load model and needed config files
    print('Loading model...')
    model = load_model(model_path,hyp, gpu, weights_path)
    model.eval()  # Set model to evaluation mode
    print('Model loaded...')

    files_before = get_files_in_dir(directory)
    for file in files_before:
        print(f"File detected: {file}")
    print(f"Start monitoring folder: {directory}")
    while True:
        time.sleep(interval)

        files_after = get_files_in_dir(directory)

        added_files = files_after - files_before
        if added_files:

            #For performance testing
            # Start timer
            start_time = time.time()

            data = []
            img_paths = []

            for file in added_files:
                file = file.replace('\\', '/')
                print(f"New file detected: {file}")
                img_paths.append(file)
                #img = cv2.imread(file)
                # -Fix for [ WARN:0@25.254] global loadsave.cpp:248 cv::findDecoder imread_( - ver 0.3.0
                stream = open(file, "rb")
                bytes = bytearray(stream.read())
                numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
                bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                img, orig_img, dim = prep_image(bgrImage,img_size)
                # ---------------------
                #img, orig_img, dim = prep_image(img,img_size)
                print(f"Detecting objects in new images...")
                detections, imgs = detect_image(model, img, img_size, conf_thres, nms_thres)
                det_data = _write_json(file,detections[0],img_size,output,classes)
                data.append(det_data)
            files_before = files_after
            print('Detections for new files done...')
            print(f'Writing JSON to {output + "/" + "detections" + ".json"}')
            # Create data for JSON
            data_for_json = {
                "timestamp": time.time(),
                "folder_path": directory,
                "images": data
            }
            # Convert the data to JSON
            json_data = json.dumps(data_for_json, indent=4)

            # Print the JSON data
            # print(json_data)

            # Save JSON  data
            f = open(output + "/" + "detections" + ".json", "w")
            f.write(json_data)
            f.close()
            #Performance testing
            # End timer
            end_time = time.time()

            # Calculate elapsed time
            elapsed_time = end_time - start_time
            print("Elapsed time: ", elapsed_time)

            print('Continue monitoring...')

def monitor_folder_ssh(host, port, username, password, directory, interval):
    pass
    '''
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port, username, password)

    try:
        sftp = client.open_sftp()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        client.close()
        return

    print(f"Start monitoring folder: {directory}")

    files_before = get_files_in_dir(sftp, directory)

    while True:
        time.sleep(interval)

        try:
            files_after = get_files_in_dir(sftp, directory)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            break

        added_files = files_after - files_before
        if added_files:
            for file in added_files:
                print(f"New file detected: {file}")
            files_before = files_after
    sftp.close()
    client.close()
    '''
@profile(filename='./logs/profiles/autoDetect.prof', stdout=False)

def run():
    date = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    ver = "0.3.0"
    '''
    #print_environment_info(ver, "output/" + date + "_detect" + ".txt")
    # Parse config file
    directory = "/data_fetching/road_camera_data/data/datalake/digitraffic/images/"
    json_path = ""
    # Connection parameters
    host = "hostname"
    port = 22  # default SSH port
    username = "username"
    password = "password"
    classes_path = "config/Lyra.names"
    #directory = "/path/to/your/folder"
    #print(f"Command line arguments: {args}")
    #Create_new detect_file
    conf_thres = 0.35
    nms_thres = 0.5
    img_size = 640
    model = "config/Lyra-tiny.cfg"
    weights = "weights/Lyra-tiny_640.weights"
    gpu = -1
    '''
    params = parse_autodetect_config("config/autodetect.cfg")
    classes = load_classes(params['classes'])
    hyp_config = parse_hyp_config(params['hyperparams'])
    if params['json_path'] == "":
        params['json_path'] = params['directory']
    # List of class names
    monitor_local_folder(params['directory'], int(params['interval']),classes,params['model'],int(params['gpu']),
                         params['weights'],int(params['img_size']),float(params['conf_thres']),float(params['nms_thres']),params['json_path'], hyp_config)


if __name__ == '__main__':
    run()
