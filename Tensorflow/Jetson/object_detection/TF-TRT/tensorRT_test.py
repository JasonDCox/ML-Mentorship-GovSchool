import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import time
import os
import cv2
import re
import pandas as pd

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

input_saved_model_dir = os.path.join("model", "saved_model")
output_saved_model_dir = os.path.join("converted_model") 


convert_choice = input("Would you like to optimize the model [y/n]")
if(convert_choice == "y" or convert_choice == "Y"):
    convert_choice = input("Are you sure? This will take a long time [y/n]")
    if(convert_choice == "y" or convert_choice == "Y"):
        device = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(device[0], True)
        tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

        print("Getting parameters")
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        print("setting parameters")
        conversion_params = conversion_params._replace(precision_mode="FP16")
        print("creating converter")
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input_saved_model_dir,
            conversion_params=conversion_params)
        print("converting")
        converter.convert()
        print("Making input function")
        def my_input_fn():
            for _ in range(100):
                ##change 640, 640 to desired image size
                inp1 = np.random.normal(size=(1, 640, 640, 3)).astype(np.uint8)
                yield inp1,
        print("building")
        converter.build(input_fn=my_input_fn)
        print("saving everything")
        converter.save(output_saved_model_dir)
        exit()

run_choice = input("Would you like to predict from a camera [c], image folder [i], video [v], or exit [e]")
while run_choice != 'c' and run_choice != 'i' and run_choice != 'v' and run_choice != 'e':
    print("Invalid choice, try again:\n")
    run_choice = input("Would you like to predict from a camera [c], image folder [i], video [v], or exit [e]")

if run_choice == "e":
    print("Exiting")
    exit()

def load_labels(path='labels.txt'):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

print("loading model")
saved_model_loaded = tf.saved_model.load(output_saved_model_dir, tags=[tag_constants.SERVING])
print("creating graph_func")
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]


if run_choice == "i":
    image_start = os.path.join("test_images")
    save_folder = os.path.join("result_images")
    images = os.listdir(image_start)

    print("Running images")

    longest = 0
    labels = load_labels("labels.txt")
    final_data = []

    total_FPS = 0
    images_skip = 5
    frames = 0

    for image_end in images:
        if image_end[-1] != "l":
            if images_skip <= 0:
                frames += 1
            curr_image_data = []
            curr_image_data.append(image_end)
            image_path = image_start + "/" + image_end
            ##print(image_path)
            image = cv2.imread(image_path)
            height, width, colors = image.shape
            print("url:", image_path)
            print("width: ", width, "height:", height)
            curr_image_data.append(width)
            curr_image_data.append(height)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640,640))
            start = time.time()
            input_tensor = tf.convert_to_tensor(img)
            input_tensor = tf.reshape(input_tensor, [1, 640, 640, 3])
            input_tensor = tf.cast(input_tensor, tf.uint8)
            #print(input_tensor)
            output = graph_func(input_tensor)
            
            boxes = np.squeeze(output["detection_boxes"])
            classes = np.squeeze(output["detection_classes"])
            scores = np.squeeze(output["detection_scores"])
            count = 100
            
            #print("boxes:", type(boxes), boxes)
            #print("classes:", type(classes), classes)
            #print("scores:", type(scores), scores)
            #print(scores[0], max(scores))

            res = []
            for i in range(count):
                if scores[i] >= 0.0:
                    print("Score: ", scores[i])
                    result = {
                        'bounding_box': boxes[i],
                        'class_id': classes[i],
                        'score': scores[i]
                    }
                    res.append(result)
            finish = time.time()

            if images_skip <= 0:
                total_FPS +=  (1 / (finish - start))
            else:
                print("skipped FPS add")
                images_skip -= 1
            
            for result in res:
                curr_image_data.append(result)

            for res_i in range(1):
                result = res[0]
                ymin, xmin, ymax, xmax  = result['bounding_box']
                xmin = int(max(1,xmin * width))
                xmax = int(min(width, xmax * width))
                ymin = int(max(1, ymin * height))
                ymax = int(min(height, ymax * height))
                
                print(labels[int(result['class_id'])-1])
                cv2.rectangle(image,(xmin, ymin),(xmax, ymax),(0,255,0),3)
                lab_text = format(result['score']*100, "0.0f") + "% " + labels[int(result['class_id'])-1]
                cv2.putText(image, lab_text,(xmin, min(ymax, height-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
            
            if len(curr_image_data) > longest:
                longest = len(curr_image_data)
            final_data.append(curr_image_data)
            cv2.imwrite(save_folder + "/" + image_end, image)

    possible_columns = ['id', 'width', 'height', '1box', '2box', '3box', '4box', '5box', '6box', '7box', '8box', '9box', '10box', '11box', '12box', '13box', '14box', '15box', '16box', '17box', '18box', '19box', '20box', '21box', '22box', '23box', '24box', '25box', '26box', '27box', '28box', '29box', '30box', '31box', '32box', '33box', '34box', '35box', '36box', '37box', '38box', '39box', '40box', '41box', '42box', '43box', '44box', '45box', '46box', '47box', '48box', '49box', '50box', '51box', '52box', '53box', '54box', '55box', '56box', '57box', '58box', '59box', '60box', '61box', '62box', '63box', '64box', '65box', '66box', '67box', '68box', '69box', '70box', '71box', '72box', '73box', '74box', '75box', '76box', '77box', '78box', '79box', '80box', '81box', '82box', '83box', '84box', '85box', '86box', '87box', '88box', '89box', '90box', '91box', '92box', '93box', '94box', '95box', '96box', '97box', '98box', '99box', '100box', '101box', '102box', '103box', '104box', '105box', '106box', '107box', '108box', '109box', '110box', '111box', '112box', '113box', '114box', '115box', '116box', '117box', '118box', '119box', '120box', '121box', '122box', '123box', '124box', '125box', '126box', '127box', '128box', '129box', '130box', '131box', '132box', '133box', '134box', '135box', '136box', '137box', '138box', '139box', '140box', '141box', '142box', '143box', '144box', '145box', '146box', '147box', '148box', '149box', '150box']
    df = pd.DataFrame(final_data, columns=possible_columns[0:longest])
    df.to_csv("TensorRT_results.csv")
    
    print("Average FPS: ", total_FPS / frames)
elif run_choice == "v":
    cap = cv2.VideoCapture("test.mp4")

    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    width = int(cap.get(3))
    height = int(cap.get(4))
    labels = load_labels("labels.txt")

    out = cv2.VideoWriter('tensorRT.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height))

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            #change to desired resolution
            img = cv2.resize(frame, (640,640))
            start = time.time()
            input_tensor = tf.convert_to_tensor(img)
            input_tensor = tf.reshape(input_tensor, [1, 640, 640, 3])
            input_tensor = tf.cast(input_tensor, tf.uint8)

            output = graph_func(input_tensor)
            
            boxes = np.squeeze(output["detection_boxes"])
            classes = np.squeeze(output["detection_classes"])
            scores = np.squeeze(output["detection_scores"])
            count = 100

            res = []
            for i in range(count):
                if scores[i] >= 0.6:
                    print("Score: ", scores[i])
                    result = {
                        'bounding_box': boxes[i],
                        'class_id': classes[i],
                        'score': scores[i]
                    }
                    res.append(result)

            finish = time.time()
            print("FPS:", 1 / (finish - start))
            
            res_counter = 0
            for result in res:
                if result["score"] >= 0.6: #change to desired confidence requirement
                    res_counter += 1
                    ymin, xmin, ymax, xmax  = result['bounding_box']
                    xmin = int(max(1,xmin * width))
                    xmax = int(min(width, xmax * width))
                    ymin = int(max(1, ymin * height))
                    ymax = int(min(height, ymax * height))
                    
                    print(labels[int(result['class_id']) - 1])
                    cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
                    lab_text = format(result['score']*100, "0.0f") + "% " + labels[int(result['class_id']) - 1]
                    cv2.putText(frame,lab_text,(xmin, min(ymax, height-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
            
            out.write(frame)
        else:
            break
    cap.release()
    out.release()

    print("Video saved and completed")
elif run_choice == "c":
    labels = load_labels("labels.txt")

    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640,640))
        start = time.time()
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = tf.reshape(input_tensor, [1, 640, 640, 3])
        input_tensor = tf.cast(input_tensor, tf.uint8)

        output = graph_func(input_tensor)
        
        boxes = np.squeeze(output["detection_boxes"])
        classes = np.squeeze(output["detection_classes"])
        scores = np.squeeze(output["detection_scores"])
        count = 100

        res = []
        for i in range(count):
            if scores[i] >= 0.6:
                print("Score: ", scores[i])
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                res.append(result)

        finish = time.time()
        print("FPS:", 1 / (finish - start))
        #print(res)

        for result in res:
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1,xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            
            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 

        cv2.imshow('Pi Feed', frame)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
else:
    print("Invalid choice, please restart")
