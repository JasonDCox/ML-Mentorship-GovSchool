# Based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/README.md
import os
import re
import cv2
from tflite_runtime.interpreter import Interpreter
import pandas as pd
#from tflite_model_maker import model_spec
import numpy as np
import time

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

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

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)
  count = int(get_output_tensor(interpreter, 2))

  #print("boxes length:", boxes)
  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def main():
    total_FPS = 0
    frames = 0
    choice = input("Would you like to test images [i], make a video with detections [v], or run a camera feed [c]? ")
    if choice == "i":
        print("Testing images")
        
        final_data = []
        longest = 0
        labels = load_labels()
        ##print("labels:", labels)
        
        interpreter = Interpreter('model.tflite')
        interpreter.allocate_tensors()
        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
        

        image_start = "test_images"
        save_folder = "tfLite_results/"
        images = os.listdir(image_start)

        for image_end in images:
            if image_end[-1] != "l":
                frames += 1
                curr_image_data = []
                curr_image_data.append(image_end)
                image_path = image_start + "/" + image_end
                ##print(image_path)
                image = cv2.imread(image_path)
                height, width, colors = image.shape
                #print("url:", image_end)
                #print("width: ", width, "height:", height)
                ##print("url: ", image_end)
                curr_image_data.append(width)
                curr_image_data.append(height)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640,640))

                start = time.time()
                #get all possible detections for evaluation on desktop software
                res = detect_objects(interpreter, img, 0)
                finish = time.time()

                print("fps: ", 1 / (finish - start))
                total_FPS += (1 / (finish - start))

                #place most confident box
                res_counter = 0
                for res_i in range(1):
                    result = res[0]
                    ##curr_image_data.append(result)
                    res_counter += 1
                    ymin, xmin, ymax, xmax  = result['bounding_box']
                    xmin = int(max(1,xmin * width))
                    xmax = int(min(width, xmax * width))
                    ymin = int(max(1, ymin * height))
                    ymax = int(min(height, ymax * height))
                    
                    print(labels[int(result['class_id'])])
                    cv2.rectangle(image,(xmin, ymin),(xmax, ymax),(0,255,0),3)
                    lab_text = format(result['score']*100, "0.0f") + "% " + labels[int(result['class_id'])]
                    cv2.putText(image,lab_text,(xmin, min(ymax, height-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
                    
                for result in res:
                    curr_image_data.append(result)

                if len(curr_image_data) > longest:
                    longest = len(curr_image_data)
                final_data.append(curr_image_data)
                cv2.imwrite(save_folder + image_end, image)
        possible_columns = ['id', 'width', 'height', '1box', '2box', '3box', '4box', '5box', '6box', '7box', '8box', '9box', '10box', '11box', '12box', '13box', '14box', '15box', '16box', '17box', '18box', '19box', '20box', '21box', '22box', '23box', '24box', '25box', '26box', '27box', '28box', '29box', '30box', '31box', '32box', '33box', '34box', '35box', '36box', '37box', '38box', '39box', '40box', '41box', '42box', '43box', '44box', '45box', '46box', '47box', '48box', '49box', '50box', '51box', '52box', '53box', '54box', '55box', '56box', '57box', '58box', '59box', '60box', '61box', '62box', '63box', '64box', '65box', '66box', '67box', '68box', '69box', '70box', '71box', '72box', '73box', '74box', '75box', '76box', '77box', '78box', '79box', '80box', '81box', '82box', '83box', '84box', '85box', '86box', '87box', '88box', '89box', '90box', '91box', '92box', '93box', '94box', '95box', '96box', '97box', '98box', '99box', '100box', '101box', '102box', '103box', '104box', '105box', '106box', '107box', '108box', '109box', '110box', '111box', '112box', '113box', '114box', '115box', '116box', '117box', '118box', '119box', '120box', '121box', '122box', '123box', '124box', '125box', '126box', '127box', '128box', '129box', '130box', '131box', '132box', '133box', '134box', '135box', '136box', '137box', '138box', '139box', '140box', '141box', '142box', '143box', '144box', '145box', '146box', '147box', '148box', '149box', '150box']
        df = pd.DataFrame(final_data, columns=possible_columns[0:longest])
        df.to_csv("tfLite_results.csv")

        print("Average FPS: ", total_FPS / frames)
    elif choice == "v":
        labels = load_labels()
        ##print("labels:", labels)
        
        interpreter = Interpreter('model.tflite')
        interpreter.allocate_tensors()
        cap = cv2.VideoCapture("test.mp4")

        if (cap.isOpened()== False):
          print("Error opening video stream or file")
        
        width = int(cap.get(3))
        height = int(cap.get(4))
    
        out = cv2.VideoWriter('tfLite.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height))
        print("Starting video")
        frame_count = 1
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
              img = cv2.resize(frame, (640,640))
              start = time.time()
              res = detect_objects(interpreter, img, 0.0)
              finish = time.time()

              res_counter = 0
              for result in res:
                if result["score"] > 0.6:
                  res_counter += 1
                  ymin, xmin, ymax, xmax  = result['bounding_box']
                  xmin = int(max(1,xmin * width))
                  xmax = int(min(width, xmax * width))
                  ymin = int(max(1, ymin * height))
                  ymax = int(min(height, ymax * height))
                  
                  print(labels[int(result['class_id'])])
                  cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
                  lab_text = format(result['score']*100, "0.0f") + "% " + labels[int(result['class_id'])]
                  cv2.putText(frame,lab_text,(xmin, min(ymax, height-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
              print("frame: ", frame_count, " fps: ", 1 / (finish-start))
              frame_count += 1
              out.write(frame)
            else:
                break
        cap.release()
        out.release()
        print("done with video")
    elif choice == "c":
        labels = load_labels()
        ##print("labels:", labels)
        
        interpreter = Interpreter('model.tflite')
        interpreter.allocate_tensors()

        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640,640))
            start = time.time()
            res = detect_objects(interpreter, img, 0.5) #change threshold value here
            finish = time.time()
            print(1 / (finish - start))
            print(res) #prints list of detections in the frame

            #draw detections for display
            for result in res:
                ymin, xmin, ymax, xmax = result['bounding_box']
                xmin = int(max(1,xmin * CAMERA_WIDTH))
                xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
                ymin = int(max(1, ymin * CAMERA_HEIGHT))
                ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
                
                print(labels[int(result['class_id'])])
                cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
                cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 

            cv2.imshow('Pi Feed', frame)

            if cv2.waitKey(10) & 0xFF ==ord('q'):
                cap.release()
                cv2.destroyAllWindows()
    else:
        print("invalid operation, please enter i, v, or c") 

if __name__ == "__main__":
    main()