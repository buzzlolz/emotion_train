import tensorflow as tf
import numpy as np
import dlib
from tensorflow.python.platform import gfile
# from contextlib import contextmanager
import argparse
import time
from pathlib import Path
import threading
# OpenCV
import cv2
import queue




def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.6,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
                      
    args = parser.parse_args()
    return args

    
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


# @contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)
        print("img path",str(image_path))
        #cv2.imshow("result", img)
        #cv2.waitKey(0)   

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

def tracker_new():
    
    tracker=cv2.TrackerKCF_create()
    return tracker
    
    


            
            
def main():
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir
    
    #videoCapture = cv2.VideoCapture('oto.avi')

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #videoWriter = cv2.VideoWriter('out.mp4',fourcc, 10, (640,480))
    que = queue.Queue()
    detector = dlib.get_frontal_face_detector()
    #image_generator=yield_images() 
    rd_img =threading.Thread(target=lambda q,arg1:q.put(yield_images_from_dir(image_dir) if image_dir else yield_images()),args=(que,'img_d'))
    #image_generator=yield_images_from_dir(image_dir) if image_dir else yield_images()
    rd_img.start()
    image_generator=que.get()
    # tracker=tracker_new()

    
    
    detect_e=tf.Graph()
    with detect_e.as_default():
        e_graph_def=tf.GraphDef()
        with tf.gfile.GFile(weight_file,'rb') as f:
            serialized_graph = f.read()
            e_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(e_graph_def,name='')
            
    e_seesion=tf.Session(graph=detect_e)
    img_size = 224
    pred_emo= detect_e.get_tensor_by_name("pred_emo/Softmax:0")
    
    count=0
    dlib_thread=threading.Thread()
    for img in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)
            

            detected = detector(input_img, 1)
            #print (detected)
            faces = np.empty((len(detected), img_size, img_size, 3))
            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                    # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                
                print("len(detect)",len(detected))    
                start = time.time()
                res_emo = e_seesion.run(pred_emo, feed_dict={'input_1:0': faces})
              
                #print(res_age)
                #res_gen = sess.run(pred_gen, feed_dict={'input_1:0': faces})
                
                end = time.time()
               

               
                for i, d in enumerate(detected):
                    if max(res_emo[i][0],res_emo[i][1],res_emo[i][2])==res_emo[i][0]:
                        label ="happy"
                    elif max(res_emo[i][0],res_emo[i][1],res_emo[i][2])==res_emo[i][1]:
                        label = "nertual"
                    else :
                        label="others"
                    #label = "{}".format("happy"if max(res_emo[i][0],res_emo[i][1],res_emo[i][2])==res_emo[i][0] else "others")
                    #draw_label(img, (d.left(), d.top()), label)
                    print("emo label",label)
                    #draw_label(img, (d.left(), d.top()), label)
                    
                    draw_label(img, (d.left(), d.top()), label)
                cv2.imshow("result", img)
            #key = cv2.waitKey(30)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break    

   
            
           
if __name__ == '__main__':
    main()