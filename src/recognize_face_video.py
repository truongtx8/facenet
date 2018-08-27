import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
import pickle
import facenet
import align.detect_face

import tfinit
from scipy import misc
from facenet import load_img

parser = argparse.ArgumentParser()

parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
parser.add_argument('--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5)
parser.add_argument('--imagePath', type=str,
        help='Source image location.')
parser.add_argument('--imagePathw', type=str,
        help='Destination image location.')

args = parser.parse_args()

# Get user supplied values Loading the stored face embedding vectors for image retrieval
#imagePath  = '/data/0/home/truongtx8/datasets/test/IMG-013.jpg'
#imagePathw = '/data/0/home/truongtx8/datasets/test/face_IMG-013.png'
imagePath  = args.imagePath
imagePathw = args.imagePathw


with open('/data/0/home/truongtx8/models/embd/dirox.pickle','rb') as f:
        feature_array = pickle.load(f)

model_exp = '/data/0/home/truongtx8/models/20180408-102900'
#graph_fr = tf.Graph()
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
#
#sess_fr = tf.Session(config=config,graph=graph_fr)
#with graph_fr.as_default():
#        saverf = tf.train.import_meta_graph(os.path.join(model_exp, 'model-20180408-102900.meta'))
#        saverf.restore(sess_fr, os.path.join(model_exp, 'model-20180408-102900.ckpt-90'))
#        pnet, rnet, onet = align.detect_face.create_mtcnn(sess_fr, None)
sess_tf, pnet, rnet, onet = tfinit.tf_init(model_exp, 'model-20180408-102900.meta', 'model-20180408-102900.ckpt-90', args.gpu_memory_fraction)
print('Loaded pre-trained models in', model_exp)

def align_face (img, pnet, rnet, onet):
        minsize = 20 # minimum size of face
        threshold = [ 0.7, 0.8, 0.8 ]  # three steps's threshold
        factor = 0.709 # scale factor

        #print("before img.size == 0")
        if img.size == 0:
                print("empty array")
                return False,img,[0,0,0,0]

        if img.ndim<2:
            print('Unable to align')

        if img.ndim == 2:
            img = to_rgb(img)

        img = img[:,:,0:3]

        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        nrof_faces = bounding_boxes.shape[0]


        if nrof_faces==0:
            return False,img,[0,0,0,0]
        else:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                if args.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))
            if len(det_arr)>0:
                faces = []
                bboxes = []
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-args.margin/2, 0)
                    bb[1] = np.maximum(det[1]-args.margin/2, 0)
                    bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                    #misc.imsave("cropped.png", scaled)
                    faces.append(scaled)
                    bboxes.append(bb)
                    #print("leaving align face")
                return True,faces,bboxes


def identify_person(image_vector, feature_array, k=9):
        top_k_ind = np.argsort([np.linalg.norm(image_vector - pred_row) \
                        for ith_row, pred_row in enumerate(feature_array.values())])[:k]
        #print(np.linalg.norm(image_vector))
        #for ith_row, pred_row in enumerate(feature_array.values()):
            #print(pred_row)
            #print(np.linalg.norm(image_vector - pred_row))
        #print([np.linalg.norm(image_vector - pred_row) for ith_row, pred_row in enumerate(feature_array.values())])
        #print(np.linalg.norm(image_vector - list(feature_array.values())[top_k_ind[0]]), np.linalg.norm(image_vector - list(feature_array.values())[top_k_ind[1]]))
        #print(feature_array.keys())

        diff = np.linalg.norm(image_vector - list(feature_array.values())[top_k_ind[0]])
        #result = list(feature_array.keys())[top_k_ind[0]]
        result = []
        #print(result, top_k_ind)
        #print(list(feature_array.keys())[top_k_ind[0]] + "\n" + list(feature_array.keys())[top_k_ind[1]])
        #result = result.[top_k_ind[0]]
        for i in range(k):
            #print(i)
            #result = list(feature_array.keys())[top_k_ind[i]]
            #print(list(feature_array.keys())[top_k_ind[i]])
            result.extend([list(feature_array.keys())[top_k_ind[i]]])
            #print (np.linalg.norm(image_vector - list(feature_array.values())[top_k_ind[i]]), result[i])

        #if (result[0].split("/")[8] == result[1].split("/")[8]) and (diff < 1.000):
        #    return result[0], diff
        if (diff < 1.000):
            return result[0], diff
        else:
            return 'unknown/unknown_000.png', diff


def recognize_face (sess, pnet, rnet, onet,feature_array):
        # Get input and output tensors
        images_placeholder = sess.graph.get_tensor_by_name("input:0")
        images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
        embeddings = sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

        image_size = args.image_size
        embedding_size = embeddings.get_shape()[1]


        cap = cv2.VideoCapture(imagePath)
        
        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        #out = cv2.VideoWriter(imagePathw,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        out = cv2.VideoWriter(imagePathw, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (1920, 1080))
        
        while(True):
            ret, gray = cap.read()
            #gray = cv2.cvtColor(frame, 0)
            gray = cv2.resize(gray, (1920, 1080))

            if ret == True:
                    #print(gray.size)
                    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    response, faces, bboxs = align_face(gray, pnet, rnet, onet)
                    #print(response)
                    if (response == True):
                        for i, image in enumerate(faces):
                            bb = bboxs[i]
                            images = load_img(image, False, False, image_size)
    
                            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                            #feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                            feature_vector = sess.run(embeddings, feed_dict)
    
                            result, diff = identify_person(feature_vector, feature_array, 5)
                            #print(result.split("/")[2])
    
                            W = int(bb[2]-bb[0])//2
                            H = int(bb[3]-bb[1])//2
    
                            if(result.split("/")[0] == 'unknown'):
                                #cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0),2)
                                #cv2.rectangle(gray,(bb[0]-1,bb[1]),(bb[2]+1,bb[1]-18),(255,0,0),cv2.FILLED)
                                #cv2.rectangle(gray,(bb[0]-1,bb[3]),(bb[2]+1,bb[3]+18),(255,0,0),cv2.FILLED)
                                #cv2.putText(gray, result.split("/")[0] + " " + str('%.2f' % diff),(bb[0],bb[1]-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,191),1,cv2.LINE_AA)
                                #cv2.putText(gray, str(diff),(bb[0],bb[3]+10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                                putTextName(gray, bb, result.split("/")[0], str(diff), (0,0,255))
                                print("\x1b[1;31;40mUnknown " + "\x1b[0m" + result.split("/")[1])
                            else:
                                #cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),2)
                                #cv2.rectangle(gray,(bb[0]-1,bb[1]),(bb[2]+1,bb[1]-18),(0,255,0),cv2.FILLED)
                                #cv2.rectangle(gray,(bb[0]-1,bb[3]),(bb[2]+1,bb[3]+18),(0,255,0),cv2.FILLED)
                                #cv2.putText(gray, result.split("/")[0] + " " + str('%.2f' % diff),(bb[0],bb[1]-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
                                #cv2.putText(gray, str(diff),(bb[0],bb[3]+10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
                                putTextName(gray, bb, result.split("/")[0], str(diff), (255,255,127))
                                print("\x1b[1;32;40mMatched " + "\x1b[0m" + result.split("/")[1])

                    #gray = cv2.resize(gray, (1280, 720))
                    out.write(gray)

def putTextName (img, bboxs, text1, text2, color):
    textScale = 1.0
    #cv2.line(img,(0,0),(511,511),(255,0,0),5)
    cv2.line(img, (bboxs[0]+10, bboxs[1]+10), (bboxs[0]-10, bboxs[1]-10), color, 2)
    cv2.line(img, (bboxs[0]-10, bboxs[1]-10), (bboxs[0]-30, bboxs[1]-10), color, 2)
    cv2.line(img, (bboxs[0]-30, bboxs[1]-10), (bboxs[0]-30, bboxs[1]-80), color, 2)

    cv2.putText(img, text1, (bboxs[0]-5, bboxs[1]-50), cv2.FONT_HERSHEY_SIMPLEX, textScale, color, 2, cv2.LINE_AA)
    cv2.putText(img, text2, (bboxs[0]-5, bboxs[1]-25), cv2.FONT_HERSHEY_SIMPLEX, textScale/2, color, 1, cv2.LINE_AA)

# Recognize face
recognize_face (sess_tf, pnet, rnet, onet, feature_array)

#def main(args):
#    recognize_face (sess_fr, pnet, rnet, onet, feature_array)
#
