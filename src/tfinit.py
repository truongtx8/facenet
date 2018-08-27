#
import os
import tensorflow as tf
import align.detect_face

#
def tf_init(model_exp, model_meta, model_data, gpu_memory_fraction):
    #model_exp = '/data/0/home/truongtx8/models/20180408-102900'
    graph_fr = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    sess_tf = tf.Session(config=config,graph=graph_fr)
    with graph_fr.as_default():
        saverf = tf.train.import_meta_graph(os.path.join(model_exp, model_meta))
        saverf.restore(sess_tf, os.path.join(model_exp, model_data))
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess_tf, None)

    return sess_tf, pnet, rnet, onet
