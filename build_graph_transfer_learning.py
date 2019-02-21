import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from datetime import datetime
import os

graphDir = './inception-2015-12-05/classify_image_graph_def.pb'

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'
BOTTLENECK_TENSOR_SIZE = 2048
learning_rate = 0.01
eval_step_interval = 20
how_many_training_steps = 10000

classes = ['dog', 'fish']



def create_graph(graphDir=graphDir):
    """"Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    print('Loading graph...')
    with tf.Session() as sess:
        with gfile.FastGFile(graphDir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    print('Done...')
    return sess.graph


def add_final_training_ops():


    layer_weights = tf.Variable(
        tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, len(classes)], stddev=0.001),
        name='final_weights')
    layer_biases = tf.Variable(tf.zeros([len(classes)]), name='final_biases')

    logits = tf.add(tf.matmul(X_Bottleneck, layer_weights, name='final_matmul'), layer_biases, name="logits")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_true)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean_2class')
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, logits


def add_evaluation_step(Ylogits):
    """Inserts the operations we need to evaluate the accuracy of our results.
    Args:
      graph: Container for the existing model's Graph.
    Returns:
      Nothing.
    """
    correct_prediction = tf.equal(tf.argmax(Ylogits, 1),
                                  tf.argmax(Y_true, 1))  # tf.equal(tf.argmax(Ylogits, 1), Y_true)#
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='eval_step_2class')
    return evaluation_step




tf.reset_default_graph()
sess = tf.Session()
graph = create_graph()
#bottleneck_tensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME + ':0')

X_Bottleneck = tf.placeholder(tf.float32, shape=[None, BOTTLENECK_TENSOR_SIZE], name="X_bottleneck")
# placeholder for true labels
Y_true = tf.placeholder(tf.float32, [None, len(classes)], name="Y_true")

train_step, cross_entropy, Ylogits = add_final_training_ops()
evaluation_step = add_evaluation_step(Ylogits)
init = tf.initialize_all_variables()
# Create a saver object which will save all the variables
saver = tf.train.Saver()
sess.run(init)
# save the graph for cold start
if not os.path.exists('./dog_v_fish_cold_graph/'):
    os.makedirs('./dog_v_fish_cold_graph/')
saver.save(sess, './dog_v_fish_cold_graph/dog_v_fish_cold_graph')

