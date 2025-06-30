import tensorflow as tf
import math
from six.moves import xrange


#def lrelu(x, leak=0.2, name="lrelu"):
#    return tf.maximum(x, leak * x)


def batch_normalization(logits, scale, offset, isCovNet=False, name="bn"):
    
    if isCovNet:
        mean, var = tf.nn.moments(logits, [0, 1, 2]) ## dimension 3 is added on June 12th 2018--37th epoch onwards. 
        ##If visual quality improves, we will use it (may be to train from scratch)
    else:
        mean, var = tf.nn.moments(logits, [0])
       
    output = tf.nn.batch_normalization(logits, mean, var, offset, scale, variance_epsilon=1e-5)

    #### this is converting 5D tensor to 4D tensor to be able to use tf.nn.fused_batch_norm instead of tf.nn.moments
    ## as the latter gives negative variance sometimes---Added on 12th June 2018
    # orig_shape = tf.shape(logits)
    # # print(logits.shape[2])
    # # print(orig_shape.get_shape().as_list())
    # # print(orig_shape[0])
    # workaround_shape = orig_shape[0], orig_shape[1], orig_shape[2]*orig_shape[3], 1
    # logits = tf.reshape(logits, workaround_shape)
    # # print(logits.shape)
    # print(scale[0,:].shape)
    # logits, mean2, var2  = tf.nn.fused_batch_norm(logits, scale=scale[0,:], offset=offset[0,:], \
    #             mean=None, variance=None, epsilon=1e-5, data_format='NHWC', is_training=True)
    # output = tf.reshape(logits, orig_shape)
    return output


def get_conv_weights(weight_shape, sess, name="get_conv_weights"):
    # TODO:truncated_normal is not the same as randn
    return math.sqrt(2 / (9.0 * 64)) * sess.run(tf.truncated_normal(weight_shape))


def get_bn_weights(weight_shape, clip_b, sess, name="get_bn_weights"):
    weights = get_conv_weights(weight_shape, sess)
    return clipping(weights, clip_b)


def clipping(A, clip_b, name="clipping"):
    h, w = A.shape
    for i in xrange(h):
        for j in xrange(w):
            if A[i, j] >= 0 and A[i, j] < clip_b:
                A[i, j] = clip_b
            elif A[i, j] > -clip_b and A[i, j] < 0:
                A[i, j] = -clip_b
    return A
