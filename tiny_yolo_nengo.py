import nengo
import tensorflow as tf
import nengo_dl
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPool2D
from yolov3.dataset import Dataset

tile_count = 7
bounding_boxes = 2
unique_classifications = 10
YOLO_STRIDES = [8, 16, 32]
STRIDES         = np.array(YOLO_STRIDES)

YOLO_ANCHORS = [[[10,  13], [16,   30], [33,   23]],
                [[30,  61], [62,   45], [59,  119]],
                [[116, 90], [156, 198], [373, 326]]]

ANCHORS = (np.array(YOLO_ANCHORS).T/STRIDES).T
YOLO_IOU_LOSS_THRESH = 0.5

YOLO_MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]

_MODEL_SIZE = (416, 416)

def decode_tensor(conv_output, NUM_CLASS, i=0):
    #print(conv_output.shape)
    # where i = 0, 1 or 2 to correspond to the three grid scales  
    #conv_output = tf.reshape(conv_output, shape)
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    #conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5+NUM_CLASS))
    #conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position     
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
    conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
    conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box 

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size,dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

    # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula  
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

#def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=YOLO_MNIST_CLASSES):
def compute_loss(labels,flattened_prediction_tensor):
    """
    grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
    """
    l1 = tf.reshape(labels[0,0,0:30420],(1,26,26,3,15))
    l2 = tf.reshape(labels[0,0,30420:30420+400],(1,100,4))
    l3 = tf.reshape(labels[0,0,30420+400:30420+400+7605],(1,13,13,3,15))
    l4 = tf.reshape(labels[0,0,30420+400+7605:],(1,100,4))
    #adjusted loss computation function, fixed to only 2 inputs (estimate and target)
    orig_labels = [[l1,l2],[l3, l4]]
    CLASSES=YOLO_MNIST_CLASSES
    #orig_conv = [[],[]]
    #print(flattened_prediction_tensor.shape)
    orig_conv1 = tf.TensorArray('float32', size = flattened_prediction_tensor.shape[0])
    orig_conv2 = tf.TensorArray('float32', size = flattened_prediction_tensor.shape[0])
    counter = 0
    for i in flattened_prediction_tensor:
        #currently hard=coded for only 1 minibatch
        #orig_conv[0].append(tf.reshape(i[0,0:30420],(26,26,45)))
        ia, ib = tf.split(i[0], [30420,7605])
        #orig_conv[0].append(tf.reshape(ia,(1,26,26,3,15)))
        #orig_conv[1].append(tf.reshape(ib, (1,13,13,3,15)))
        orig_conv1.write(counter, tf.reshape(ia, (1,26,26,3,15)))
        #counter+=1
        orig_conv2.write(counter, tf.reshape(ib,(1,13,13,3,15)))
        counter+=1
                
    total_loss = 0
    
    for i in range(2):
        if i==0:
            conv = orig_conv1.read(0)#orig_conv[i][0]
        else:
            conv = orig_conv2.read(0)
        pred = decode_tensor(conv, 10, i)

        label = orig_labels[i][0]
        bboxes = orig_labels[i][1]
        
        NUM_CLASS = len(CLASSES)
        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = STRIDES[i] * output_size
        #conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]
    
        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]
    
        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]
    
        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)
    
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
    
        iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        # Find the value of IoU with the real box The largest prediction box
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    
        # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < YOLO_IOU_LOSS_THRESH, tf.float32 )
    
        conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    
        # Calculate the loss of confidence
        # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )
    
        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
    
        total_loss += tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        total_loss += tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        total_loss += tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return total_loss
print('building model')
model = nengo.Network()

with model as net:
    #size_in = 1, size_out = 3
    #input_layer = nengo.Node(size_in = 1, size_out = 3, output = tf.keras.layers.Reshape((416,416,3)), label = 'input-image')
    input_layer = nengo.Node(output = np.ones(416*416*3), label = 'input image',)
    node_list = [input_layer]
    #node_list = []
    inp_sec = nengo_dl.TensorNode(tf.keras.layers.Reshape((416,416,3)), shape_in = (1,416*416*3,1), shape_out = (416,416,3), pass_time = False)
    node_list.append(inp_sec)
    prev_shape = (416,416,3) 
    NUM_CLASSES = 10
    conv_shapes = [(3,16), (3,32), (3,64), (3,128), (3,256)]
    strides = 1; padding = 'same'; activate=True; bn=True;
    for i, filters_shape in enumerate(conv_shapes):
        next_shape = (prev_shape[0], prev_shape[0], filters_shape[1])
        conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))
        node_list.append(nengo_dl.TensorNode(conv, pass_time = False, label = 'DN_conv {}'.format(i),shape_in = prev_shape, shape_out = next_shape))
        node_list.append(nengo_dl.TensorNode(BatchNormalization(),  pass_time = False, label = 'DN_BN {}'.format(i),
            shape_in = node_list[-1].shape_out, shape_out = next_shape))
        node_list.append(nengo_dl.TensorNode(LeakyReLU(alpha=0.1),  pass_time = False, label = 'DN_Act {}'.format(i),
            shape_in = node_list[-1].shape_out, shape_out = next_shape))
        if i==4:
            DN_route_1 = node_list[-1]
        node_list.append(nengo_dl.TensorNode(MaxPool2D(2,2,'same'),  pass_time = False, label = 'DN_MxPool {}'.format(i), 
            shape_in = node_list[-1].shape_out, shape_out = (next_shape[0]//2, next_shape[1]//2, next_shape[2])))
        prev_shape = node_list[-1].shape_out
    
    i+=1
    filters_shape = (3,512)
    next_shape = (prev_shape[0], prev_shape[1], filters_shape[1])
    conv = Conv2D(filters=filters_shape[-1],  kernel_size = filters_shape[0], strides=strides,
              padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
              bias_initializer=tf.constant_initializer(0.))
    node_list.append(nengo_dl.TensorNode(conv,  pass_time = False, label = 'DN_conv {}'.format(i),shape_in = prev_shape, shape_out = next_shape))
    node_list.append(nengo_dl.TensorNode(BatchNormalization(),  pass_time = False, label = 'DN_BN {}'.format(i),
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    node_list.append(nengo_dl.TensorNode(LeakyReLU(alpha=0.1),  pass_time = False, label = 'DN_Act {}'.format(i),
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    node_list.append(nengo_dl.TensorNode(MaxPool2D(2,1,'same'),  pass_time = False, label = 'DN_MxPool {}'.format(i), 
        shape_in = node_list[-1].shape_out, shape_out = (next_shape[0], next_shape[1], next_shape[2])))
    prev_shape = node_list[-1].shape_out
    
    i+=1
    filters_shape = (3,1024)
    next_shape = (prev_shape[0], prev_shape[0], filters_shape[1])
    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
              padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
              bias_initializer=tf.constant_initializer(0.))
    node_list.append(nengo_dl.TensorNode(conv,  pass_time = False, label = 'DN_conv {}'.format(i),shape_in = prev_shape, shape_out = next_shape))
    node_list.append(nengo_dl.TensorNode(BatchNormalization(),  pass_time = False, label = 'DN_BN {}'.format(i),
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    node_list.append(nengo_dl.TensorNode(LeakyReLU(alpha=0.1),  pass_time = False, label = 'DN_Act {}'.format(i),
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    prev_shape = next_shape
    DN_route_2 = node_list[-1]    
    
    #print('full route shape: ',DN_route_2.shape_out)
    #should be 13,13,1024
    #print('route1 shape: ', DN_route_1.shape_out)
    #should be 26,26,512
    #--------------------------------end of darknet routine----------------------------------#    
    i=0
    filters_shape = (1,256)    
    next_shape = (prev_shape[0], prev_shape[0], filters_shape[1])   
    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
              padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
              bias_initializer=tf.constant_initializer(0.))
    node_list.append(nengo_dl.TensorNode(conv,  pass_time = False, label = 'PostDN_conv',shape_in = prev_shape, shape_out = next_shape))
    node_list.append(nengo_dl.TensorNode(BatchNormalization(),  pass_time = False, label = 'PostDN_BN',
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    node_list.append(nengo_dl.TensorNode(LeakyReLU(alpha=0.1),  pass_time = False, label = 'PostDN_Act',
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    prev_shape = next_shape
    checkpoint = node_list[-1]
    
    #large bound box route
    filters_shape = (3,512)
    next_shape = (prev_shape[0], prev_shape[0], filters_shape[1])   
    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
              padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
              bias_initializer=tf.constant_initializer(0.))
    node_list.append(nengo_dl.TensorNode(conv,  pass_time = False, label = 'LBB_conv',shape_in = prev_shape, shape_out = next_shape))
    node_list.append(nengo_dl.TensorNode(BatchNormalization(),  pass_time = False, label = 'LBB_BN',
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    node_list.append(nengo_dl.TensorNode(LeakyReLU(alpha=0.1),  pass_time = False, label = 'LBB_Act',
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    prev_shape = next_shape
    
    filters_shape = (1,3*(NUM_CLASSES + 5))
    next_shape = (prev_shape[0], prev_shape[1], filters_shape[1])
    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
              padding=padding, use_bias=bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
              bias_initializer=tf.constant_initializer(0.))
    node_list.append(nengo_dl.TensorNode(conv,  pass_time = False, label = 'CONV_LBBOX',shape_in = prev_shape, shape_out = next_shape))
    
    CONV_LBBOX = node_list[-1]
    #print('conv-lbbox size: ',CONV_LBBOX.shape_out)
    
    for i in range(len(node_list)-1):
        nengo.Connection(node_list[i], node_list[i+1], synapse = None)
    
    prev_shape = checkpoint.shape_out
    filters_shape = (1,128)
    next_shape = (prev_shape[0], prev_shape[0], filters_shape[1])   
    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
              padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
              bias_initializer=tf.constant_initializer(0.))
    node_list.append(nengo_dl.TensorNode(conv,  pass_time = False, label = 'MBB_conv1',shape_in = prev_shape, shape_out = next_shape))
    nengo.Connection(checkpoint, node_list[-1], synapse = None)
    node_list.append(nengo_dl.TensorNode(BatchNormalization(),  pass_time = False, label = 'MBB_BN1',
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    nengo.Connection(node_list[-2], node_list[-1], synapse = None)
    node_list.append(nengo_dl.TensorNode(LeakyReLU(alpha=0.1),  pass_time = False, label = 'MBB_Act1',
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    nengo.Connection(node_list[-2], node_list[-1], synapse = None)
    prev_shape = next_shape    
    pre_concat_shape = prev_shape
    #print(DN_route_1.shape_out)
    def concatenate(x):
        x0 = x[:, :np.prod(pre_concat_shape)]
        x1 = x[:, np.prod(pre_concat_shape):]
        x0 = tf.image.resize(tf.keras.layers.Reshape((13, 13, 128))(x0),(26,26), method = 'nearest')
        x1 = tf.keras.layers.Reshape((26, 26, 256))(x1)
        y = tf.keras.layers.Concatenate(axis=-1)([x0, x1])
        return y
    
    node_list.append(nengo_dl.TensorNode(concatenate,  pass_time = False, shape_in = (np.prod(prev_shape)+np.prod(DN_route_1.shape_out),), shape_out = (26,26,384), ))
    concat_node = node_list[-1]
    nengo.Connection(node_list[-2], node_list[-1][:np.prod(pre_concat_shape)], synapse = None)
    nengo.Connection(DN_route_1, node_list[-1][np.prod(pre_concat_shape):], synapse = None)
    prev_shape = node_list[-1].shape_out
    #print('concatenated shape: ',prev_shape)
    
    filters_shape = (3,256)
    next_shape = (prev_shape[0], prev_shape[1], filters_shape[1])
    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
              padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
              bias_initializer=tf.constant_initializer(0.))
    node_list.append(nengo_dl.TensorNode(conv,  pass_time = False, label = 'MBB_conv2',shape_in = prev_shape, shape_out = next_shape))
    nengo.Connection(node_list[-2], node_list[-1], synapse = None)    
    node_list.append(nengo_dl.TensorNode(BatchNormalization(),  pass_time = False, label = 'MBB_BN2',
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    nengo.Connection(node_list[-2], node_list[-1], synapse = None)
    node_list.append(nengo_dl.TensorNode(LeakyReLU(alpha=0.1),  pass_time = False, label = 'MBB_Act2',
        shape_in = node_list[-1].shape_out, shape_out = next_shape))
    nengo.Connection(node_list[-2], node_list[-1], synapse = None)
    prev_shape = next_shape  
    
    filters_shape = (1,3*(NUM_CLASSES+5))
    next_shape = (prev_shape[0], prev_shape[1], filters_shape[1])
    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
              padding=padding, use_bias= bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
              bias_initializer=tf.constant_initializer(0.))
    node_list.append(nengo_dl.TensorNode(conv, pass_time = False,  label = 'CONV_MBBOX',shape_in = prev_shape, shape_out = next_shape))
    nengo.Connection(node_list[-2], node_list[-1], synapse = None)    
    
    CONV_MBBOX = node_list[-1]
    #print('conv-mbbox size: ', CONV_MBBOX.shape_out)

    final_node2 = nengo.Node(size_in = np.prod(CONV_MBBOX.shape_out)+np.prod(CONV_LBBOX.shape_out))
    nengo.Connection(CONV_MBBOX, final_node2[:np.prod(CONV_MBBOX.shape_out)], synapse = None)
    nengo.Connection(CONV_LBBOX, final_node2[np.prod(CONV_MBBOX.shape_out):], synapse = None)
    
    out_probe = nengo.Probe(final_node2, label = 'out_p')
      
print('model finished building')
minibatch_size = 1

print('importing and reshaping data')
trainset = Dataset('train')
testset = Dataset('test')

train_images = []
train_labels = []
for i, j in trainset:
    #j[0][0] is shape (4,52,52,3,15), not used
    #j[1][0] is shape (4,26,26,3,15), medium
    #j[2][0] is shape (4,13,13,3,15), large
    #in each j[index], there exist 2 objects; one of above shape, second of size (4,100,4)
    for ind, ii in enumerate(i):
        train_images.append(np.array(ii[:,:,:]).flatten())
        train_labels.append([[j[1][0][ind], j[1][1][ind]],
         [j[2][0][ind],j[2][1][ind]]])
    """
    for ind in range(4):        
        train_images.append(i[ind])
        m = j[1][ind].flatten()
        l = j[2][ind].flatten()
        train_labels.append(np.concatenate((m,l)))
    """
    
train_images = np.array(train_images)
train_labels = np.array([np.concatenate((i[0][0].flatten(), i[0][1].flatten(), 
                                         i[1][0].flatten(),i[1][1].flatten()),axis = -1) for i in train_labels])
print('data successfully improted')
#ti1 = [np.array([[i]]) for i in train_images[0:10]]
ti1 = np.array(train_images)#[:,None,:]
ti1 = ti1.reshape((ti1.shape[0], 1, -1))
#ti1 = train_images[0:10]
tl1 = train_labels
tl1 = tl1.reshape((tl1.shape[0], 1, -1))

print('building simulator')
#each label is:
#   (26,26,3,15), (100, 4), (13,13,3,15), (100,4)
with nengo_dl.Simulator(net, minibatch_size=1) as sim:
    sim.compile(
        optimizer=tf.optimizers.Adam(learning_rate = .01),
        loss=compute_loss,
        #loss={'y': compute_loss}
    )
    sim.fit(x = ti1, y = tl1, epochs=10, )
        
