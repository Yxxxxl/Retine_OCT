import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import os

##################################读取文件#####################################
def get_all_files(file_path, is_random=True):
    """
    获取图片路径及其标签
    :param file_path: a sting, 图片所在目录
    :param is_random: True or False, 是否乱序
    :return:
    """
    image_list = []
    label_list = []
    num_files=0

    background_count = 0
    one_count = 0
    two_count = 0
    three_count = 0
    four_count = 0
    five_count = 0
    six_count = 0
    seven_count = 0
    eight_count = 0

    for fn in os.listdir(file_path):
        num_files += 1
    
    for i in range (num_files):
        k=i+1
        image_path=file_path+str(k)+'.png'
        image_list.append(image_path)
    
    
    for item in os.listdir(file_path):
        item_label = item.split('.')[0]  # 文件名形如  cat.0.jpg,只需要取第一个
#        if os.path.isfile(item_path):
#            image_list.append(item_path)
#        else:
#            raise ValueError('文件夹中有非文件项.')

        if item_label == 'background':  # 猫标记为'0'
            label_list.append(0)
            background_count += 1
        elif  item_label == 'one' : # 狗标记为'1'
            label_list.append(1)
            one_count += 1
        elif item_label == 'two':  # 狗标记为'1'
            label_list.append(2)
            two_count += 1
        elif  item_label == 'three' : # 狗标记为'1'
            label_list.append(3)
            three_count += 1
        elif  item_label == 'four' : # 狗标记为'1'
            label_list.append(4)
            four_count += 1
        elif  item_label == 'five' : # 狗标记为'1'
            label_list.append(5)
            five_count += 1
        elif  item_label == 'six' : # 狗标记为'1'
            label_list.append(6)
            six_count += 1
        elif  item_label == 'seven' : # 狗标记为'1'
            label_list.append(7)
            seven_count += 1
        else : # 狗标记为'1'
            label_list.append(8)
            eight_count += 1

    print('数据集中有%d张背景,%d张1，%d张2，%d张3，%d张4，%d张5，%d张6，%d张7，%d张8' % (background_count, one_count,two_count,three_count,four_count,five_count,six_count,seven_count,eight_count))

    image_list = np.asarray(image_list)
    label_list = np.asarray(label_list)
    # 乱序文件
    if is_random:
        rnd_index = np.arange(len(image_list))
        np.random.shuffle(rnd_index)
        image_list = image_list[rnd_index]
        label_list = label_list[rnd_index]

    return image_list,label_list

##################################组合数据#####################################
def get_batch(train_list, image_size, batch_size, capacity, is_random=True):
    """
    获取训练批次
    :param train_list: 2-D list, [image_list, label_list]
    :param image_size: a int, 训练图像大小
    :param batch_size: a int, 每个批次包含的样本数量
    :param capacity: a int, 队列容量
    :param is_random: True or False, 是否乱序
    :return:
    """

    intput_queue = tf.train.slice_input_producer(train_list, shuffle=False)

    # 从路径中读取图片
    image_train = tf.read_file(intput_queue[0])
    image_train = tf.image.decode_png(image_train, channels=3)  # 这里是jpg格式
    image_train = tf.image.resize_images(image_train, [image_size, image_size])
    image_train = tf.cast(image_train, tf.float32) / 255.  # 转换数据类型并归一化

    # 图片标签
    label_train = intput_queue[1]

    # 获取批次
    if is_random:
        image_train_batch, label_train_batch = tf.train.shuffle_batch([image_train, label_train],
                                                                      batch_size=batch_size,
                                                                      capacity=capacity,
                                                                      min_after_dequeue=100,
                                                                      num_threads=2)
    else:
        image_train_batch, label_train_batch = tf.train.batch([image_train, label_train],
                                                              batch_size=1,
                                                              capacity=capacity,
                                                              num_threads=1)
    return image_train_batch, label_train_batch

################################网络结构#######################################
def inference(images, n_classes):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')

    # full-connect1
    with tf.variable_scope("fc1") as scope:
        reshape = layers.flatten(norm2)
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
        # softmax_linear = tf.nn.softmax(softmax_linear)

    return softmax_linear



################################预测###########################################
def eval():
    N_CLASSES = 9
    IMG_SIZE = 208
    BATCH_SIZE = 1
    CAPACITY = 200
    MAX_STEP = 959*215

    test_dir = 'E:\\YxL\\Retine_Version1\\data\\test\\'
    logs_dir = 'logs_1'     # 检查点目录

    sess = tf.Session()

    train_list = get_all_files(test_dir, is_random=False)
    image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, False)
    train_logits = inference(image_train_batch, N_CLASSES)
    train_logits = tf.nn.softmax(train_logits)  # 用softmax转化为百分比数值

    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    all_p=np.zeros((MAX_STEP,9))
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
           
            image, prediction = sess.run([image_train_batch, train_logits])

            if step % 500 == 0:  # 实时记录训练过程并显示
               # runtime=time.time()
                print('Step: %6d,'% (step))

            all_p[step,:] = prediction
            
        return all_p
            #prediction = np.argmax(prediction)
#            if max_index == 0:
#                label = '%.2f%% is a background.' % (prediction[0][0] * 100)
#            elif max_index == 1:
#                label = '%.2f%% is a one.' % (prediction[0][1] * 100)
#            elif max_index == 2:
#                label = '%.2f%% is a two.' % (prediction[0][2] * 100)
#            elif max_index == 3:
#                label = '%.2f%% is a three.' % (prediction[0][3] * 100)
#            elif max_index == 4:
#                label = '%.2f%% is a four.' % (prediction[0][4] * 100)
#            elif max_index == 5:
#                label = '%.2f%% is a five.' % (prediction[0][5] * 100)
#            elif max_index == 6:
#                label = '%.2f%% is a six.' % (prediction[0][6] * 100)
#            elif max_index == 7:
#                label = '%.2f%% is a seven.' % (prediction[0][7] * 100)
#            else :
#                label = '%.2f%% is a eight.' % (prediction[0][8] * 100)
#
#            plt.imshow(image[0])
#            plt.title(label)
#            plt.show()
          

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()

##################################前期工作#####################################
OrImg = Image.open('E:\\YxL\\Retine_Version1\\0.jpg')
#Img.show()
##################################输出图片裁剪#################################
OrImg1 = OrImg.crop((0,100,1024,380))
Img_0 = OrImg1.crop((33,33,992,248))
plt.imshow(Img_0)
img = np.asarray(Img_0,dtype='float64')

##############################################################################
Img = OrImg.crop((0,100,1024,380))
Img = Img.convert('L')#转灰度图
Img = Img.crop((0,100,1024,380))#切出视网膜区域，减少运算量
#Img.show()

####按像素点切割小图
#print(Img.size)
#count=0
#all_batch = np.asarray([65,65,1024*280])

#k=0
#all_img =np.zeros([65,65,959*215],dtype = 'float32')
#for col in range(959):
#    for row in range(215):
#        a = Img.crop((col,row,col+65,row+65))
#        k+=1
#        file_dir = 'C:\\Users\\Administrator\\Desktop\\Try\\data\\test\\' + str(k)+'.png'
#        a.save(file_dir)
 
prediction=eval()

#####################################把每个像素点的不同class概率分开#############

L0 = np.zeros((215,959))
L1 = np.zeros((215,959))
L2 = np.zeros((215,959))
L3 = np.zeros((215,959))
L4 = np.zeros((215,959))
L5 = np.zeros((215,959))
L6 = np.zeros((215,959))
L7 = np.zeros((215,959))
L8 = np.zeros((215,959))


p=0
for i in range(959):
    for j in range(215):
        L0[j][i]=prediction[p][0]
        L1[j][i]=prediction[p][1]
        L2[j][i]=prediction[p][2]
        L3[j][i]=prediction[p][3]
        L4[j][i]=prediction[p][4]
        L5[j][i]=prediction[p][5]
        L6[j][i]=prediction[p][6]
        L7[j][i]=prediction[p][7]
        L8[j][i]=prediction[p][8]
        p+=1


M0=L0>0.8
M0=M0*255.

M1=L1>0.8
M1=M1*255.
#plt.imshow(M1)

M2=L2>0.8
M2=M2*255.
#plt.imshow(M2)

M3=L3>0.8
M3=M3*255.
#plt.imshow(M3)

M4=L4>0.8
M4=M4*255.
#plt.imshow(M4)
      
M5=L5>0.8
M5=M5*255.
#plt.imshow(M5)

M6=L6>0.8
M6=M6*255.
#plt.imshow(M6)

M7=L7>0.8
M7=M7*255.
#plt.imshow(M7)
       
M8=L8>0.8
M8=M8*255.
#plt.imshow(M8)
img[:,:,1]=M0
IMG=Image.fromarray(np.uint8(img))
plt.imshow(IMG)
plt.show()
      
img[:,:,1]=M1
IMG=Image.fromarray(np.uint8(img))
plt.imshow(IMG)
plt.show()

img[:,:,1]=M2
IMG=Image.fromarray(np.uint8(img))
plt.imshow(IMG)
plt.show()

img[:,:,1]=M3
IMG=Image.fromarray(np.uint8(img))
plt.imshow(IMG)
plt.show()

img[:,:,1]=M4
IMG=Image.fromarray(np.uint8(img))
plt.imshow(IMG)
plt.show()

img[:,:,1]=M5
IMG=Image.fromarray(np.uint8(img))
plt.imshow(IMG)
plt.show()

img[:,:,1]=M6
IMG=Image.fromarray(np.uint8(img))
plt.imshow(IMG)
plt.show()

img[:,:,1]=M7
IMG=Image.fromarray(np.uint8(img))
plt.imshow(IMG)
plt.show()

img[:,:,1]=M8
IMG=Image.fromarray(np.uint8(img))
plt.imshow(IMG)
plt.show()






