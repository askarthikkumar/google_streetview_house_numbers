import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def convert_labels(labels):
    for i in range(len(labels)):
        if len(labels[i])==2:
            labels[i].append(10)
        elif len(labels[i])==1:
            labels[i].append(10)
            labels[i].append(10)
    return labels

with open('whole_data.pickle','rb') as f:
    data_obj=pickle.load(f)
train_images=np.asarray(data_obj['train_data'][0],np.float32)
train_labels=np.asarray(convert_labels(data_obj['train_data'][1]),np.int32)
test_images=np.asarray(data_obj['test_data'][0],np.float32)
test_labels=np.asarray(convert_labels(data_obj['test_data'][1]),np.int32)
extra_images=np.asarray(data_obj['extra_data'][0],np.float32)
extra_labels=np.asarray(convert_labels(data_obj['extra_data'][1]),np.int32)
whole_train=[train_images,train_labels]
whole_test=[test_images,test_labels]
whole_extra=[extra_images,extra_labels]
whole_test[0]=np.reshape(whole_test[0][:],[whole_test[0].shape[0],32,32,1])
whole_train[0]=np.reshape(whole_train[0][:],[whole_train[0].shape[0],32,32,1])
whole_extra[0]=np.reshape(whole_extra[0][:],[whole_extra[0].shape[0],32,32,1])

def accuracy(predictions, labels):
    count = predictions.shape[1]
    return 100.0 * (count - np.sum([1 for i in np.argmax(predictions, 2).T == labels[:,0:3] if False in i])) / count

graph = tf.Graph()

with graph.as_default():
    loss=tf.Variable([0],dtype=tf.float32,name='LOSS')
    with tf.name_scope('Pipelines'):
        train_input_queue=tf.train.slice_input_producer([whole_train[0],whole_train[1]],shuffle=True)
        valid_input_queue=tf.train.slice_input_producer([whole_extra[0],whole_extra[1]],shuffle=True)
        train_image=train_input_queue[0]
        train_label=train_input_queue[1]
        valid_image=valid_input_queue[0]
        valid_label=valid_input_queue[1]
        #I HAVE INTERCHANGED VALIDATION AND TRAINING DATA
        image_batch1,label_batch1=tf.train.batch([train_image,train_label],batch_size=64,capacity=100+3*64,enqueue_many=False,name='validation_pipe')
        image_batch,label_batch=tf.train.batch([valid_image,valid_label],batch_size=64,capacity=100+3*64,enqueue_many=False,name='training_pipe')
        tf.summary.image('images',image_batch)
        train_data,train_labels=image_batch,label_batch
        valid_dataset=image_batch1
    def initial(shape): 
        return tf.constant(0.1, shape = shape)
    with tf.name_scope('global_vars'):
        global_step = tf.Variable(0)
#     with tf.name_scope('conv1'):
        weights1=tf.get_variable('weights1',shape=[5,5,1,16],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        tf.summary.histogram('weights1',weights1)
        bias1=tf.Variable(initial([16]),name='bias1')
        tf.summary.histogram('bias1',bias1)
#     with tf.name_scope('conv2'):
        weights2=tf.get_variable('weights2',shape=[5,5,16,32],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        tf.summary.histogram('weights2',weights2)
        bias2=tf.Variable(initial([32]),name='bias2')
        tf.summary.histogram('bias2',bias2)
#     with tf.name_scope('conv3'):
        weights3=tf.get_variable('weights3',shape=[5,5,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias3=tf.Variable(initial([64]),name='bias3')
        tf.summary.histogram('weights3',weights3)
        tf.summary.histogram('bias3',bias3)
#     with tf.name_scope('fc'):
        fc_weights1=tf.get_variable('fc_weights1',shape=[1024,1024],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        fc_bias1=tf.Variable(initial([1024]),name='fc_bias1')
        tf.summary.histogram('fc_weights1',fc_weights1)
        tf.summary.histogram('fc_bias1',fc_bias1)
#     with tf.name_scope('d1'):
        d1_weights=tf.get_variable('d1_weights',shape=[1024,11],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        d1_bias=tf.Variable(initial([11]),name='d1_bias')
        tf.summary.histogram('d1_weights',d1_weights)
        tf.summary.histogram('d1_bias',d1_bias)
#     with tf.name_scope('d2'):
        d2_weights=tf.get_variable('d2_weights',shape=[1024,11],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        d2_bias=tf.Variable(initial([11]),name='d2_bias')
        tf.summary.histogram('d2_weights',d2_weights)
        tf.summary.histogram('d2_bias',d2_bias)
#     with tf.name_scope('d3'):
        d3_weights=tf.get_variable('d3_weights',shape=[1024,11],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        d3_bias=tf.Variable(initial([11]),name='d3_bias')
        tf.summary.histogram('d3_weights',d3_weights)
        tf.summary.histogram('d3_bias',d3_bias)
    #build the model
    def model(data,k):
        #in layer one the image is of the form 32*32
        with tf.name_scope('conv1'):
            conv1=tf.nn.relu(tf.nn.conv2d(data,weights1,[1,1,1,1],padding='SAME',name='conv1')+bias1)
            tf.summary.histogram('conv1',conv1)
            lrn1=tf.nn.local_response_normalization(conv1,name='lrn1')
            max1=tf.nn.max_pool(lrn1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #in layer two the image is in the form 16*16
        with tf.name_scope('conv2'):
            conv2=tf.nn.relu(tf.nn.conv2d(max1,weights2,[1,1,1,1],padding='SAME',name='conv2')+bias2)
            tf.summary.histogram('conv2',conv2)
            lrn2=tf.nn.local_response_normalization(conv2,name='lrn2')
            max2=tf.nn.max_pool(lrn2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #in layer three the image is in the form 8*8
        with tf.name_scope('conv3'):
            conv3=tf.nn.relu(tf.nn.conv2d(max2,weights3,[1,1,1,1],padding='SAME',name='conv3')+bias3)
            tf.summary.histogram('conv3',conv3)
            lrn3=tf.nn.local_response_normalization(conv3,name='lrn3')
            max3=tf.nn.max_pool(lrn3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #output of layer three is of the form 1*4*4*64
        #therefore it has total of 1024 nodes...we need to flatten them to construct our fc1
        with tf.name_scope('dropout'):
            max3_reshape=tf.reshape(max3,[64,1024])
            max3_dropout=tf.nn.dropout(max3,keep_prob=k)
        with tf.name_scope('fc'):
            fc1=tf.matmul(max3_reshape,fc_weights1)+fc_bias1
            tf.summary.histogram('fc1',fc1)
        with tf.name_scope('d1'):
            d1=tf.matmul(fc1,d1_weights)+d1_bias
            tf.summary.histogram('d1',d1)
        with tf.name_scope('d2'):
            d2=tf.matmul(fc1,d2_weights)+d2_bias
            tf.summary.histogram('d2',d2)
        with tf.name_scope('d3'):
            d3=tf.matmul(fc1,d3_weights)+d3_bias
            tf.summary.histogram('d3',d3)
        return [d1,d2,d3]
#     LOGITS ARE OF THE SHAPE 3*64*11
    logits=model(train_data,0.5)    
    loss_arr = [tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=train_labels[:,i],
                        logits=logits[i],   
                    )) 
                   for i in range(0,3)]

    loss=tf.add_n(loss_arr)
    tf.summary.scalar('loss',loss)
    learning_rate=tf.train.exponential_decay(0.001, global_step, 1000, 0.80, staircase=True)
    optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
#     return loss

    def softmax(data):
        with tf.name_scope('prediction_func'):
            logits=model(data,1.0)
            prediction = tf.stack([
                tf.nn.softmax(logits[0]),
                tf.nn.softmax(logits[1]),
                tf.nn.softmax(logits[2])
                ])
        return prediction
    
    # Predictions for the training, validation.
    with tf.name_scope('prediction_train'):
        train_prediction = softmax(train_data)
    with tf.name_scope('prediction_valid'):
        valid_prediction = softmax(valid_dataset)
    
    # Save Model 
    saver = tf.train.Saver()
    summaries=tf.summary.merge_all()

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    writer=tf.summary.FileWriter('/tmp/Mark2_logs',graph=graph)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord) 
    print('hi')
    for step in range(20000):
        _,loss1,batch_labels,predictions=sess.run([optimizer,loss,train_labels,train_prediction])
        if (step % 100 == 0):
            s=sess.run(summaries)
            writer.add_summary(s)
            print('Minibatch loss at step %d: %f' % (step, loss1))
            print('Minibatch image accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            predictions1,batch1_labels=sess.run([valid_prediction,label_batch1])
            print('Validation image accuracy: %.1f%%' % accuracy(predictions1, batch1_labels))
            if (step % 500 == 0):
                saver.save(sess,'/tmp/Mark2_logs/gsv.ckpt')

    coord.request_stop()
    coord.join(threads)
    sess.close()


