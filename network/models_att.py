import tensorflow as tf
import scipy.sparse
import numpy as np
import os, time, collections, shutil, sys
ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(ROOT_PATH)

import math

class base_model(object):
    
    def __init__(self):
        self.regularizers = []
        self.checkpoints = 'final'
    
    # High-level interface which runs the constructed computational graph.
    
    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty((size, self.out_joints*3))
        close_sess_flag = True if sess is None else False
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            # If the last batch is smaller than a usual batch, fill with zeros.
            end = begin + self.batch_size
            end = min([end, size])
            
            batch_data = np.zeros((self.batch_size,)+data.shape[1:])
            tmp_data = data[begin:end]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 0, self.ph_istraining: False}
            
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros((self.batch_size,)+labels.shape[1:])
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin]

        if close_sess_flag:
            sess.close()

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions
        
    def evaluate(self, data, labels, sess=None):
        """
        """
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)

        string = 'loss: {:.4e}'.format(loss)

        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        return string, loss

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(graph=self.graph, config=config)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.compat.v1.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'),'final', 'model')
        best_path = os.path.join(self._get_path('checkpoints'),'best', 'model')
        sess.run(self.op_init)

        # Training.
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        epoch_steps = int(train_data.shape[0] / self.batch_size)
        min_loss = 10000
        for step in range(1, num_steps+1):

            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx, ...], train_labels[idx, ...]
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout,
                         self.ph_istraining: True}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.4e}'.format(learning_rate, loss_average))

                string, loss = self.evaluate(val_data, val_labels, sess)
                losses.append(loss)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))

                # Summaries for TensorBoard.
                summary = tf.compat.v1.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)
                if loss < min_loss:
                    min_loss = loss
                    self.op_best_saver.save(sess, best_path, global_step=step)

        print('validation loss: trough = {:.4f}, mean = {:.2f}'.format(min_loss, np.mean(losses[-10:])))
        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return losses, t_step
    
    def build_graph(self, M_0, in_F):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Mask.
            self.initialize_mask()

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.compat.v1.placeholder(tf.float32, (self.batch_size, M_0*in_F), 'data')
                self.ph_labels = tf.compat.v1.placeholder(tf.float32, (self.batch_size, M_0*3), 'labels')
                self.ph_dropout = tf.compat.v1.placeholder(tf.float32, (), 'dropout')
                self.ph_istraining = tf.compat.v1.placeholder(tf.bool, (), 'istraining')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_type, self.decay_params)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.compat.v1.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.compat.v1.summary.merge_all()
            self.op_saver = tf.compat.v1.train.Saver(max_to_keep=1)
            self.op_best_saver = tf.compat.v1.train.Saver(max_to_keep=1)
        
        self.graph.finalize()

    def initialize_mask(self):
        self._initialize_mask()
    
    def inference(self, data, dropout):
        logits = self._inference_lcn(data, data_dropout=dropout)
        return logits
    
    def probabilities(self, logits):
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        with tf.name_scope('prediction'):
            prediction = tf.compat.v1.identity(logits)
            return prediction

    def loss(self, logits, labels):
        with tf.name_scope('loss'):
            loss = 0
            with tf.name_scope('mse_loss'):
                mse_loss = tf.reduce_mean(tf.square(logits - labels))
                # logits = tf.reshape(logits, [-1, self.out_joints, 3])
                # labels = tf.reshape(labels, [-1, self.out_joints, 3])
                # mse_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(logits - labels), axis=2)))
            loss = loss + mse_loss

            if self.regularization != 0:
                with tf.name_scope('reg_loss'):
                    reg_loss = self.regularization * tf.add_n(self.regularizers)
                loss += reg_loss

            # Summaries for TensorBoard.
            tf.compat.v1.summary.scalar('loss/mse_loss', mse_loss)
            tf.compat.v1.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.compat.v1.train.ExponentialMovingAverage(0.9)
                loss_dict = {'mse': mse_loss, 'total': loss}
                op_averages = averages.apply(list(loss_dict.values()))
                for k, v in loss_dict.items():
                    tf.compat.v1.summary.scalar('loss/avg/%s' % k, averages.average(v))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.compat.v1.identity(averages.average(loss), name='control')
            return loss, loss_average

    def training(self, loss, learning_rate, decay_type, decay_params):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_type == 'exp':
                learning_rate = tf.compat.v1.train.exponential_decay(
                        learning_rate, global_step, decay_params['decay_steps'], decay_params['decay_rate'], staircase=False)
            elif decay_type == 'step':
                learning_rate = tf.compat.v1.train.piecewise_constant(global_step, decay_params['boundaries'], decay_params['lr_values'])
            else:
                assert 0, 'not implemented lr decay types!'
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads = optimizer.compute_gradients(loss)
                op_gradients = optimizer.apply_gradients(grads, global_step=global_step)

            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.compat.v1.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.compat.v1.identity(learning_rate, name='control')
            return op_train

    # Helper methods.
    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', 'experiment', self.dir_name, folder)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(graph=self.graph, config=config)
            filename = tf.compat.v1.train.latest_checkpoint(os.path.join(self._get_path('checkpoints'), self.checkpoints))
            print('restore from %s' % filename)
            self.op_best_saver.restore(sess, filename)
        return sess

    def _variable(self, name, initializer, shape, regularization=True):
        var = tf.compat.v1.get_variable(name, shape, tf.float32, initializer=initializer, trainable=True)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.compat.v1.summary.histogram(var.op.name, var)
        return var

class cgcnn(base_model):
    """
    """
    def __init__(self, F=64, mask_type='locally_connected', init_type='ones', neighbour_matrix=None, in_joints=17, out_joints=17, in_F=2,
                num_layers=2, residual=True, batch_norm=True, max_norm=True,
                num_epochs=200, learning_rate=0.001, decay_type='exp', decay_params=None,
                regularization=0.0, dropout=0, batch_size=200, eval_frequency=200, dir_name='', checkpoints='final'):
        super().__init__()
        
        self.F = F
        self.mask_type = mask_type
        self.init_type = init_type
        assert neighbour_matrix.shape[0] == neighbour_matrix.shape[1]
        assert neighbour_matrix.shape[0] == in_joints
        self.neighbour_matrix = neighbour_matrix

        self.in_joints = in_joints
        self.out_joints = out_joints
        self.num_layers = num_layers
        self.residual, self.batch_norm, self.max_norm = residual, batch_norm, max_norm
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_type, self.decay_params = decay_type, decay_params
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.checkpoints = checkpoints
        self.activation = tf.nn.leaky_relu
        self.in_F = in_F
        
        # Build the computational graph.
        self.build_graph(in_joints, self.in_F)

    def _initialize_mask(self):
        """
        Parameter
            mask_type
                locally_connected
                locally_connected_learnable
            init_type
                same: use L to init learnable part in mask
                ones: use 1 to init learnable part in mask
                random: use random to init learnable part in mask
        """
        if 'locally_connected' in self.mask_type:
            assert self.neighbour_matrix is not None
            L = self.neighbour_matrix.T
            assert L.shape == (self.in_joints, self.in_joints)
            if 'learnable' not in self.mask_type:
                self.mask = tf.constant(L)
            else:
                if self.init_type == 'same':
                    initializer = L
                elif self.init_type == 'ones':
                    initializer = tf.initializers.ones
                elif self.init_type == 'random':
                    initializer = tf.initializers.random_uniform_initializer(0, 1)
                var_mask = tf.get_variable(name='mask', shape=[self.in_joints, self.out_joints] if self.init_type != 'same' else None,
                    dtype=tf.float32, initializer=initializer)
                var_mask = tf.nn.softmax(var_mask, axis=0)
                # self.mask = var_mask
                self.mask = var_mask * tf.constant(L != 0, dtype=tf.float32)

    def mask_weights(self, weights):
        input_size, output_size = weights.get_shape()
        input_size, output_size = int(input_size), int(output_size)
        assert input_size % self.in_joints == 0 and output_size % self.in_joints == 0
        in_F = int(input_size / self.in_joints)
        out_F = int(output_size / self.in_joints)
        weights = tf.reshape(weights, [self.in_joints, in_F, self.in_joints, out_F])
        mask = tf.reshape(self.mask, [self.in_joints, 1, self.in_joints, 1])
        masked_weights = weights * mask
        masked_weights = tf.reshape(masked_weights, [input_size, output_size])
        return masked_weights

    def batch_normalization_warp(self, y, training, name):
        keras_bn = tf.keras.layers.BatchNormalization(axis=-1, name=name)

        _, output_size = y.get_shape()
        output_size = int(output_size)
        out_F = int(output_size / self.in_joints)
        y = tf.reshape(y, [-1, self.in_joints, out_F])
        y = keras_bn(y, training=training)
        y = tf.reshape(y, [-1, output_size])

        for item in keras_bn.updates:
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, item)

        return y


    def kaiming(self, shape, dtype, partition_info=None):
        """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf

        Args
            shape: dimensions of the tf array to initialize
            dtype: data type of the array
            partition_info: (Optional) info about how the variable is partitioned.
                See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
                Needed to be used as an initializer.
        Returns
            Tensorflow array with initial weights
        """
        return(tf.random.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

    def two_linear(self, xin, data_dropout, idx):
        """
        Make a bi-linear block with optional residual connection

        Args
            xin: the batch that enters the block
            idx: integer. Number of layer (for naming/scoping)
            Returns
        y: the batch after it leaves the block
        """

        with tf.compat.v1.variable_scope( "two_linear_"+str(idx) ) as scope:

            
            output_size = self.in_joints * self.F

            # Linear 1
            input_size2 = int(xin.get_shape()[1])
            w2 = self._variable("w2_"+str(idx), self.kaiming, [input_size2, output_size], regularization=self.regularization!=0)
            b2 = self._variable("b2_"+str(idx), self.kaiming, [output_size], regularization=self.regularization!=0)
            w2 = tf.clip_by_norm(w2,1) if self.max_norm else w2
            w2 = self.mask_weights(w2)
            y = tf.matmul(xin, w2) + b2

            if self.batch_norm:
                y = self.batch_normalization_warp(y, training=self.ph_istraining, name="batch_normalization1"+str(idx))
            y = self.activation(y)
            y = tf.nn.dropout(y, rate=data_dropout)
            # ====================

            # Linear 2
            input_size3 = int(y.get_shape()[1])
            w3 = self._variable("w3_"+str(idx), self.kaiming, [input_size3, output_size], regularization=self.regularization!=0)
            b3 = self._variable("b3_"+str(idx), self.kaiming, [output_size], regularization=self.regularization!=0)
            w3 = tf.clip_by_norm(w3,1) if self.max_norm else w3
            w3 = self.mask_weights(w3)
            y = tf.matmul(y, w3) + b3
            if self.batch_norm:
                y = self.batch_normalization_warp(y, training=self.ph_istraining, name="batch_normalization2"+str(idx))
            y = self.activation(y)
            y = tf.nn.dropout(y, rate=data_dropout)
            # ====================

            # Residual every 2 blocks
            y = (xin + y) if self.residual else y
        return y

    def _inference_lcn(self, x, data_dropout):

        with tf.compat.v1.variable_scope('linear_model'):

            mid_size = self.in_joints * self.F

            # === First layer===
            w1 = self._variable("w1", self.kaiming, [self.in_joints * self.in_F, mid_size], regularization=self.regularization!=0)
            b1 = self._variable("b1", self.kaiming, [mid_size], regularization=self.regularization!=0)  # equal to b2leaky_relu
            w1 = tf.clip_by_norm(w1,1) if self.max_norm else w1
            
            w1 = self.mask_weights(w1)
            y3 = tf.matmul(x, w1) + b1

            if self.batch_norm:
                y3 = self.batch_normalization_warp(y3, training=self.ph_istraining, name="batch_normalization")
            y3 = self.activation(y3)
            y3 = tf.nn.dropout(y3, rate=data_dropout)

            # === Create multiple bi-linear layers ===
            for idx in range(self.num_layers):
                y3 = self.two_linear(y3, data_dropout=data_dropout, idx=idx)

            # === Last layer ===
            input_size4 = int(y3.get_shape()[1])
            w4 = self._variable("w4", self.kaiming, [input_size4, self.out_joints*3], regularization=self.regularization!=0)
            b4 = self._variable("b4", self.kaiming, [self.out_joints*3], regularization=self.regularization!=0)
            w4 = tf.clip_by_norm(w4,1) if self.max_norm else w4
            
            w4 = self.mask_weights(w4)
            y = tf.matmul(y3, w4) + b4
            # === End linear model ===

            x = tf.reshape(x, [-1, self.in_joints, self.in_F])  # [N, J, 3]
            y = tf.reshape(y, [-1, self.out_joints, 3])  # [N, J, 3]
            y = tf.concat([x[:, :, :2] + y[:, :, :2], tf.expand_dims(y[:, :, 2], axis=-1)], axis=2)  # [N, J, 3]
            y = tf.reshape(y, [-1, self.out_joints*3])

        return y