import numpy as np
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DeepQNetwork():
    def __init__(self,
                 n_actions,
                 n_features,
                 width=64,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.8,
                 replace_target_iter=500,
                 memory_size=5000,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False, ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.width = width
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = 1.0
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy
        self.memory_counter = 0

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [
                tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        tf.reset_default_graph()
        # tf.compat.v1.reset_default_graph()
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(
            tf.float32, [None, self.n_features], name='s')  # input State
        # print('_build_net self.n_features :', self.n_features)
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(
            0., 0.1), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            # tf.layers.dense
            e1 = tf.layers.dense(self.s, self.width * 2, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, self.width, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            # e3 = tf.layers.dense(e2, self.width / 2, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e3')
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, self.width * 2, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, self.width, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            # t3 = tf.layers.dense(t2, self.width / 2, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='t3')
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * \
                tf.reduce_max(self.q_next, axis=1,
                              name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(
                params=self.q_eval, indices=a_indices)  # shape=(None, )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(
                self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, force_random=False):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        random_val = np.random.uniform()
        # print(self.epsilon, random_val)
        if random_val < self.epsilon and not force_random:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment 
        self.learn_step_counter += 1

