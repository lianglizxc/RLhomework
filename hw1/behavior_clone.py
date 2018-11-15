import pickle
import tensorflow as tf
import numpy as np
import gym
import load_policy
from tensorflow.contrib.layers import fully_connected


def train_bs():
    with open('expert_Data/Humanoid-v2.pkl', 'rb') as file:
        data = pickle.load(file)

    actions = data['actions']
    observations = data['observations']

    actions_shape = list(actions.shape)
    observations_shape = list(observations.shape)

    actions_shape[0] = None
    observations_shape[0] = None

    input = tf.placeholder(dtype=tf.float32,shape=observations_shape)
    label = tf.placeholder(dtype=tf.float32,shape=actions_shape)

    hidden1 = fully_connected(input, num_outputs=256, activation_fn=tf.nn.tanh)
    hidden2 = fully_connected(hidden1, num_outputs=128, activation_fn=tf.nn.tanh)
    hidden3 = fully_connected(hidden2, num_outputs=64, activation_fn=tf.nn.tanh)
    output = fully_connected(hidden3, num_outputs=actions_shape[-1], activation_fn=None)
    output = tf.expand_dims(output, axis=1)

    loss = tf.losses.mean_squared_error(labels=label, predictions=output)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    def get_batch(index, batch_size = 1024):

          np.random.shuffle(index)
          batchs= []
          for i in range(len(index)):
              batchs.append(i)
              if len(batchs) == batch_size:
                  yield batchs
                  batchs = []

          if len(batchs) != 0:
              yield batchs

    epoch = 60
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        index = np.arange(len(actions))
        for i in range(epoch):
            batch_loss =[]
            for batch_index in get_batch(index):
                batch_x, batch_y = observations[batch_index], actions[batch_index]
                _, loss_val = sess.run([optimizer, loss], feed_dict={input: batch_x, label:batch_y})
                batch_loss.append(loss_val)
            print('batch_loss is {}'.format(np.average(batch_loss)))

        policy = lambda x: sess.run(output, feed_dict={input: x})
        get_performence(policy)

def get_performence(policy):

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations= []
    actions= []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr, total_bs = 0., 0.
        steps = 0
        while not done:
            action = policy(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    train_bs()







