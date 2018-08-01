#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for collecting samples.
Example usage:
    python get_samples.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=50,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        damages=[[-1,1], [-.7,.7], [-.5,.5]]
        bigdata = np.empty([0, max_steps, 26])
        y_data = np.empty([0,2])
        clas = np.empty([0,1])
        for it, j in enumerate(damages):
            print ('damage',it)
            env.env.model.actuator_ctrlrange = np.array([[j, damages[0], damages[0]]])

            for i in range(args.num_rollouts):
                returns = []
                observations = []
                actions = []
                newobs = []
                rewards = []
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = policy_fn(obs[None,:])
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    newobs.append(obs)
                    rewards.append(r)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

                # print('returns', returns)
                # print('mean return', np.mean(returns))
                # print('std of return', np.std(returns))

                observations = np.array(observations)
                actions = np.array(actions)
                actions = actions.reshape(np.shape(actions)[0], np.shape(actions)[2])
                newobs = np.array(newobs)
                rewards = np.array(rewards)
                rewards = rewards.reshape(np.shape(rewards)[0], 1)




                # print (observations.shape)
                # print(actions.shape)
                # print(newobs.shape)
                # print(rewards.shape)

                data = np.concatenate((observations, actions, newobs, rewards), axis=1)
                data = np.around(data, 6)
                data = data.reshape(1,max_steps, 26)
                #print(len(j))
                y_data = np.append(y_data, [j], axis=0)
                #print(y_data)
                #print(bigdata.shape, data.shape)
                bigdata = np.append(bigdata, data, axis=0)

                it=np.reshape(np.array(it), (1,1))
                clas = np.append(clas, it, axis=0)

        print(bigdata.shape)
        print(y_data.shape)
        print(clas.shape)
        #print (data[100])

        #
        #
        train_data = {'bigdata': bigdata,
                        'y_data': y_data,
                        "class": clas}
        pickle_out = open("data_pickles/" + args.envname + "_traintest.dict", 'wb')
        pickle.dump(train_data, pickle_out)
        pickle_out.close()

if __name__ == '__main__':
    main()
