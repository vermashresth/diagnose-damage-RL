#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for collecting samples.
Example usage:
    python sampler.py experts/Ant-v1.pkl Ant-v1 --render \
            --num_rollouts 20
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

from joblib import Parallel, delayed
import itertools
import gym, itertools
from sklearn.preprocessing import normalize
def batch_env(workers, args):
    envs=[]
    for i in range(workers):
            env = gym.make('MyAnt-v1')
            env_h = gym.make('Ant-v1')
            envs.append([env,env_h])
    return envs
def main():



    a=range(args.num_rollouts)
    # dampairs = list(itertools.product(damages,damages,damages,damages,damages,damages,damages,damages))
    #dampairs = list(itertools.product(damages,damages,damages,damages))
    dampairs = batch_env(4, args)
    # print dampairs
    ar = [args]

    pol=[1]
    paramlist = list(itertools.product(a,dampairs, ar, pol))
    ntrials = args.num_rollouts * (len(dampairs))*len(pol)

    out = Parallel(n_jobs=12, verbose=1, backend="multiprocessing")(
                 map(delayed(sampling),paramlist))

    #bigdata1, bigdata2, bigdata3, y_data, clas = sampling(args)
    # print(len(out))
    # print(len(out[:][1]))



    bigdata1=np.array([row[0] for row in out]).reshape((ntrials,args.max_timesteps,n_obs))
    #bigdata2=np.array([row[1] for row in out]).reshape((ntrials,args.max_timesteps,n_obs*2+n_act))
    #bigdata3=np.array([row[2] for row in out]).reshape((ntrials,args.max_timesteps,n_obs+n_act))

    y_data=np.array([row[1] for row in out]).reshape((ntrials, 1))
    clas=np.array([row[2] for row in out]).reshape((ntrials,1))
    print(bigdata1.shape)
    # print(bigdata.shape)
    # print(y_data.shape)
    # print(clas.shape)
    train_data = {'bigdata1': bigdata1,
                    'y_data': y_data,
                    "class": clas}

    pickle_out = open("data_pickles/" + args.envname + "_4joints"+str(args.max_timesteps)+"diff"+str(args.num_rollouts)+"2type"+str(len(pol))+".dict", 'wb')
    pickle.dump(train_data, pickle_out)
    pickle_out.close()


def sampling(arguments):

    it, e, args, pol = arguments
    print it
    env, env_h = e


    max_steps = args.max_timesteps or env.spec.timestep_limit


    with tf.Session():
        tf_util.initialize()


        #itr = dampairs.index(dampair)

        bigdata1 = np.empty([0, max_steps, n_obs])
        bigdata2 = np.empty([0, max_steps, n_obs*2+n_act])
        bigdata3 = np.empty([0, max_steps, n_obs+n_act])

        y_data = np.empty([0,1])
        clas = np.empty([0,1])

        for it, j in enumerate(damages):
            # print ('damage',it)
            # print(dampair)
            # env.env.model.actuator_ctrlrange = np.array(dampair)
            # if dampair<0:
            #     ori = np.array(env.env.model.jnt_range)
            #     ori[dampair] = [-0.1, .1]
            #     env.env.model.jnt_range = ori
            # else:
            #     ori = np.array(env.env.model.actuator_ctrlrange)
            #     ori[dampair] = [-0.01, 0.01]
            #     env.env.model.actuator_ctrlrange = ori

            for i in range(args.num_rollouts):
                returns = []
                observations = []
                actions = []
                observations_h = []
                rewards = []
                # print('iter', i)

                ranseed = np.random.randint(100000)

                s=env.seed(ranseed)
                s=env_h.seed(ranseed)
                obs = env.reset()
                obs_h = env_h.reset()
                done = False
                totalr = 0.
                totalr_h = 0.
                steps = 0
                while not False:
                    action = policy_fn(obs[None,:])*pol
                    action_h = policy_fn(obs_h[None,:])*pol
                    observations.append(obs)

                    observations_h.append(obs_h)
                    actions.append(normalize(action))
                    obs, r, done, _ = env.step(action)

                    obs_h, r_h, done_h, _ = env_h.step(action)
                    rewards.append(r)
                    #print(rewards)
                    totalr += r
                    totalr_h += r_h
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)
                #print totalr, totalr_h
                # print('returns', returns)
                # print('mean return', np.mean(returns))
                # print('std of return', np.std(returns))

                observations = np.array(observations)
                #actions = np.array(actions)
                #actions = actions.reshape(np.shape(actions)[0], np.shape(actions)[2])
                observations_h = np.array(observations_h)
                #rewards = np.array(rewards)
                #rewards = normalize(rewards.reshape(np.shape(rewards)[0], 1))

                # print (observations.shape)
                # print(actions.shape)
                # print(newobs.shape)
                # print(rewards.shape)

                data1 = observations_h-observations
                #print data1.shape
                data1 = data1.reshape(1,max_steps, n_obs)
                #rint data1
                #data2 = np.concatenate((observations, actions, newobs), axis=1)
                #data2 = data2.reshape(1,max_steps, n_obs*2+n_act)

                #data3 = np.concatenate((observations - newobs, actions), axis=1)
                #data3 = data3.reshape(1,max_steps, n_obs+n_act)
                #print(len(j))
                itr = env.env.model.numeric_data[-1]
                dampair = itr
                #print itr
                #print y_data.shape, np.array([dampair]).shape
                y_data = np.append(y_data, np.reshape([dampair],(1,1)), axis=0)
                #print(y_data)
                #print(bigdata1.shape, data1.shape)
                bigdata1 = np.append(bigdata1, data1, axis=0)
                #bigdata2 = np.append(bigdata2, data2, axis=0)
                #bigdata3 = np.append(bigdata3, data3, axis=0)

                itr=np.reshape(np.array(itr), (1,1))
                clas = np.append(clas, itr, axis=0)
                return bigdata1, y_data, clas

#if __name__ == '__main__':

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('expert_policy_file', type=str)
parser.add_argument('envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=50,
                    help='Number of expert roll outs')
args = parser.parse_args()

e = gym.make(args.envname)
n_obs=e.observation_space.shape[0]
n_act=e.action_space.shape[0]


damages=[[-1,1], [-.5,.5]]
# dampairs = list(itertools.product(damages,damages,damages,damages,damages,damages,damages,damages))
dampairs = [-2,-4,-6,-8, 0, 2, 4, 6]
print('loading and building expert policy')
policy_fn = load_policy.load_policy(args.expert_policy_file)
print('loaded and built')

main()
