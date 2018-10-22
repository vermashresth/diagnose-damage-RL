## One Shot Damage Diagnosis

This is the code for a portion of my B.Tech Thesis Project `Deep Reinforcement Learning for Stability and Safe Adaptation in Damaged Robots`. It can diagnoze damage in any locomatory OpenAI gym agent using only only rollout of motion.

### Requirements
1. Keras
2. Tensorflow
3. OpenAI gym and Mujoco (See installation instructions [here](https://github.com/openai/gym]))
4. Joblib


### How to use
First collect samples of damaged robot data using 
```bash
python sampler.py experts/Ant-v1.pkl Ant-v1 --num_rollouts 2000
```
(Note that this step is paraellized over multiple threads. I have written this code for 4 threads, it can easily be scaled up for clusters having large number of available threads.)

Then load the pickled data and train the LSTM network
```bash
python rnn_train.py data_pickles/Ant-v1_4joints20diff102type1.dict -s saved_models/myclean.h5 -e 50
```
Generate some test data again using sampler and run testing network
```bash
python rnn_test.py -m saved_models/my_modelant4jointsday32_eff_div_2type.h5 -d data_pickles/Ant-v1_4joints20diff1002type1.dict
```
This work is still in progress. Feel free to contact me if you are interested in this kind of architecture or want to discuss any ideas.
