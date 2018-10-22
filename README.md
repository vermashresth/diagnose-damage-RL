python rnn_train.py data_pickles/Ant-v1_4joints20diff102type1.dict -s saved_models/myclean.h5 -e 5
python eff_div_par_sampler experts/Ant-v1.pkl Ant-v1 --num_rollouts 2000
