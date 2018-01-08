
# setup

    import os
    os.chdir('marketbot')

Create input provider

    import  src.dataloader as dataloader

    get_data = dataloader.get_data
    WindowGen = dataloader.WindowGen
    quantize = dataloader.quantize
    path='data/data2.csv'; predict_length=10; interval_length=1; window_length=100
    features, outcomes = get_data(path, interval_length, predict_length)

    import numpy as np
    amin=-0.01
    amax=0.01
    step=1e-5
    Y_n_categories = int(np.round((amax-amin)/step))

    q_outcomes = quantize(outcomes, amin=amin, amax=amax, step=step)
    q_outcomes = np.expand_dims(q_outcomes, axis=1)
    gen = WindowGen(features, q_outcomes, window_length, predict_length, Y_n_categories)

    from importlib import reload
    reload(dataloader)

    import marketbot.src.model1.runner as runner

    from collections import namedtuple
    flagdct = {'batch_size': 128,
    	   'data_dir': '/tmp/dat/',
    	   'hidden_dim': 200,
    	   'l1reg_coeff': 1e-10,
    	   'l2reg_coeff': 1e-9,
    	   # 'l1reg_coeff': 1,
    	   # 'l2reg_coeff': 1,
    	   'latent_dim': 160,
    	   'logdir': '/tmp/log/',
    	   'n_epochs': 100000,
    	   'n_iterations': 100000,
    	   'n_samples_predict': 20,
    	   'n_samples_train': 10,
    	   'print_every': 1000, 
    	   'huber_loss_delta': .1,
    	   'use_update_ops': False}  # update_ops control dependency is necessary for batch norm
    FLAGS = namedtuple('FLAGS',flagdct.keys())(**flagdct)
    ff_params = dict(dim_hidden=20, rnn_stack_height=3)
    
    catmodel = runner.Learner(ff_params, FLAGS)

    catmodel.initialize_train_graph(gen)


# train

    catmodel.train(100)

    from importlib import reload
    reload(runner)


# predict

