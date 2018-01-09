    import os
    os.chdir('marketbot')

    import  src.dataloader as dataloader
    get_data = dataloader.get_data
    path='data/data2.csv'; predict_length=10; interval_length=1; window_length=100
    features, outcomes = get_data(path, interval_length, predict_length)

    import src.playground1 as pg
    h = 10
    split=.5
    env = pg.Environment(features, h, split)  

    tr = env.reset()

    cur, loss, doneflag, info = env.step(0.0)
    cur, loss, doneflag, info

    (array([  3.00000000e-02,   1.31317100e+04,   2.22044605e-16]), 0.0, False, {})

    from importlib import reload
    reload(pg)

