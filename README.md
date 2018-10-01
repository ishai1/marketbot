# MarketBot 2.0

This project tries to forecast short term bitcoin price fluctuations using neural networks.

It's primary objective was to provide an opprtunity to try out the new release of [Julia](https://github.com/JuliaLang/julia/) (0.7/1.0) as well as try out sequence forecasting using attention.

Using [python websockets](./src/utils/collect.py) to extract data from coinbase, it uses julia to [process and upsample](./Clean%20Up.ipynb) the collected data and then calls an [estimator based tensorflow model](./src/pytf) to train on the data.