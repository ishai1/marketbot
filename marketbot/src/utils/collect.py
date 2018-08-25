"""
    Stream data from gdax to file
"""
import logging
import requests
import json
import asyncio
import sys
import time
import csv
import websockets
import os
logger = logging.getLogger('websockets')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Gather product ids
HTTPURI = 'https://api-public.sandbox.pro.coinbase.com/products'
ret = requests.get(HTTPURI)
products = json.loads(ret.text)
logger.info("{} available products".format(len(products)))
logger.info("Product pairs:\n {}".format([p['id'] for p in products]))

WSURI = 'wss://ws-feed.pro.coinbase.com'
SUBSCRIBE_REQUEST = {
    'type': 'subscribe',
    'product_ids': [p['id'] for p in products],
    'channels': [
        'matches',
        'heartbeat']
}

COLUMNS = ['time', 'price', 'size', 'product_id']
NUM_MATCHES = 0
FILEPATH = None


async def _connect(path):
    async with websockets.connect(path) as ws:
        print('connecting')
        await ws.send(json.dumps(SUBSCRIBE_REQUEST))
        print('Connected')
        message = await ws.recv()
        print(' {} '.format(message))
        await _listen(ws)


async def _listen(ws):
    while True:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=20)
        except asyncio.TimeoutError:
            # No data in 20 seconds, check the connection.
            try:
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=10)
            except asyncio.TimeoutError:
                # No response to ping in 10 seconds, disconnect.
                break
        except Exception as e:
            # No response to ping in 10 seconds, disconnect.
            # logger.exception("General listener exception")
            break
        else:
            handle_message(json.loads(msg))


def match_counter():
    global NUM_MATCHES
    NUM_MATCHES += 1
    if NUM_MATCHES % 10 == 0:
        sys.stdout.write("\r matches entered: {}".format(NUM_MATCHES))
        sys.stdout.flush()


def handle_message(msg):
    if msg['type'] == 'match':
        match_counter()
        with open(FILEPATH, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([msg[key] for key in COLUMNS])


if __name__ == "__main__":
    time = time.strftime("%Y%m%d-%H%M%S")
    FILEPATH = os.path.join(os.getcwd(), 'data/raw/{}.csv'.format(time))
    loop = asyncio.get_event_loop()
    try:
        while True:
            loop.run_until_complete(_connect(WSURI))
            # asyncio.ensure_future(_connect(WSURI), loop=loop)
        
    finally:
        loop.close()
    # try:
    #     loop.run_forever()
    # finally:
    #     loop.close()
