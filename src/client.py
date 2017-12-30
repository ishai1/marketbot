import websockets
import asyncio
import sys
import json
from pandas import DataFrame, Series

WS_PATHS = {
    'sandbox': 'wss://ws-feed-public.sandbox.gdax.com',
    'live': 'wss://ws-feed.gdax.com'
}

SUBSCRIBE_REQUEST = {
    'type': 'subscribe',
    'product_ids': ['BTC-USD'],
    'channels': [
        'matches',
        'heartbeat'
    ] 
}

DATAFRAME = DataFrame(columns=['price', 'size', 'side', 'time'])
NUM_MATCHES = 0

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
        else:
            handle_message(json.loads(msg))
    return
    
def match_counter():
    global NUM_MATCHES
    NUM_MATCHES += 1
    if NUM_MATCHES % 10 == 0:
        sys.stdout.write("\r matches entered: {}".format(NUM_MATCHES))
        sys.stdout.flush()

def handle_message(msg):
    if msg['type'] == 'match':
        match_counter()
        DATAFRAME.loc[msg['sequence']] = Series(msg)

if __name__ == "__main__":
    try:
        path = WS_PATHS[sys.argv[1]]
    except Exception:
        print("Require argument <sandbox> or <option>")
        exit()
    else:
        try:
            asyncio.get_event_loop().run_until_complete(_connect(path))
        except KeyboardInterrupt:
            pass
            
