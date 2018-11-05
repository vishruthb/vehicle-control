import asyncio
import json
import logging
import numpy as np
import time
import websockets


logging.basicConfig()

port = 4567

async def on_message(ws, path):
    data = await ws.recv()
    data_in = json.loads(data[2:])
    if data_in[0] != 'telemetry':
        return

    msg_json = {}
    msg_json['steering_angle'] = 0;
    msg_json['throttle'] = .5;
    msg = "42[\"steer\",%s]" % json.dumps(msg_json)
    print(msg)
    
    await ws.send(msg)
    time.sleep(0.01)

start_server = websockets.serve(on_message, 'localhost', port)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
