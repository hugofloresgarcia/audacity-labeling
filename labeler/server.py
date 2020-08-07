import time
import zmq
import os

from predict import predict_audacity_labels


#TODO close the port properly
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5555")

while True:
    try:
        # wait for next request from client
        print(f'waiting for request...')
        message = socket.recv()
        print(f"received request {message}")

        paths_to_audio = message.decode('utf-8')

        base = os.path.basename(paths_to_audio)
        root = os.path.dirname(paths_to_audio)
        audio_name = os.path.splitext(base)[0]

        paths_to_output = os.path.join(root, audio_name + '.txt')

        print(f'our path to output is {paths_to_output}')

        predict_audacity_labels(paths_to_audio, paths_to_output)

        # send reply
        socket.send_string(paths_to_output)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.close()