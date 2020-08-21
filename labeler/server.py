import time
import zmq
import os

from predict import predict_audacity_labels


#TODO: figure out a way to start and kill the server from within Audacity. 
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5555")


try:
    while True:
# wait for next request from client
        print(f'waiting for request...')
        message = socket.recv()
        print(f"received request {message}")

        paths_to_audio = message.decode('utf-8')

        base = os.path.basename(paths_to_audio)
        root = os.path.dirname(paths_to_audio)
        audio_name = os.path.splitext(base)[0]

        paths_to_output = os.path.join(root, audio_name + '-labels' '.txt')

        print(f'our path to output is {paths_to_output}')

        predict_audacity_labels(paths_to_audio, paths_to_output)

        # send reply
        socket.send_string(paths_to_output)
except Exception as e:
    print(f'exception occured: {e}')
    pass
finally:
    context.term()