import time
import zmq
import os
import traceback

from predict import write_audacity_labels


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

        audio_path = message.decode('utf-8')

        base = os.path.basename(audio_path)
        root = os.path.dirname(audio_path)
        audio_name = os.path.splitext(base)[0]

        label_path = os.path.join(root, audio_name + '-labels' '.txt')

        print(f'our label output path is {label_path}')

        write_audacity_labels(audio_path, label_path)

        # send reply
        socket.send_string(label_path)
except Exception as e:
    socket.send_string("/Users/hugoffg/Documents/lab/philharmonia-dataset/data/philharmonia/all-samples/cello/cello_Gs5_05_pianissimo_arco-normal-labels.txt")
    print(f'exception occured: {e}')
    print(traceback.format_exc())
    pass
finally:
    context.term()