
//
//  Labeler.cpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//  Modified by Hugo Flores on 7/31/20.

#include <zmq.hpp>

#include <iostream>
#include <cmath>

#include <wx/textfile.h>

#include "Labeler.hpp"
#include "WaveTrack.h"
#include "../FileNames.h"
#include "../commands/CommandContext.h"
#include "../LabelTrack.h"
#include "../ProjectHistory.h"

void IALLabeler::LabelTrack(const CommandContext &context, const std::string &filepath) {
    auto &project = context.project;
    auto &trackFactory = TrackFactory::Get( project );
    auto &tracks = TrackList::Get( project );

    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:5555");

    while (true) {
        zme::message_t re
    }

    // // set our python environment to our virtual env
    // // Py_SetPythonHome("/Users/hugoffg/Documents/lab/audacity-labeling/labeler/venv-labeler/");
    // // Py_SetProgramName("/Users/hugoffg/Documents/lab/audacity-labeling/labeler/venv-labeler.py");

    // // wchar_t* a = Py_GetPath();
    // Py_Initialize();

    // // PySys_SetArgvEx("/Users/hugoffg/Documents/lab/audacity-labeling/labeler/predict.py -p /Users/hugoffg/Music/songs/harvest-moon.mp3 -o /Users/hugoffg/Documents/lab/audacity-labeling/labeler/output/harvest-moon.txt")
    
    // // PyObject* main_module = PyImport_AddModule("__main__");
    // // PyObject* main_dict = PyModule_GetDict(main_module);


    // // std::string PathToPredict = "/Users/hugoffg/Documents/lab/audacity-labeling/labeler/predict.py";
    // // FILE* predictor = fopen(PathToPredict.c_str(), "r");

    // // PyRun_SimpleFileEx(predictor, PathToPredict.c_str(), 1);

    Py_Finalize();
}