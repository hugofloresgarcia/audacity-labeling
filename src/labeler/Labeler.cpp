
//
//  Labeler.cpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//  Modified by Hugo Flores on 7/31/20.

#include <zmq.hpp>

// lol I need to actually go through these and see which ones I actually use
#include "../Audacity.h" // for USE_* macros
#include "../Experimental.h"

#include "../BatchCommands.h"
#include "../Clipboard.h"
#include "../CommonCommandFlags.h"
#include "../FileNames.h"
#include "../LabelTrack.h"
#include "../NoteTrack.h"
#include "../Prefs.h"
#include "../Printing.h"
#include "../Project.h"
#include "../ProjectFileIO.h"
#include "../ProjectFileManager.h"
#include "../ProjectHistory.h"
#include "../ProjectManager.h"
#include "../ProjectWindow.h"
#include "../SelectUtilities.h"
#include "../TrackPanel.h"
#include "../UndoManager.h"
#include "../ViewInfo.h"
#include "../WaveTrack.h"
#include "../commands/CommandContext.h"
#include "../commands/CommandManager.h"
#include "../export/ExportMultiple.h"
#include "../import/Import.h"
#include "../import/ImportMIDI.h"
#include "../import/ImportRaw.h"
#include "../widgets/AudacityMessageBox.h"
#include "../widgets/FileHistory.h"

#include <iostream>
#include <cmath>

#include <wx/textfile.h>

#include "Labeler.hpp"
#include "WaveTrack.h"
#include "../FileNames.h"
#include "../commands/CommandContext.h"
#include "../LabelTrack.h"
#include "../ProjectHistory.h"

using namespace std;

void IALLabeler::LabelTrack(const CommandContext &context, wxArrayString selectedFiles){
   for (size_t ff = 0; ff < selectedFiles.size(); ff++) {
      wxString fileName = selectedFiles[ff];
      LabelTrack(context, fileName.ToStdString());
   }
}
void IALLabeler::LabelTrack(const CommandContext &context, const std::string &filepath) {
    auto &project = context.project;
    auto &trackFactory = TrackFactory::Get( project );
    auto &tracks = TrackList::Get( project );

    const string endpoint = "tcp://127.0.0.1:5555";

    // initiallize the zeromq context
    zmq::context_t ctx;
    zmq::socket_t socket(ctx, zmq::socket_type::req);

    //open the connection
    cout<<"connecting to labeler server"<<endl;
    //bind to the socket
    socket.connect(endpoint);
    cout<<"connected to labaler server!";

    //send a request to label the track
    cout<<"sending request to labeler";
    zmq::message_t message(filepath);

    socket.send(message);

    zmq::message_t reply;
    socket.recv (&reply);

    string pathToLabelFile = string(static_cast<char*>(reply.data()), reply.size());
    wxString fileName(pathToLabelFile);

    if (!fileName.empty()) {
      wxTextFile f;

      f.Open(fileName);
      if (!f.IsOpened()) {
         AudacityMessageBox(
            XO("Could not open file: %s").Format( fileName ) );
         return;
      }

      auto newTrack = trackFactory.NewLabelTrack();
      wxString sTrackName;
      wxFileName::SplitPath(fileName, NULL, NULL, &sTrackName, NULL);
      newTrack->SetName(sTrackName);

      newTrack->Import(f);

      SelectUtilities::SelectNone( project );
      newTrack->SetSelected(true);
      tracks.Add( newTrack );

      ProjectHistory::Get( project ).PushState(
         XO("Imported labels from '%s'").Format( fileName ),
            XO("Import Labels"));
   }
    
}   