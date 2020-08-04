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

// Added for the TrackLabeler
#include "../labeler/Labeler.hpp"

#ifdef USE_MIDI
#include "../import/ImportMIDI.h"
#endif // USE_MIDI

#include <wx/menu.h>

// private helper classes and functions
namespace {
void DoExport( AudacityProject &project, const FileExtension & Format )
{
   auto &tracks = TrackList::Get( project );
   auto &projectFileIO = ProjectFileIO::Get( project );
   
   Exporter e{ project };

   double t0 = 0.0;
   double t1 = tracks.GetEndTime();

   // Prompt for file name and/or extension?
   bool bPromptingRequired =
      (project.mBatchMode == 0) || projectFileIO.GetFileName().empty() ||
      Format.empty();
   wxString filename;

   if (!bPromptingRequired) {

      // We're in batch mode, and we have an mFileName and Format.
      wxString extension = Format;
      extension.MakeLower();

      filename =
         MacroCommands::BuildCleanFileName(projectFileIO.GetFileName(), extension);

      // Bug 1854, No warning of file overwrite
      // (when export is called from Macros).
      int counter = 0;
      bPromptingRequired = wxFileExists(filename);

      // We'll try alternative names to avoid overwriting.
      while ( bPromptingRequired && counter < 100 ) {
         counter++;
         wxString number;
         number.Printf("%03i", counter);
         // So now the name has a number in it too.
         filename = MacroCommands::BuildCleanFileName(
            projectFileIO.GetFileName() + number, extension);
         bPromptingRequired = wxFileExists(filename);
      }
      // If we've run out of alternative names, we will fall back to prompting
      // - even if in a macro.
   }


   if (bPromptingRequired)
   {
      // Do export with prompting.
      e.SetDefaultFormat(Format);
      e.Process(false, t0, t1);
   }
   else
   {
      FileHistory::Global().Append(filename);
      // We're in batch mode, the file does not exist already.
      // We really can proceed without prompting.
      int nChannels = MacroCommands::IsMono( &project ) ? 1 : 2;
      e.Process(
         nChannels,  // numChannels,
         Format,     // type, 
         filename,   // filename,
         false,      // selectedOnly, 
         t0,         // t0
         t1          // t1
      );
   }

}
}

// Menu handler functions

namespace FileActions {

struct Handler : CommandHandlerObject {

void OnNew(const CommandContext & )
{
   ( void ) ProjectManager::New();
}

void OnOpen(const CommandContext &context )
{
   auto &project = context.project;
   ProjectManager::OpenFiles(&project);
}

// JKC: This is like OnClose, except it empties the project in place,
// rather than creating a new empty project (with new toolbars etc).
// It does not test for unsaved changes.
// It is not in the menus by default.  Its main purpose is/was for 
// developers checking functionality of ResetProjectToEmpty().
void OnProjectReset(const CommandContext &context)
{
   auto &project = context.project;
   ProjectManager::Get( project ).ResetProjectToEmpty();
}

void OnClose(const CommandContext &context )
{
   auto &project = context.project;
   auto &window = ProjectWindow::Get( project );
   ProjectFileManager::Get( project ).SetMenuClose(true);
   window.Close();
}

void OnCompact(const CommandContext &context)
{
   auto &project = context.project;
   auto &undoManager = UndoManager::Get(project);
   auto &clipboard = Clipboard::Get();
   auto &projectFileIO = ProjectFileIO::Get(project);

   // Purpose of this is to remove the -wal file.
   projectFileIO.ReopenProject();

   auto currentTracks = TrackList::Create( nullptr );
   auto &tracks = TrackList::Get( project );
   for (auto t : tracks.Any())
   {
      currentTracks->Add(t->Duplicate());
   }

   int64_t total = projectFileIO.GetTotalUsage();
   int64_t used = projectFileIO.GetCurrentUsage(currentTracks);

   auto before = wxFileName::GetSize(projectFileIO.GetFileName());

   int id = AudacityMessageBox(
      XO("Compacting this project will free up disk space by removing unused bytes within the file.\n\n"
         "There is %s of free disk space and this project is currently using %s.\n"
         "\n"
         "If you proceed, the current Undo History and clipboard contents will be discarded "
         "and you will recover approximately %s of disk space.\n"
         "\n"
         "Do you want to continue?")
      .Format(Internat::FormatSize(projectFileIO.GetFreeDiskSpace()),
              Internat::FormatSize(before.GetValue()),
              Internat::FormatSize(total - used)),
      XO("Compact Project"),
      wxYES_NO);

   if (id == wxYES)
   {
      // Want to do this before removing the states so that it becomes the
      // current state.
      ProjectHistory::Get(project)
         .PushState(XO("Compacted project file"), XO("Compact"), UndoPush::CONSOLIDATE);

      // Now we can remove all previous states.
      auto numStates = undoManager.GetNumStates();
      undoManager.RemoveStates(numStates - 1);

      // And clear the clipboard
      clipboard.Clear();

      // Refresh the before space usage since it may have changed due to the
      // above actions.
      auto before = wxFileName::GetSize(projectFileIO.GetFileName());

      projectFileIO.Compact(currentTracks, true);

      auto after = wxFileName::GetSize(projectFileIO.GetFileName());

      AudacityMessageBox(
         XO("Compacting actually freed %s of disk space.")
         .Format(Internat::FormatSize((before - after).GetValue())),
         XO("Compact Project"));
   }

   currentTracks.reset();
}

void OnSave(const CommandContext &context )
{
   auto &project = context.project;
   auto &projectFileManager = ProjectFileManager::Get( project );
   projectFileManager.Save();
}

void OnSaveAs(const CommandContext &context )
{
   auto &project = context.project;
   auto &projectFileManager = ProjectFileManager::Get( project );
   projectFileManager.SaveAs();
}

void OnSaveCopy(const CommandContext &context )
{
   auto &project = context.project;
   auto &projectFileManager = ProjectFileManager::Get( project );
   projectFileManager.SaveCopy();
}

void OnExportMp3(const CommandContext &context)
{
   auto &project = context.project;
   DoExport(project, "MP3");
}

void OnExportWav(const CommandContext &context)
{
   auto &project = context.project;
   DoExport(project, "WAV");
}

void OnExportOgg(const CommandContext &context)
{
   auto &project = context.project;
   DoExport(project, "OGG");
}

void OnExportAudio(const CommandContext &context)
{
   auto &project = context.project;
   DoExport(project, "");
}

void OnExportSelection(const CommandContext &context)
{
   auto &project = context.project;
   auto &selectedRegion = ViewInfo::Get( project ).selectedRegion;
   Exporter e{ project };

   e.SetFileDialogTitle( XO("Export Selected Audio") );
   e.Process(true, selectedRegion.t0(),
      selectedRegion.t1());
}

void OnExportLabels(const CommandContext &context)
{
   auto &project = context.project;
   auto &tracks = TrackList::Get( project );
   auto &window = GetProjectFrame( project );

   /* i18n-hint: filename containing exported text from label tracks */
   wxString fName = _("labels.txt");
   auto trackRange = tracks.Any<const LabelTrack>();
   auto numLabelTracks = trackRange.size();

   if (numLabelTracks == 0) {
      AudacityMessageBox( XO("There are no label tracks to export.") );
      return;
   }
   else
      fName = (*trackRange.rbegin())->GetName();

   fName = FileNames::SelectFile(FileNames::Operation::Export,
      XO("Export Labels As:"),
      wxEmptyString,
      fName,
      wxT("txt"),
      { FileNames::TextFiles },
      wxFD_SAVE | wxFD_OVERWRITE_PROMPT | wxRESIZE_BORDER,
      &window);

   if (fName.empty())
      return;

   // Move existing files out of the way.  Otherwise wxTextFile will
   // append to (rather than replace) the current file.

   if (wxFileExists(fName)) {
#ifdef __WXGTK__
      wxString safetyFileName = fName + wxT("~");
#else
      wxString safetyFileName = fName + wxT(".bak");
#endif

      if (wxFileExists(safetyFileName))
         wxRemoveFile(safetyFileName);

      wxRename(fName, safetyFileName);
   }

   wxTextFile f(fName);
   f.Create();
   f.Open();
   if (!f.IsOpened()) {
      AudacityMessageBox(
         XO( "Couldn't write to file: %s" ).Format( fName ) );
      return;
   }

   for (auto lt : trackRange)
      lt->Export(f);

   f.Write();
   f.Close();
}

void OnExportMultiple(const CommandContext &context)
{
   auto &project = context.project;
   ExportMultipleDialog em(&project);

   em.ShowModal();
}

#ifdef USE_MIDI
void OnExportMIDI(const CommandContext &context)
{
   auto &project = context.project;
   auto &tracks = TrackList::Get( project );
   auto &window = GetProjectFrame( project );

   // Make sure that there is
   // exactly one NoteTrack selected.
   const auto range = tracks.Selected< const NoteTrack >();
   const auto numNoteTracksSelected = range.size();

   if(numNoteTracksSelected > 1) {
      AudacityMessageBox(
         XO("Please select only one Note Track at a time.") );
      return;
   }
   else if(numNoteTracksSelected < 1) {
      AudacityMessageBox(
         XO("Please select a Note Track.") );
      return;
   }

   wxASSERT(numNoteTracksSelected);
   if (!numNoteTracksSelected)
      return;

   const auto nt = *range.begin();

   while(true) {

      wxString fName;

      fName = FileNames::SelectFile(FileNames::Operation::Export,
         XO("Export MIDI As:"),
         wxEmptyString,
         fName,
         wxT("mid"),
         {
            { XO("MIDI file"),    { wxT("mid") }, true },
            { XO("Allegro file"), { wxT("gro") }, true },
         },
         wxFD_SAVE | wxFD_OVERWRITE_PROMPT | wxRESIZE_BORDER,
         &window);

      if (fName.empty())
         return;

      if(!fName.Contains(wxT("."))) {
         fName = fName + wxT(".mid");
      }

      // Move existing files out of the way.  Otherwise wxTextFile will
      // append to (rather than replace) the current file.

      if (wxFileExists(fName)) {
#ifdef __WXGTK__
         wxString safetyFileName = fName + wxT("~");
#else
         wxString safetyFileName = fName + wxT(".bak");
#endif

         if (wxFileExists(safetyFileName))
            wxRemoveFile(safetyFileName);

         wxRename(fName, safetyFileName);
      }

      if(fName.EndsWith(wxT(".mid")) || fName.EndsWith(wxT(".midi"))) {
         nt->ExportMIDI(fName);
      } else if(fName.EndsWith(wxT(".gro"))) {
         nt->ExportAllegro(fName);
      } else {
         auto msg = XO(
"You have selected a filename with an unrecognized file extension.\nDo you want to continue?");
         auto title = XO("Export MIDI");
         int id = AudacityMessageBox( msg, title, wxYES_NO );
         if (id == wxNO) {
            continue;
         } else if (id == wxYES) {
            nt->ExportMIDI(fName);
         }
      }
      break;
   }
}
#endif // USE_MIDI

void OnImport(const CommandContext &context)
{
    ImportAudio(context);
}
    
void OnImportLabeledAudio(const CommandContext &context)
{
    ImportAudio(context, true);
}
    
void ImportAudio(const CommandContext &context, bool labelAudio = false)
{
   auto &project = context.project;
   auto &window = ProjectWindow::Get( project );

   auto selectedFiles = ProjectFileManager::ShowOpenDialog(FileNames::Operation::Import);
   if (selectedFiles.size() == 0) {
      Importer::SetLastOpenType({});
      return;
   }

   // PRL:  This affects FFmpegImportPlugin::Open which resets the preference
   // to false.  Should it also be set to true on other paths that reach
   // AudacityProject::Import ?
   gPrefs->Write(wxT("/NewImportingSession"), true);

   selectedFiles.Sort(FileNames::CompareNoCase);

   auto cleanup = finally( [&] {
      Importer::SetLastOpenType({});
      window.HandleResize(); // Adjust scrollers for NEW track sizes.
   } );

   wxString fileName = selectedFiles[0];
   for (size_t ff = 0; ff < selectedFiles.size(); ff++) {
      wxString fileName = selectedFiles[ff];

      FileNames::UpdateDefaultPath(FileNames::Operation::Import, ::wxPathOnly(fileName));

      ProjectFileManager::Get( project ).Import(fileName);
   }

   if (labelAudio) {
           IALLabeler::LabelTrack(context, fileName.ToStdString());
   }

   window.ZoomAfterImport(nullptr);
}

void OnImportLabels(const CommandContext &context)
{
   auto &project = context.project;
   auto &trackFactory = TrackFactory::Get( project );
   auto &tracks = TrackList::Get( project );
   auto &window = ProjectWindow::Get( project );

   wxString fileName =
       FileNames::SelectFile(FileNames::Operation::Open,
         XO("Select a text file containing labels"),
         wxEmptyString,     // Path
         wxT(""),       // Name
         wxT("txt"),   // Extension
         { FileNames::TextFiles, FileNames::AllFiles },
         wxRESIZE_BORDER,        // Flags
         &window);    // Parent

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

      window.ZoomAfterImport(nullptr);
   }
}

#ifdef USE_MIDI
void OnImportMIDI(const CommandContext &context)
{
   auto &project = context.project;
   auto &window = GetProjectFrame( project );

   wxString fileName = FileNames::SelectFile(FileNames::Operation::Open,
      XO("Select a MIDI file"),
      wxEmptyString,     // Path
      wxT(""),       // Name
      wxT(""),       // Extension
      {
         { XO("MIDI and Allegro files"),
           { wxT("mid"), wxT("midi"), wxT("gro"), }, true },
         { XO("MIDI files"),
           { wxT("mid"), wxT("midi"), }, true },
         { XO("Allegro files"),
           { wxT("gro"), }, true },
         FileNames::AllFiles
      },
      wxRESIZE_BORDER,        // Flags
      &window);    // Parent

   if (!fileName.empty())
      DoImportMIDI(project, fileName);
}
#endif

void OnImportRaw(const CommandContext &context)
{
   auto &project = context.project;
   auto &trackFactory = TrackFactory::Get( project );
   auto &window = ProjectWindow::Get( project );

   wxString fileName =
       FileNames::SelectFile(FileNames::Operation::Open,
         XO("Select any uncompressed audio file"),
         wxEmptyString,     // Path
         wxT(""),       // Name
         wxT(""),       // Extension
         { FileNames::AllFiles },
         wxRESIZE_BORDER,        // Flags
         &window);    // Parent

   if (fileName.empty())
      return;

   TrackHolders newTracks;

   ::ImportRaw(&window, fileName, &trackFactory, newTracks);

   if (newTracks.size() <= 0)
      return;

   ProjectFileManager::Get( project )
      .AddImportedTracks(fileName, std::move(newTracks));
   window.HandleResize(); // Adjust scrollers for NEW track sizes.
}

void OnPageSetup(const CommandContext &context)
{
   auto &project = context.project;
   auto &window = GetProjectFrame( project );
   HandlePageSetup(&window);
}

void OnPrint(const CommandContext &context)
{
   auto &project = context.project;
   auto name = project.GetProjectName();
   auto &tracks = TrackList::Get( project );
   auto &window = GetProjectFrame( project );
   HandlePrint(&window, name, &tracks, TrackPanel::Get( project ));
}

void OnExit(const CommandContext &WXUNUSED(context) )
{
   // Simulate the application Exit menu item
   wxCommandEvent evt{ wxEVT_MENU, wxID_EXIT };
   wxTheApp->AddPendingEvent( evt );
}

}; // struct Handler

} // namespace

static CommandHandlerObject &findCommandHandler(AudacityProject &) {
   // Handler is not stateful.  Doesn't need a factory registered with
   // AudacityProject.
   static FileActions::Handler instance;
   return instance;
};

// Menu definitions

#define FN(X) (& FileActions::Handler :: X)

namespace {
using namespace MenuTable;

BaseItemSharedPtr FileMenu()
{
   using Options = CommandManager::Options;

   static BaseItemSharedPtr menu{
   ( FinderScope{ findCommandHandler },
   Menu( wxT("File"), XXO("&File"),
      Section( "Basic",
         /*i18n-hint: "New" is an action (verb) to create a NEW project*/
         Command( wxT("New"), XXO("&New"), FN(OnNew),
            AudioIONotBusyFlag(), wxT("Ctrl+N") ),

         /*i18n-hint: (verb)*/
         Command( wxT("Open"), XXO("&Open..."), FN(OnOpen),
            AudioIONotBusyFlag(), wxT("Ctrl+O") ),

   #ifdef EXPERIMENTAL_RESET
         // Empty the current project and forget its name and path.  DANGEROUS
         // It's just for developers.
         // Do not translate this menu item (no XXO).
         // It MUST not be shown to regular users.
         Command( wxT("Reset"), XXO("&Dangerous Reset..."), FN(OnProjectReset),
            AudioIONotBusyFlag() ),
   #endif

   /////////////////////////////////////////////////////////////////////////////

         Menu( wxT("Recent"),
   #ifdef __WXMAC__
            /* i18n-hint: This is the name of the menu item on Mac OS X only */
            XXO("Open Recent")
   #else
            /* i18n-hint: This is the name of the menu item on Windows and Linux */
            XXO("Recent &Files")
   #endif
            ,
            Special( wxT("PopulateRecentFilesStep"),
            [](AudacityProject &, wxMenu &theMenu){
               // Recent Files and Recent Projects menus
               auto &history = FileHistory::Global();
               history.UseMenu( &theMenu );

               wxWeakRef<wxMenu> recentFilesMenu{ &theMenu };
               wxTheApp->CallAfter( [=] {
                  // Bug 143 workaround.
                  // The bug is in wxWidgets.  For a menu that has scrollers,
                  // the scrollers have an ID of 0 (not wxID_NONE which is -3).
                  // Therefore wxWidgets attempts to find a help string. See
                  // wxFrameBase::ShowMenuHelp(int menuId)
                  // It finds a bogus automatic help string of "Recent &Files"
                  // from that submenu.
                  // So we set the help string for command with Id 0 to empty.
                  if ( recentFilesMenu )
                     recentFilesMenu->GetParent()->SetHelpString( 0, "" );
               } );
            } )
         ),

   /////////////////////////////////////////////////////////////////////////////

         Command( wxT("Close"), XXO("&Close"), FN(OnClose),
            AudioIONotBusyFlag(), wxT("Ctrl+W") )
      ),

      Section( "Save",
         Menu( wxT("Save"), XXO("&Save Project"),
            Command( wxT("Save"), XXO("&Save Project"), FN(OnSave),
               AudioIONotBusyFlag(), wxT("Ctrl+S") ),
            Command( wxT("SaveAs"), XXO("Save Project &As..."), FN(OnSaveAs),
               AudioIONotBusyFlag() ),
            Command( wxT("SaveCopy"), XXO("&Backup Project..."), FN(OnSaveCopy),
               AudioIONotBusyFlag() )
         ),

         Command( wxT("Compact"), XXO("Co&mpact Project"), FN(OnCompact),
            AudioIONotBusyFlag() )
      ),

      Section( "Import-Export",
         Menu( wxT("Export"), XXO("&Export"),
            // Enable Export audio commands only when there are audio tracks.
            Command( wxT("ExportMp3"), XXO("Export as MP&3"), FN(OnExportMp3),
               AudioIONotBusyFlag() | WaveTracksExistFlag() ),

            Command( wxT("ExportWav"), XXO("Export as &WAV"), FN(OnExportWav),
               AudioIONotBusyFlag() | WaveTracksExistFlag() ),

            Command( wxT("ExportOgg"), XXO("Export as &OGG"), FN(OnExportOgg),
               AudioIONotBusyFlag() | WaveTracksExistFlag() ),

            Command( wxT("Export"), XXO("&Export Audio..."), FN(OnExportAudio),
               AudioIONotBusyFlag() | WaveTracksExistFlag(), wxT("Ctrl+Shift+E") ),

            // Enable Export Selection commands only when there's a selection.
            Command( wxT("ExportSel"), XXO("Expo&rt Selected Audio..."),
               FN(OnExportSelection),
               AudioIONotBusyFlag() | TimeSelectedFlag() | WaveTracksSelectedFlag(),
               Options{}.UseStrictFlags() ),

            Command( wxT("ExportLabels"), XXO("Export &Labels..."),
               FN(OnExportLabels),
               AudioIONotBusyFlag() | LabelTracksExistFlag() ),
            // Enable Export audio commands only when there are audio tracks.
            Command( wxT("ExportMultiple"), XXO("Export &Multiple..."),
               FN(OnExportMultiple),
               AudioIONotBusyFlag() | WaveTracksExistFlag(), wxT("Ctrl+Shift+L") )
   #if defined(USE_MIDI)
            ,
            Command( wxT("ExportMIDI"), XXO("Export MI&DI..."), FN(OnExportMIDI),
               AudioIONotBusyFlag() | NoteTracksExistFlag() )
   #endif
         ),

         Menu( wxT("Import"), XXO("&Import"),
            Command( wxT("ImportAudio"), XXO("&Audio..."), FN(OnImport),
               AudioIONotBusyFlag(), wxT("Ctrl+Shift+I") ),
            Command( wxT("ImportLabeledAudio"), XXO("&Labeled Audio..."), FN(OnImportLabeledAudio), AudioIONotBusyFlag() ),
            Command( wxT("ImportLabels"), XXO("&Labels..."), FN(OnImportLabels),
               AudioIONotBusyFlag() ),
      #ifdef USE_MIDI
            Command( wxT("ImportMIDI"), XXO("&MIDI..."), FN(OnImportMIDI),
               AudioIONotBusyFlag() ),
      #endif // USE_MIDI
            Command( wxT("ImportRaw"), XXO("&Raw Data..."), FN(OnImportRaw),
               AudioIONotBusyFlag() )
         )
      ),

      Section( "Print",
         Command( wxT("PageSetup"), XXO("Pa&ge Setup..."), FN(OnPageSetup),
            AudioIONotBusyFlag() | TracksExistFlag() ),
         /* i18n-hint: (verb) It's item on a menu. */
         Command( wxT("Print"), XXO("&Print..."), FN(OnPrint),
            AudioIONotBusyFlag() | TracksExistFlag() )
      ),

      Section( "Exit",
         // On the Mac, the Exit item doesn't actually go here...wxMac will
         // pull it out
         // and put it in the Audacity menu for us based on its ID.
         /* i18n-hint: (verb) It's item on a menu. */
         Command( wxT("Exit"), XXO("E&xit"), FN(OnExit),
            AlwaysEnabledFlag, wxT("Ctrl+Q") )
      )
   ) ) };
   return menu;
}

AttachedItem sAttachment1{
   wxT(""),
   Shared( FileMenu() )
};
}

#undef FN
