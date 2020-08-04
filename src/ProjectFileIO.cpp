/**********************************************************************

Audacity: A Digital Audio Editor

ProjectFileIO.cpp

Paul Licameli split from AudacityProject.cpp

**********************************************************************/

#include "ProjectFileIO.h"

#include <atomic>
#include <sqlite3.h>
#include <wx/crt.h>
#include <wx/frame.h>
#include <wx/progdlg.h>
#include <wx/sstream.h>
#include <wx/xml/xml.h>

#include "ActiveProjects.h"
#include "DBConnection.h"
#include "FileNames.h"
#include "Project.h"
#include "ProjectFileIORegistry.h"
#include "ProjectSerializer.h"
#include "ProjectSettings.h"
#include "SampleBlock.h"
#include "Sequence.h"
#include "Tags.h"
#include "TimeTrack.h"
#include "ViewInfo.h"
#include "WaveClip.h"
#include "WaveTrack.h"
#include "widgets/AudacityMessageBox.h"
#include "widgets/NumericTextCtrl.h"
#include "widgets/ProgressDialog.h"
#include "xml/XMLFileReader.h"

wxDEFINE_EVENT(EVT_PROJECT_TITLE_CHANGE, wxCommandEvent);

static const int ProjectFileID = ('A' << 24 | 'U' << 16 | 'D' << 8 | 'Y');
static const int ProjectFileVersion = 1;

// Navigation:
//
// Bindings are marked out in the code by, e.g. 
// BIND SQL sampleblocks
// A search for "BIND SQL" will find all bindings.
// A search for "SQL sampleblocks" will find all SQL related 
// to sampleblocks.

static const char *ProjectFileSchema =
   // These are persistent and not connection based
   //
   // See the CMakeList.txt for the SQLite lib for more
   // settings.
   "PRAGMA <schema>.application_id = %d;"
   "PRAGMA <schema>.user_version = %d;"
   ""
   // project is a binary representation of an XML file.
   // it's in binary for speed.
   // One instance only.  id is always 1.
   // dict is a dictionary of fieldnames.
   // doc is the binary representation of the XML
   // in the doc, fieldnames are replaced by 2 byte dictionary
   // index numbers.
   // This is all opaque to SQLite.  It just sees two
   // big binary blobs.
   // There is no limit to document blob size.
   // dict will be smallish, with an entry for each 
   // kind of field.
   "CREATE TABLE IF NOT EXISTS <schema>.project"
   "("
   "  id                   INTEGER PRIMARY KEY,"
   "  dict                 BLOB,"
   "  doc                  BLOB"
   ");"
   ""
   // CREATE SQL autosave
   // autosave is a binary representation of an XML file.
   // it's in binary for speed.
   // One instance only.  id is always 1.
   // dict is a dictionary of fieldnames.
   // doc is the binary representation of the XML
   // in the doc, fieldnames are replaced by 2 byte dictionary
   // index numbers.
   // This is all opaque to SQLite.  It just sees two
   // big binary blobs.
   // There is no limit to document blob size.
   // dict will be smallish, with an entry for each 
   // kind of field.
   "CREATE TABLE IF NOT EXISTS <schema>.autosave"
   "("
   "  id                   INTEGER PRIMARY KEY,"
   "  dict                 BLOB,"
   "  doc                  BLOB"
   ");"
   ""
   // CREATE SQL tags
   // tags is not used (yet)
   "CREATE TABLE IF NOT EXISTS <schema>.tags"
   "("
   "  name                 TEXT,"
   "  value                BLOB"
   ");"
   ""
   // CREATE SQL sampleblocks
   // 'samples' are fixed size blocks of int16, int32 or float32 numbers.
   // The blocks may be partially empty.
   // The quantity of valid data in the blocks is
   // provided in the project blob.
   // 
   // sampleformat specifies the format of the samples stored.
   //
   // blockID is a 64 bit number.
   //
   // Rows are immutable -- never updated after addition, but may be
   // deleted.
   //
   // summin to summary64K are summaries at 3 distance scales.
   "CREATE TABLE IF NOT EXISTS <schema>.sampleblocks"
   "("
   "  blockid              INTEGER PRIMARY KEY AUTOINCREMENT,"
   "  sampleformat         INTEGER,"
   "  summin               REAL,"
   "  summax               REAL,"
   "  sumrms               REAL,"
   "  summary256           BLOB,"
   "  summary64k           BLOB,"
   "  samples              BLOB"
   ");";

// This singleton handles initialization/shutdown of the SQLite library.
// It is needed because our local SQLite is built with SQLITE_OMIT_AUTOINIT
// defined.
//
// It's safe to use even if a system version of SQLite is used that didn't
// have SQLITE_OMIT_AUTOINIT defined.
class SQLiteIniter
{
public:
   SQLiteIniter()
   {
      // Enable URI filenames for all connections
      mRc = sqlite3_config(SQLITE_CONFIG_URI, 1);

      if (mRc == SQLITE_OK)
      {
         mRc = sqlite3_initialize();
      }

#if !defined(__WXMSW__)
      if (mRc == SQLITE_OK)
      {
         // Use the "unix-excl" VFS to make access to the DB exclusive.  This gets
         // rid of the "<database name>-shm" shared memory file.
         //
         // Though it shouldn't, it doesn't matter if this fails.
         auto vfs = sqlite3_vfs_find("unix-excl");
         if (vfs)
         {
            sqlite3_vfs_register(vfs, 1);
         }
      }
#endif
   }
   ~SQLiteIniter()
   {
      // This function must be called single-threaded only
      // It returns a value, but there's nothing we can do with it
      (void) sqlite3_shutdown();
   }
   int mRc;
};

bool ProjectFileIO::InitializeSQL()
{
   static SQLiteIniter sqliteIniter;
   return sqliteIniter.mRc == SQLITE_OK;
}

static void RefreshAllTitles(bool bShowProjectNumbers )
{
   for ( auto pProject : AllProjects{} ) {
      if ( !GetProjectFrame( *pProject ).IsIconized() ) {
         ProjectFileIO::Get( *pProject ).SetProjectTitle(
            bShowProjectNumbers ? pProject->GetProjectNumber() : -1 );
      }
   }
}

TitleRestorer::TitleRestorer(
   wxTopLevelWindow &window, AudacityProject &project )
{
   if( window.IsIconized() )
      window.Restore();
   window.Raise(); // May help identifying the window on Mac

   // Construct this project's name and number.
   sProjName = project.GetProjectName();
   if ( sProjName.empty() ) {
      sProjName = _("<untitled>");
      UnnamedCount = std::count_if(
         AllProjects{}.begin(), AllProjects{}.end(),
         []( const AllProjects::value_type &ptr ){
            return ptr->GetProjectName().empty();
         }
      );
      if ( UnnamedCount > 1 ) {
         sProjNumber.Printf(
            _("[Project %02i] "), project.GetProjectNumber() + 1 );
         RefreshAllTitles( true );
      } 
   }
   else
      UnnamedCount = 0;
}

TitleRestorer::~TitleRestorer() {
   if( UnnamedCount > 1 )
      RefreshAllTitles( false );
}

static const AudacityProject::AttachedObjects::RegisteredFactory sFileIOKey{
   []( AudacityProject &parent ){
      auto result = std::make_shared< ProjectFileIO >( parent );
      return result;
   }
};

ProjectFileIO &ProjectFileIO::Get( AudacityProject &project )
{
   auto &result = project.AttachedObjects::Get< ProjectFileIO >( sFileIOKey );
   return result;
}

const ProjectFileIO &ProjectFileIO::Get( const AudacityProject &project )
{
   return Get( const_cast< AudacityProject & >( project ) );
}

ProjectFileIO::ProjectFileIO(AudacityProject &project)
   : mProject{ project }
{
   mPrevConn = nullptr;

   mRecovered = false;
   mModified = false;
   mTemporary = true;

   UpdatePrefs();
}

ProjectFileIO::~ProjectFileIO()
{
}

sqlite3 *ProjectFileIO::DB()
{
   auto &curConn = CurrConn();
   if (!curConn)
   {
      if (!OpenConnection())
      {
         throw SimpleMessageBoxException
         {
            XO("Failed to open the project's database")
         };
      }
   }

   return curConn->DB();
}

bool ProjectFileIO::OpenConnection(FilePath fileName /* = {}  */)
{
   auto &curConn = CurrConn();
   wxASSERT(!curConn);
   bool isTemp = false;

   if (fileName.empty())
   {
      fileName = GetFileName();
      if (fileName.empty())
      {
         fileName = FileNames::UnsavedProjectFileName();
         isTemp = true;
      }
   }
   else
   {
      // If this project resides in the temporary directory, then we'll mark it
      // as temporary.
      wxFileName temp(FileNames::TempDir(), wxT(""));
      wxFileName file(fileName);
      file.SetFullName(wxT(""));
      if (file == temp)
      {
         isTemp = true;
      }
   }

   // Pass weak_ptr to project into DBConnection constructor
   curConn = std::make_unique<DBConnection>(mProject.shared_from_this());
   if (!curConn->Open(fileName))
   {
      curConn.reset();
      return false;
   }

   if (!CheckVersion())
   {
      CloseConnection();
      return false;
   }

   mTemporary = isTemp;

   SetFileName(fileName);

   return true;
}

bool ProjectFileIO::CloseConnection()
{
   auto &curConn = CurrConn();
   wxASSERT(curConn);

   if (!curConn->Close())
   {
      return false;
   }
   curConn.reset();

   SetFileName({});

   return true;
}

// Put the current database connection aside, keeping it open, so that
// another may be opened with OpenConnection()
void ProjectFileIO::SaveConnection()
{
   // Should do nothing in proper usage, but be sure not to leak a connection:
   DiscardConnection();

   mPrevConn = std::move(CurrConn());
   mPrevFileName = mFileName;
   mPrevTemporary = mTemporary;

   SetFileName({});
}

// Close any set-aside connection
void ProjectFileIO::DiscardConnection()
{
   if (mPrevConn)
   {
      if (!mPrevConn->Close())
      {
         // Store an error message
         SetDBError(
            XO("Failed to discard connection")
         );
      }

      // If this is a temporary project, we no longer want to keep the
      // project file.
      if (mPrevTemporary)
      {
         // This is just a safety check.
         wxFileName temp(FileNames::TempDir(), wxT(""));
         wxFileName file(mPrevFileName);
         file.SetFullName(wxT(""));
         if (file == temp)
         {
            wxRemoveFile(mPrevFileName);
         }
      }
      mPrevConn = nullptr;
      mPrevFileName.clear();
   }
}

// Close any current connection and switch back to using the saved
void ProjectFileIO::RestoreConnection()
{
   auto &curConn = CurrConn();
   if (curConn)
   {
      if (!curConn->Close())
      {
         // Store an error message
         SetDBError(
            XO("Failed to restore connection")
         );
      }
   }

   curConn = std::move(mPrevConn);
   SetFileName(mPrevFileName);
   mTemporary = mPrevTemporary;

   mPrevFileName.clear();
}

void ProjectFileIO::UseConnection(Connection &&conn, const FilePath &filePath)
{
   auto &curConn = CurrConn();
   wxASSERT(!curConn);

   curConn = std::move(conn);
   SetFileName(filePath);
}

bool ProjectFileIO::TransactionStart(const wxString &name)
{
   char *errmsg = nullptr;

   int rc = sqlite3_exec(DB(),
                         wxT("SAVEPOINT ") + name + wxT(";"),
                         nullptr,
                         nullptr,
                         &errmsg);

   if (errmsg)
   {
      SetDBError(
         XO("Failed to create savepoint:\n\n%s").Format(name)
      );
      sqlite3_free(errmsg);
   }

   return rc == SQLITE_OK;
}

bool ProjectFileIO::TransactionCommit(const wxString &name)
{
   char *errmsg = nullptr;

   int rc = sqlite3_exec(DB(),
                         wxT("RELEASE ") + name + wxT(";"),
                         nullptr,
                         nullptr,
                         &errmsg);

   if (errmsg)
   {
      SetDBError(
         XO("Failed to release savepoint:\n\n%s").Format(name)
      );
      sqlite3_free(errmsg);
   }

   return rc == SQLITE_OK;
}

bool ProjectFileIO::TransactionRollback(const wxString &name)
{
   char *errmsg = nullptr;

   int rc = sqlite3_exec(DB(),
                         wxT("ROLLBACK TO ") + name + wxT(";"),
                         nullptr,
                         nullptr,
                         &errmsg);

   if (errmsg)
   {
      SetDBError(
         XO("Failed to release savepoint:\n\n%s").Format(name)
      );
      sqlite3_free(errmsg);
   }

   return rc == SQLITE_OK;
}

static int ExecCallback(void *data, int cols, char **vals, char **names)
{
   auto &cb = *static_cast<const ProjectFileIO::ExecCB *>(data);
   // Be careful not to throw anything across sqlite3's stack frames.
   return GuardedCall<int>(
      [&]{ return cb(cols, vals, names); },
      MakeSimpleGuard( 1 )
   );
}

int ProjectFileIO::Exec(const char *query, const ExecCB &callback)
{
   char *errmsg = nullptr;

   const void *ptr = &callback;
   int rc = sqlite3_exec(DB(), query, ExecCallback,
                         const_cast<void*>(ptr), &errmsg);

   if (rc != SQLITE_ABORT && errmsg)
   {
      SetDBError(
         XO("Failed to execute a project file command:\n\n%s").Format(query)
      );
      mLibraryError = Verbatim(errmsg);
   }
   if (errmsg)
   {
      sqlite3_free(errmsg);
   }

   return rc;
}

bool ProjectFileIO::Query(const char *sql, const ExecCB &callback)
{
   int rc = Exec(sql, callback);
   // SQLITE_ABORT is a non-error return only meaning the callback
   // stopped the iteration of rows early
   if ( !(rc == SQLITE_OK || rc == SQLITE_ABORT) )
   {
      return false;
   }

   return true;
}

bool ProjectFileIO::GetValue(const char *sql, wxString &result)
{
   // Retrieve the first column in the first row, if any
   result.clear();
   auto cb = [&result](int cols, char **vals, char **){
      if (cols > 0)
         result = vals[0];
      // Stop after one row
      return 1;
   };

   return Query(sql, cb);
}

bool ProjectFileIO::GetBlob(const char *sql, wxMemoryBuffer &buffer)
{
   auto db = DB();
   int rc;

   buffer.Clear();

   sqlite3_stmt *stmt = nullptr;
   auto cleanup = finally([&]
   {
      if (stmt)
      {
         sqlite3_finalize(stmt);
      }
   });

   rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
   if (rc != SQLITE_OK)
   {
      SetDBError(
         XO("Unable to prepare project file command:\n\n%s").Format(sql)
      );
      return false;
   }

   rc = sqlite3_step(stmt);

   // A row wasn't found...not an error
   if (rc == SQLITE_DONE)
   {
      return true;
   }

   if (rc != SQLITE_ROW)
   {
      SetDBError(
         XO("Failed to retrieve data from the project file.\nThe following command failed:\n\n%s").Format(sql)
      );
      // AUD TODO handle error
      return false;
   }

   const void *blob = sqlite3_column_blob(stmt, 0);
   int size = sqlite3_column_bytes(stmt, 0);

   buffer.AppendData(blob, size);

   return true;
}

bool ProjectFileIO::CheckVersion()
{
   auto db = DB();
   int rc;

   // Install our schema if this is an empty DB
   wxString result;
   if (!GetValue("SELECT Count(*) FROM sqlite_master WHERE type='table';", result))
   {
      return false;
   }

   // If the return count is zero, then there are no tables defined, so this
   // must be a new project file.
   if (wxStrtol<char **>(result, nullptr, 10) == 0)
   {
      return InstallSchema(db);
   }

   // Check for our application ID
   if (!GetValue("PRAGMA application_ID;", result))
   {
      return false;
   }

   // It's a database that SQLite recognizes, but it's not one of ours
   if (wxStrtoul<char **>(result, nullptr, 10) != ProjectFileID)
   {
      SetError(XO("This is not an Audacity project file"));
      return false;
   }

   // Get the project file version
   if (!GetValue("PRAGMA user_version;", result))
   {
      return false;
   }

   long version = wxStrtol<char **>(result, nullptr, 10);

   // Project file version is higher than ours. We will refuse to
   // process it since we can't trust anything about it.
   if (version > ProjectFileVersion)
   {
      SetError(
         XO("This project was created with a newer version of Audacity:\n\nYou will need to upgrade to process it")
      );
      return false;
   }
   
   // Project file is older than ours, ask the user if it's okay to
   // upgrade.
   if (version < ProjectFileVersion)
   {
      return UpgradeSchema();
   }

   return true;
}

bool ProjectFileIO::InstallSchema(sqlite3 *db, const char *schema /* = "main" */)
{
   int rc;

   wxString sql;
   sql.Printf(ProjectFileSchema, ProjectFileID, ProjectFileVersion);
   sql.Replace("<schema>", schema);

   rc = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
   if (rc != SQLITE_OK)
   {
      SetDBError(
         XO("Unable to initialize the project file")
      );
      return false;
   }

   return true;
}

bool ProjectFileIO::UpgradeSchema()
{
   // To do
   return true;
}

// The orphan block handling should be removed once autosave and related
// blocks become part of the same transaction.

// An SQLite function that takes a blockid and looks it up in a set of
// blockids captured during project load.  If the blockid isn't found
// in the set, it will be deleted.
void ProjectFileIO::InSet(sqlite3_context *context, int argc, sqlite3_value **argv)
{
   BlockIDs *blockids = (BlockIDs *) sqlite3_user_data(context);
   SampleBlockID blockid = sqlite3_value_int64(argv[0]);

   sqlite3_result_int(context, blockids->find(blockid) != blockids->end());
}

bool ProjectFileIO::DeleteBlocks(const BlockIDs &blockids, bool complement)
{
   auto db = DB();
   int rc;

   auto cleanup = finally([&]
   {
      // Remove our function, whether it was successfully defined or not.
      sqlite3_create_function(db, "inset", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC, nullptr, nullptr, nullptr, nullptr);
   });

   // Add the function used to verify each row's blockid against the set of active blockids
   const void *p = &blockids;
   rc = sqlite3_create_function(db, "inset", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC, const_cast<void*>(p), InSet, nullptr, nullptr);
   if (rc != SQLITE_OK)
   {
      wxLogDebug(wxT("Unable to add 'inset' function"));
      return false;
   }

   // Delete all rows in the set, or not in it
   auto sql = wxString::Format(
      "DELETE FROM sampleblocks WHERE %sinset(blockid);",
      complement ? "NOT " : "" );
   rc = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
   if (rc != SQLITE_OK)
   {
      wxLogWarning(XO("Cleanup of orphan blocks failed").Translation());
      return false;
   }

   // Mark the project recovered if we deleted any rows
   int changes = sqlite3_changes(db);
   if (changes > 0)
   {
      wxLogInfo(XO("Total orphan blocks deleted %d").Translation(), changes);
      mRecovered = true;
   }

   return true;
}

bool ProjectFileIO::CopyTo(const FilePath &destpath,
                           const TranslatableString &msg,
                           bool isTemporary,
                           bool prune /* = false */,
                           const std::shared_ptr<TrackList> &tracks /* = nullptr */)
{
   // Get access to the active tracklist
   auto pProject = &mProject;
   auto &tracklist = tracks ? *tracks : TrackList::Get(*pProject);

   SampleBlockIDSet blockids;

   // Collect all active blockids
   if (prune)
   {
      InspectBlocks( tracklist, {}, &blockids );
   }
   // Collect ALL blockids
   else
   {
      auto cb = [&blockids](int cols, char **vals, char **){
         SampleBlockID blockid;
         wxString{ vals[0] }.ToLongLong(&blockid);
         blockids.insert(blockid);
         return 0;
      };

      if (!Query("SELECT blockid FROM sampleblocks;", cb))
      {
         return false;
      }
   }

   // Create the project doc
   ProjectSerializer doc;
   WriteXMLHeader(doc);
   WriteXML(doc, false, tracks);

   auto db = DB();
   Connection destConn = nullptr;
   bool success = false;
   int rc;
   ProgressResult res = ProgressResult::Success;

   // Cleanup in case things go awry
   auto cleanup = finally([&]
   {
      if (!success)
      {
         if (destConn)
         {
            destConn->Close();
            destConn = nullptr;
         }

         sqlite3_exec(db, "DETACH DATABASE outbound;", nullptr, nullptr, nullptr);

         wxRemoveFile(destpath);
      }
   });

   // Attach the destination database 
   wxString sql;
   sql.Printf("ATTACH DATABASE '%s' AS outbound;", destpath);

   rc = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
   if (rc != SQLITE_OK)
   {
      SetDBError(
         XO("Unable to attach destination database")
      );
      return false;
   }

   // Ensure attached DB connection gets configured
   //
   // NOTE:  Between the above attach and setting the mode here, a normal DELETE
   //        mode journal will be used and will briefly appear in the filesystem.
   CurrConn()->FastMode("outbound");

   // Install our schema into the new database
   if (!InstallSchema(db, "outbound"))
   {
      // Message already set
      return false;
   }

   // Copy over tags (not really used yet)
   rc = sqlite3_exec(db,
                     "INSERT INTO outbound.tags SELECT * FROM main.tags;",
                     nullptr,
                     nullptr,
                     nullptr);
   if (rc != SQLITE_OK)
   {
      SetDBError(
         XO("Failed to copy tags")
      );

      return false;
   }

   {
      // Ensure statement gets cleaned up
      sqlite3_stmt *stmt = nullptr;
      auto cleanup = finally([&]
      {
         if (stmt)
         {
            sqlite3_finalize(stmt);
         }
      });

      // Prepare the statement only once
      rc = sqlite3_prepare_v2(db,
                              "INSERT INTO outbound.sampleblocks"
                              "  SELECT * FROM main.sampleblocks"
                              "  WHERE blockid = ?;",
                              -1,
                              &stmt,
                              nullptr);
      if (rc != SQLITE_OK)
      {
         SetDBError(
            XO("Unable to prepare project file command:\n\n%s").Format(sql)
         );
         return false;
      }

      /* i18n-hint: This title appears on a dialog that indicates the progress
         in doing something.*/
      ProgressDialog progress(XO("Progress"), msg, pdlgHideStopButton);
      ProgressResult result = ProgressResult::Success;

      wxLongLong_t count = 0;
      wxLongLong_t total = blockids.size();

      // Start a transaction.  Since we're running without a journal,
      // this really doesn't provide rollback.  It just prevents SQLite
      // from auto committing after each step through the loop.
      //
      // Also note that we will have an open transaction if we fail
      // while copying the blocks. This is fine since we're just going
      // to delete the database anyway.
      sqlite3_exec(db, "BEGIN;", nullptr, nullptr, nullptr);

      // Copy sample blocks from the main DB to the outbound DB
      for (auto blockid : blockids)
      {
         // Bind statement parameters
         if (sqlite3_bind_int64(stmt, 1, blockid) != SQLITE_OK)
         {
            wxASSERT_MSG(false, wxT("Binding failed...bug!!!"));
         }

         // Process it
         rc = sqlite3_step(stmt);
         if (rc != SQLITE_DONE)
         {
            SetDBError(
               XO("Failed to update the project file.\nThe following command failed:\n\n%s").Format(sql)
            );
            return false;
         }

         // Reset statement to beginning
         if (sqlite3_reset(stmt) != SQLITE_OK)
         {
            THROW_INCONSISTENCY_EXCEPTION;
         }

         result = progress.Update(++count, total);
         if (result != ProgressResult::Success)
         {
            // Note that we're not setting success, so the finally
            // block above will take care of cleaning up
            return false;
         }
      }

      // Write the doc.
      //
      // If we're compacting a temporary project (user initiated from the File
      // menu), then write the doc to the "autosave" table since temporary
      // projects do not have a "project" doc.
      if (!WriteDoc(isTemporary ? "autosave" : "project", doc, "outbound"))
      {
         return false;
      }

      // See BEGIN above...
      sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);
   }

   // Detach the destination database
   rc = sqlite3_exec(db, "DETACH DATABASE outbound;", nullptr, nullptr, nullptr);
   if (rc != SQLITE_OK)
   {
      SetDBError(
         XO("Destination project could not be detached")
      );

      return false;
   }

   // Tell cleanup everything is good to go
   success = true;

   return true;
}

bool ProjectFileIO::ShouldCompact(const std::shared_ptr<TrackList> &tracks)
{
   SampleBlockIDSet active;
   unsigned long long current = 0;

   InspectBlocks( *tracks,
      BlockSpaceUsageAccumulator( current ),
      &active // Visit unique blocks only
   );

   // Get the number of blocks and total length from the project file.
   unsigned long long total = GetTotalUsage();
   unsigned long long blockcount = 0;
   
   auto cb = [&blockcount](int cols, char **vals, char **)
   {
      // Convert
      wxString(vals[0]).ToULongLong(&blockcount);
      return 0;
   };

   if (!Query("SELECT Count(*) FROM sampleblocks;", cb) || blockcount == 0)
   {
      // Shouldn't compact since we don't have the full picture
      return false;
   }

   // Remember if we had unused blocks in the project file
   mHadUnused = (blockcount > active.size());

   // Let's make a percentage...should be plenty of head room
   current *= 100;

   wxLogDebug(wxT("used = %lld total = %lld %lld"), current, total, total ? current / total : 0);
   if (!total || current / total > 80)
   {
      wxLogDebug(wxT("not compacting"));
      return false;
   }
   wxLogDebug(wxT("compacting"));

   return true;
}

Connection &ProjectFileIO::CurrConn()
{
   auto &connectionPtr = ConnectionPtr::Get( mProject );
   return connectionPtr.mpConnection;
}

void ProjectFileIO::Compact(const std::shared_ptr<TrackList> &tracks, bool force /* = false */)
{
   // Haven't compacted yet
   mWasCompacted = false;

   // Assume we have unused block until we found out otherwise. That way cleanup
   // at project close time will still occur.
   mHadUnused = true;

   // Don't compact if this is a temporary project or if it's determined there are not
   // enough unused blocks to make it worthwhile
   if (!force)
   {
      if (IsTemporary() || !ShouldCompact(tracks))
      {
         // Delete the AutoSave doc it if exists
         if (IsModified())
         {
            // PRL:  not clear what to do if the following fails, but the worst should
            // be, the project may reopen in its present state as a recovery file, not
            // at the last saved state.
            (void) AutoSaveDelete();
         }

         return;
      }
   }

   wxString origName = mFileName;
   wxString backName = origName + "_compact_back";
   wxString tempName = origName + "_compact_temp";

   // Copy the original database to a new database. Only prune sample blocks if
   // we have a tracklist.
   if (CopyTo(tempName, XO("Compacting project"), IsTemporary(), tracks != nullptr, tracks))
   {
      // Must close the database to rename it
      if (CloseConnection())
      {
         // Only use the new file if it is actually smaller than the original.
         //
         // If the original file doesn't have anything to compact (original and new
         // are basically identical), the file could grow by a few pages because of
         // differences in how SQLite constructs the b-tree.
         //
         // In this case, just toss the new file and continue to use the original.
         //
         // Also, do this after closing the connection so that the -wal file
         // gets cleaned up.
         if (wxFileName::GetSize(tempName) < wxFileName::GetSize(origName))
         {
            // Rename the original to backup
            if (wxRenameFile(origName, backName))
            {
               // Rename the temporary to original
               if (wxRenameFile(tempName, origName))
               {
                  // Open the newly compacted original file
                  OpenConnection(origName);

                  // Remove the old original file
                  wxRemoveFile(backName);

                  // Remember that we compacted
                  mWasCompacted = true;

                  return;
               }

               wxRenameFile(backName, origName);
            }
         }

         OpenConnection(origName);
      }

      wxRemoveFile(tempName);
   }

   return;
}

bool ProjectFileIO::WasCompacted()
{
   return mWasCompacted;
}

bool ProjectFileIO::HadUnused()
{
   return mHadUnused;
}

void ProjectFileIO::UpdatePrefs()
{
   SetProjectTitle();
}

// Pass a number in to show project number, or -1 not to.
void ProjectFileIO::SetProjectTitle(int number)
{
   auto &project = mProject;
   auto pWindow = project.GetFrame();
   if (!pWindow)
   {
      return;
   }
   auto &window = *pWindow;
   wxString name = project.GetProjectName();

   // If we are showing project numbers, then we also explicitly show "<untitled>" if there
   // is none.
   if (number >= 0)
   {
      name =
      /* i18n-hint: The %02i is the project number, the %s is the project name.*/
      XO("[Project %02i] Audacity \"%s\"")
         .Format( number + 1,
                 name.empty() ? XO("<untitled>") : Verbatim((const char *)name))
         .Translation();
   }
   // If we are not showing numbers, then <untitled> shows as 'Audacity'.
   else if (name.empty())
   {
      name = _TS("Audacity");
   }

   if (mRecovered)
   {
      name += wxT(" ");
      /* i18n-hint: E.g this is recovered audio that had been lost.*/
      name += _("(Recovered)");
   }

   if (name != window.GetTitle())
   {
      window.SetTitle( name );
      window.SetName(name);       // to make the nvda screen reader read the correct title

      project.QueueEvent(
         safenew wxCommandEvent{ EVT_PROJECT_TITLE_CHANGE } );
   }
}

const FilePath &ProjectFileIO::GetFileName() const
{
   return mFileName;
}

void ProjectFileIO::SetFileName(const FilePath &fileName)
{
   auto &project = mProject;

   if (!mFileName.empty())
   {
      ActiveProjects::Remove(mFileName);
   }

   mFileName = fileName;

   if (!mFileName.empty())
   {
      ActiveProjects::Add(mFileName);
   }

   if (IsTemporary())
   {
      project.SetProjectName({});
   }
   else
   {
      project.SetProjectName(wxFileName(mFileName).GetName());
   }

   SetProjectTitle();
}

bool ProjectFileIO::HandleXMLTag(const wxChar *tag, const wxChar **attrs)
{
   auto &project = mProject;
   auto &window = GetProjectFrame(project);
   auto &viewInfo = ViewInfo::Get(project);
   auto &settings = ProjectSettings::Get(project);

   wxString fileVersion;
   wxString audacityVersion;
   int requiredTags = 0;
   long longVpos = 0;

   // loop through attrs, which is a null-terminated list of
   // attribute-value pairs
   while (*attrs)
   {
      const wxChar *attr = *attrs++;
      const wxChar *value = *attrs++;

      if (!value || !XMLValueChecker::IsGoodString(value))
      {
         break;
      }

      if (viewInfo.ReadXMLAttribute(attr, value))
      {
         // We need to save vpos now and restore it below
         longVpos = std::max(longVpos, long(viewInfo.vpos));
         continue;
      }

      else if (!wxStrcmp(attr, wxT("version")))
      {
         fileVersion = value;
         requiredTags++;
      }

      else if (!wxStrcmp(attr, wxT("audacityversion")))
      {
         audacityVersion = value;
         requiredTags++;
      }

      else if (!wxStrcmp(attr, wxT("rate")))
      {
         double rate;
         Internat::CompatibleToDouble(value, &rate);
         settings.SetRate( rate );
      }

      else if (!wxStrcmp(attr, wxT("snapto")))
      {
         settings.SetSnapTo(wxString(value) == wxT("on") ? true : false);
      }

      else if (!wxStrcmp(attr, wxT("selectionformat")))
      {
         settings.SetSelectionFormat(
            NumericConverter::LookupFormat( NumericConverter::TIME, value) );
      }

      else if (!wxStrcmp(attr, wxT("audiotimeformat")))
      {
         settings.SetAudioTimeFormat(
            NumericConverter::LookupFormat( NumericConverter::TIME, value) );
      }

      else if (!wxStrcmp(attr, wxT("frequencyformat")))
      {
         settings.SetFrequencySelectionFormatName(
            NumericConverter::LookupFormat( NumericConverter::FREQUENCY, value ) );
      }

      else if (!wxStrcmp(attr, wxT("bandwidthformat")))
      {
         settings.SetBandwidthSelectionFormatName(
            NumericConverter::LookupFormat( NumericConverter::BANDWIDTH, value ) );
      }
   } // while

   if (longVpos != 0)
   {
      // PRL: It seems this must happen after SetSnapTo
      viewInfo.vpos = longVpos;
   }

   if (requiredTags < 2)
   {
      return false;
   }

   // Parse the file version from the project
   int fver;
   int frel;
   int frev;
   if (!wxSscanf(fileVersion, wxT("%i.%i.%i"), &fver, &frel, &frev))
   {
      return false;
   }

   // Parse the file version Audacity was build with
   int cver;
   int crel;
   int crev;
   wxSscanf(wxT(AUDACITY_FILE_FORMAT_VERSION), wxT("%i.%i.%i"), &cver, &crel, &crev);

   if (cver < fver || crel < frel || crev < frev)
   {
      /* i18n-hint: %s will be replaced by the version number.*/
      auto msg = XO("This file was saved using Audacity %s.\nYou are using Audacity %s. You may need to upgrade to a newer version to open this file.")
         .Format(audacityVersion, AUDACITY_VERSION_STRING);

      AudacityMessageBox(
         msg,
         XO("Can't open project file"),
         wxOK | wxICON_EXCLAMATION | wxCENTRE,
         &window);

      return false;
   }

   if (wxStrcmp(tag, wxT("project")))
   {
      return false;
   }

   // All other tests passed, so we succeed
   return true;
}

XMLTagHandler *ProjectFileIO::HandleXMLChild(const wxChar *tag)
{
   auto &project = mProject;
   auto fn = ProjectFileIORegistry::Lookup(tag);
   if (fn)
   {
      return fn(project);
   }

   return nullptr;
}

void ProjectFileIO::WriteXMLHeader(XMLWriter &xmlFile) const
{
   xmlFile.Write(wxT("<?xml "));
   xmlFile.Write(wxT("version=\"1.0\" "));
   xmlFile.Write(wxT("standalone=\"no\" "));
   xmlFile.Write(wxT("?>\n"));

   xmlFile.Write(wxT("<!DOCTYPE "));
   xmlFile.Write(wxT("project "));
   xmlFile.Write(wxT("PUBLIC "));
   xmlFile.Write(wxT("\"-//audacityproject-1.3.0//DTD//EN\" "));
   xmlFile.Write(wxT("\"http://audacity.sourceforge.net/xml/audacityproject-1.3.0.dtd\" "));
   xmlFile.Write(wxT(">\n"));
}

void ProjectFileIO::WriteXML(XMLWriter &xmlFile,
                             bool recording /* = false */,
                             const std::shared_ptr<TrackList> &tracks /* = nullptr */)
// may throw
{
   auto &proj = mProject;
   auto &tracklist = tracks ? *tracks : TrackList::Get(proj);
   auto &viewInfo = ViewInfo::Get(proj);
   auto &tags = Tags::Get(proj);
   const auto &settings = ProjectSettings::Get(proj);

   //TIMER_START( "AudacityProject::WriteXML", xml_writer_timer );

   xmlFile.StartTag(wxT("project"));
   xmlFile.WriteAttr(wxT("xmlns"), wxT("http://audacity.sourceforge.net/xml/"));

   xmlFile.WriteAttr(wxT("version"), wxT(AUDACITY_FILE_FORMAT_VERSION));
   xmlFile.WriteAttr(wxT("audacityversion"), AUDACITY_VERSION_STRING);

   viewInfo.WriteXMLAttributes(xmlFile);
   xmlFile.WriteAttr(wxT("rate"), settings.GetRate());
   xmlFile.WriteAttr(wxT("snapto"), settings.GetSnapTo() ? wxT("on") : wxT("off"));
   xmlFile.WriteAttr(wxT("selectionformat"),
                     settings.GetSelectionFormat().Internal());
   xmlFile.WriteAttr(wxT("frequencyformat"),
                     settings.GetFrequencySelectionFormatName().Internal());
   xmlFile.WriteAttr(wxT("bandwidthformat"),
                     settings.GetBandwidthSelectionFormatName().Internal());

   tags.WriteXML(xmlFile);

   unsigned int ndx = 0;
   tracklist.Any().Visit([&](Track *t)
   {
      auto useTrack = t;
      if ( recording ) {
         // When append-recording, there is a temporary "shadow" track accumulating
         // changes and displayed on the screen but it is not yet part of the
         // regular track list.  That is the one that we want to back up.
         // SubstitutePendingChangedTrack() fetches the shadow, if the track has
         // one, else it gives the same track back.
         useTrack = t->SubstitutePendingChangedTrack().get();
      }
      else if ( useTrack->GetId() == TrackId{} ) {
         // This is a track added during a non-appending recording that is
         // not yet in the undo history.  The UndoManager skips backing it up
         // when pushing.  Don't auto-save it.
         return;
      }
      useTrack->WriteXML(xmlFile);
   });

   xmlFile.EndTag(wxT("project"));

   //TIMER_STOP( xml_writer_timer );
}

bool ProjectFileIO::AutoSave(bool recording)
{
   ProjectSerializer autosave;
   WriteXMLHeader(autosave);
   WriteXML(autosave, recording);

   if (WriteDoc("autosave", autosave))
   {
      mModified = true;
      return true;
   }

   return false;
}

bool ProjectFileIO::AutoSaveDelete(sqlite3 *db /* = nullptr */)
{
   int rc;

   if (!db)
   {
      db = DB();
   }

   rc = sqlite3_exec(db, "DELETE FROM autosave;", nullptr, nullptr, nullptr);
   if (rc != SQLITE_OK)
   {
      SetDBError(
         XO("Failed to remove the autosave information from the project file.")
      );
      return false;
   }

   mModified = false;

   return true;
}

bool ProjectFileIO::WriteDoc(const char *table,
                             const ProjectSerializer &autosave,
                             const char *schema /* = "main" */)
{
   auto db = DB();
   int rc;

   // For now, we always use an ID of 1. This will replace the previously
   // writen row every time.
   char sql[256];
   sqlite3_snprintf(sizeof(sql),
                    sql,
                    "INSERT INTO %s.%s(id, dict, doc) VALUES(1, ?1, ?2)"
                    "       ON CONFLICT(id) DO UPDATE SET dict = ?1, doc = ?2;",
                    schema,
                    table);

   sqlite3_stmt *stmt = nullptr;
   auto cleanup = finally([&]
   {
      if (stmt)
      {
         sqlite3_finalize(stmt);
      }
   });

   rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
   if (rc != SQLITE_OK)
   {
      SetDBError(
         XO("Unable to prepare project file command:\n\n%s").Format(sql)
      );
      return false;
   }

   const wxMemoryBuffer &dict = autosave.GetDict();
   const wxMemoryBuffer &data = autosave.GetData();

   // Bind statement parameters
   // Might return SQL_MISUSE which means it's our mistake that we violated
   // preconditions; should return SQL_OK which is 0
   if (sqlite3_bind_blob(stmt, 1, dict.GetData(), dict.GetDataLen(), SQLITE_STATIC) ||
       sqlite3_bind_blob(stmt, 2, data.GetData(), data.GetDataLen(), SQLITE_STATIC))
   {
      wxASSERT_MSG(false, wxT("Binding failed...bug!!!"));
   }

   rc = sqlite3_step(stmt);
   if (rc != SQLITE_DONE)
   {
      SetDBError(
         XO("Failed to update the project file.\nThe following command failed:\n\n%s").Format(sql)
      );
      return false;
   }

   return true;
}

// Importing an AUP3 project into an AUP3 project is a bit different than
// normal importing since we need to copy data from one DB to the other
// while adjusting the sample block IDs to represent the newly assigned
// IDs.
bool ProjectFileIO::ImportProject(const FilePath &fileName)
{
   // Get access to the current project file
   auto db = DB();

   bool success = false;
   bool restore = true;
   int rc;

   // Ensure the inbound database gets detached
   auto detach = finally([&]
   {
      sqlite3_exec(db, "DETACH DATABASE inbound;", nullptr, nullptr, nullptr);
   });

   // Attach the inbound project file
   wxString sql;
   sql.Printf("ATTACH DATABASE 'file:%s?immutable=1&mode=ro' AS inbound;", fileName);

   rc = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
   if (rc != SQLITE_OK)
   {
      SetDBError(
         XO("Unable to attach %s project file").Format(fileName)
      );

      return false;
   }

   // We need either the autosave or project docs from the inbound AUP3
   wxMemoryBuffer buffer;

   // Get the autosave doc, if any
   if (!GetBlob("SELECT dict || doc FROM inbound.project WHERE id = 1;", buffer))
   {
      // Error already set
      return false;
   }

   // If we didn't have an autosave doc, load the project doc instead
   if (buffer.GetDataLen() == 0)
   {
      if (!GetBlob("SELECT dict || doc FROM inbound.autosave WHERE id = 1;", buffer))
      {
         // Error already set
         return false;
      }

      // Missing both the autosave and project docs. This can happen if the
      // system were to crash before the first autosave into a temporary file.
      if (buffer.GetDataLen() == 0)
      {
         SetError(XO("Unable to load project or autosave documents"));
         return false;
      }
   }

   wxString project;
   BlockIDs blockids;

   // Decode it while capturing the associated sample blockids
   project = ProjectSerializer::Decode(buffer, blockids);
   if (project.size() == 0)
   {
      SetError(XO("Unable to decode project document"));

      return false;
   }

   // Parse the project doc
   wxStringInputStream in(project);
   wxXmlDocument doc;
   if (!doc.Load(in))
   {
      return false;
   }

   // Get the root ("project") node
   wxXmlNode *root = doc.GetRoot();
   wxASSERT(root->GetName().IsSameAs(wxT("project")));

   // Soft delete all non-essential attributes to prevent updating the active
   // project. This takes advantage of the knowledge that when a project is
   // parsed, unrecognized attributes are simply ignored.
   //
   // This is necessary because we don't want any of the active project settings
   // to be modified by the inbound project.
   for (wxXmlAttribute *attr = root->GetAttributes(); attr; attr = attr->GetNext())
   {
      wxString name = attr->GetName();
      if (!name.IsSameAs(wxT("version")) && !name.IsSameAs(wxT("audacityversion")))
      {
         attr->SetName(name + wxT("_deleted"));
      }
   }

   // Recursively find and collect all waveblock nodes
   std::vector<wxXmlNode *> blocknodes;
   std::function<void(wxXmlNode *)> findblocks = [&](wxXmlNode *node)
   {
      while (node)
      {
         if (node->GetName().IsSameAs(wxT("waveblock")))
         {
            blocknodes.push_back(node);
         }
         else
         {
            findblocks(node->GetChildren());
         }

         node = node->GetNext();
      }
   };

   // Get access to the active tracklist
   auto pProject = &mProject;
   auto &tracklist = TrackList::Get(*pProject);

   // Search for a timetrack and remove it if the project already has one
   if (*tracklist.Any<TimeTrack>().begin())
   {
      // Find a timetrack and remove it if it exists
      for (wxXmlNode *node = doc.GetRoot()->GetChildren(); node; node = node->GetNext())
      {
         if (node->GetName().IsSameAs(wxT("timetrack")))
         {
            AudacityMessageBox(
               XO("The active project already has a time track and one was encountered in the project being imported, bypassing imported time track."),
               XO("Project Import"),
               wxOK | wxICON_EXCLAMATION | wxCENTRE,
               &GetProjectFrame(*pProject));

            root->RemoveChild(node);
            break;
         }
      }
   }

   // Find all waveblocks in all wavetracks
   for (wxXmlNode *node = doc.GetRoot()->GetChildren(); node; node = node->GetNext())
   {
      if (node->GetName().IsSameAs(wxT("wavetrack")))
      {
         findblocks(node->GetChildren());
      }
   }

   {
      // Cleanup...
      sqlite3_stmt *stmt = nullptr;
      auto cleanup = finally([&]
      {
         // Ensure the prepared statement gets cleaned up
         if (stmt)
         {
            sqlite3_finalize(stmt);
         }
      });

      // Prepare the statement to copy the sample block from the inbound project to the
      // active project.  All columns other than the blockid column get copied.
      wxString columns(wxT("sampleformat, summin, summax, sumrms, summary256, summary64k, samples"));
      sql.Printf("INSERT INTO main.sampleblocks (%s)"
                 "   SELECT %s"
                 "   FROM inbound.sampleblocks"
                 "   WHERE blockid = ?;",
                 columns,
                 columns);

      rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
      if (rc != SQLITE_OK)
      {
         SetDBError(
            XO("Unable to prepare project file command:\n\n%s").Format(sql)
         );
         return false;
      }

      /* i18n-hint: This title appears on a dialog that indicates the progress
         in doing something.*/
      ProgressDialog progress(XO("Progress"), XO("Importing project"), pdlgHideStopButton);
      ProgressResult result = ProgressResult::Success;

      wxLongLong_t count = 0;
      wxLongLong_t total = blocknodes.size();

      sqlite3_exec(db, "BEGIN;", nullptr, nullptr, nullptr);

      // Copy all the sample blocks from the inbound project file into
      // the active one, while remembering which were copied.
      for (auto node : blocknodes)
      {
         // Find the blockid attribute...it should always be there
         wxXmlAttribute *attr = node->GetAttributes();
         while (attr && !attr->GetName().IsSameAs(wxT("blockid")))
         {
            attr = attr->GetNext();
         }
         wxASSERT(attr != nullptr);

         // And get the blockid
         SampleBlockID blockid;
         attr->GetValue().ToLongLong(&blockid);

         // Bind statement parameters
         // Might return SQL_MISUSE which means it's our mistake that we violated
         // preconditions; should return SQL_OK which is 0
         if (sqlite3_bind_int64(stmt, 1, blockid) != SQLITE_OK)
         {
            wxASSERT_MSG(false, wxT("Binding failed...bug!!!"));
         }

         // Process it
         rc = sqlite3_step(stmt);
         if (rc != SQLITE_DONE)
         {
            SetDBError(
               XO("Failed to import sample block.\nThe following command failed:\n\n%s").Format(sql)
            );

            break;
         }

         // Replace the original blockid with the new one
         attr->SetValue(wxString::Format(wxT("%lld"), sqlite3_last_insert_rowid(db)));

         // Reset the statement for the next iteration
         if (sqlite3_reset(stmt) != SQLITE_OK)
         {
            THROW_INCONSISTENCY_EXCEPTION;
         }

         // Remember that we copied this node in case the user cancels
         result = progress.Update(++count, total);
         if (result != ProgressResult::Success)
         {
            break;
         }
      }

      // Bail if the import was cancelled or failed. If the user stopped the
      // import or it completed, then we continue on.
      if (rc != SQLITE_DONE || result == ProgressResult::Cancelled || result == ProgressResult::Failed)
      {
         sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
         return false;
      }

      // Go ahead and commit now
      sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);

      // Copy over tags...likely to produce duplicates...needs work once used
      rc = sqlite3_exec(db,
                        "INSERT INTO main.tags SELECT * FROM inbound.tags;",
                        nullptr,
                        nullptr,
                        nullptr);
      if (rc != SQLITE_OK)
      {
         SetDBError(
            XO("Failed to import tags")
         );

         return false;
      }
   }

   // Recreate the project doc with the revisions we've made above
   wxStringOutputStream output;
   doc.Save(output);

   // Now load the document as normal
   XMLFileReader xmlFile;
   if (!xmlFile.ParseString(this, output.GetString()))
   {
      SetError(
         XO("Unable to parse project information.")
      );
      mLibraryError = xmlFile.GetErrorStr();

      return false;
   }

   return true;
}

bool ProjectFileIO::LoadProject(const FilePath &fileName)
{
   bool success = false;

   auto cleanup = finally([&]
   {
      if (!success)
      {
         RestoreConnection();
      }
   });

   SaveConnection();

   // Open the project file
   if (!OpenConnection(fileName))
   {
      return false;
   }

   BlockIDs blockids;
   wxString project;
   wxMemoryBuffer buffer;
   bool usedAutosave = true;

   // Get the autosave doc, if any
   if (!GetBlob("SELECT dict || doc FROM autosave WHERE id = 1;", buffer))
   {
      // Error already set
      return false;
   }
 
   // If we didn't have an autosave doc, load the project doc instead
   if (buffer.GetDataLen() == 0)
   {
      usedAutosave = false;

      if (!GetBlob("SELECT dict || doc FROM project WHERE id = 1;", buffer))
      {
         // Error already set
         return false;
      }
   }

   // Missing both the autosave and project docs. This can happen if the
   // system were to crash before the first autosave into a temporary file.
   // This should be a recoverable scenario.
   if (buffer.GetDataLen() == 0)
   {
      mRecovered = true;
   }
   else
   {
      // Decode it while capturing the associated sample blockids
      project = ProjectSerializer::Decode(buffer, blockids);
      if (project.empty())
      {
         SetError(XO("Unable to decode project document"));

         return false;
      }

      // Check for orphans blocks...sets mRecovered if any were deleted
      if (blockids.size() > 0)
      {
         if (!DeleteBlocks(blockids, true))
         {
            return false;
         }
      }

      XMLFileReader xmlFile;

      // Load 'er up
      success = xmlFile.ParseString(this, project);
      if (!success)
      {
         SetError(
            XO("Unable to parse project information.")
         );
         mLibraryError = xmlFile.GetErrorStr();
         return false;
      }

      // Remember if we used autosave or not
      if (usedAutosave)
      {
         mRecovered = true;
      }
   }

   // Mark the project modified if we recovered it
   if (mRecovered)
   {
      mModified = true;
   }

   // A previously saved project will have a document in the project table, so
   // we use that knowledge to determine if this file is an unsaved/temporary
   // file or a permanent project file
   wxString result;
   if (!GetValue("SELECT Count(*) FROM project;", result))
   {
      return false;
   }

   mTemporary = !result.IsSameAs(wxT("1"));

   SetFileName(fileName);

   DiscardConnection();

   success = true;

   return true;
}

bool ProjectFileIO::SaveProject(const FilePath &fileName, const std::shared_ptr<TrackList> &lastSaved)
{
   // In the case where we're saving a temporary project to a permanent project,
   // we'll try to simply rename the project to save a bit of time. We then fall
   // through to the normal Save (not SaveAs) processing.
   if (IsTemporary() && mFileName != fileName)
   {
      FilePath savedName = mFileName;
      if (CloseConnection())
      {
         if (wxRenameFile(savedName, fileName))
         {
            if (!OpenConnection(fileName))
            {
               wxRenameFile(fileName, savedName);
               OpenConnection(savedName);
            }
         }
      }
   }

   // If we're saving to a different file than the current one, then copy the
   // current to the new file and make it the active file.
   if (mFileName != fileName)
   {
      // Do NOT prune here since we need to retain the Undo history
      // after we switch to the new file.
      if (!CopyTo(fileName, XO("Saving project"), false))
      {
         return false;
      }

      // Open the newly created database
      Connection newConn = std::make_unique<DBConnection>(mProject.shared_from_this());

      // NOTE: There is a noticeable delay here when dealing with large multi-hour
      //       projects that we just created. The delay occurs in Open() when it
      //       calls SafeMode() and is due to the switch from the NONE journal mode
      //       to the WAL journal mode.
      //
      //       So, we do the Open() in a thread and display a progress dialog. Since
      //       this is currently the only known instance where this occurs, we do the
      //       threading here. If more instances are identified, then the threading
      //       should be moved to DBConnection::Open(), wrapping the SafeMode() call
      //       there.
      {
         std::atomic_bool done = {false};
         bool success = false;
         auto thread = std::thread([&]
         {
            success = newConn->Open(fileName);
            done = true;
         });

         // Provides a progress dialog with indeterminate mode
         wxGenericProgressDialog pd(XO("Syncing").Translation(),
                                    XO("This may take several seconds").Translation(),
                                    300000,     // range
                                    nullptr,    // parent
                                    wxPD_APP_MODAL | wxPD_ELAPSED_TIME | wxPD_SMOOTH);

         // Wait for the checkpoints to end
         while (!done)
         {
            wxMilliSleep(50);
            pd.Pulse();
         }
         thread.join();

         if (!success)
         {
            SetDBError(
               XO("Failed to open copy of project file")
            );

            newConn = nullptr;

            return false;
         }
      }

      // Autosave no longer needed in original project file
      AutoSaveDelete();

      // Try to compact the orignal project file
      Compact(lastSaved ? lastSaved : TrackList::Create(&mProject));

      // Save to close the original project file now
      CloseProject();

      // And make it the active project file 
      UseConnection(std::move(newConn), fileName);
   }
   else
   {
      ProjectSerializer doc;
      WriteXMLHeader(doc);
      WriteXML(doc);

      if (!WriteDoc("project", doc))
      {
         return false;
      }

      // Autosave no longer needed
      AutoSaveDelete();
   }

   // Reaching this point defines success and all the rest are no-fail
   // operations:

   // No longer modified
   mModified = false;

   // No longer recovered
   mRecovered = false;

   // No longer a temporary project
   mTemporary = false;

   // Adjust the title
   SetProjectTitle();

   return true;
}

bool ProjectFileIO::SaveCopy(const FilePath& fileName)
{
   return CopyTo(fileName, XO("Backing up project"), false, true);
}

bool ProjectFileIO::OpenProject()
{
   return OpenConnection();
}

bool ProjectFileIO::CloseProject()
{
   auto &currConn = CurrConn();
   wxASSERT(currConn);

   // Protect...
   if (!currConn)
   {
      return true;
   }

   // Save the filename since CloseConnection() will clear it
   wxString filename = mFileName;

   // Not much we can do if this fails.  The user will simply get
   // the recovery dialog upon next restart.
   if (CloseConnection())
   {
      // If this is a temporary project, we no longer want to keep the
      // project file.
      if (IsTemporary())
      {
         // This is just a safety check.
         wxFileName temp(FileNames::TempDir(), wxT(""));
         wxFileName file(filename);
         file.SetFullName(wxT(""));
         if (file == temp)
         {
            wxRemoveFile(filename);
         }
      }
   }

   return true;
}

bool ProjectFileIO::ReopenProject()
{
   FilePath fileName = mFileName;
   if (!CloseConnection())
   {
      return false;
   }

   return OpenConnection(fileName);
}

bool ProjectFileIO::IsModified() const
{
   return mModified;
}

bool ProjectFileIO::IsTemporary() const
{
   return mTemporary;
}

bool ProjectFileIO::IsRecovered() const
{
   return mRecovered;
}

void ProjectFileIO::Reset()
{
   wxASSERT_MSG(!CurrConn(), wxT("Resetting project with open project file"));

   mModified = false;
   mRecovered = false;

   SetFileName({});
}

wxLongLong ProjectFileIO::GetFreeDiskSpace() const
{
   wxLongLong freeSpace;
   if (wxGetDiskSpace(wxPathOnly(mFileName), NULL, &freeSpace))
   {
      return freeSpace;
   }

   return -1;
}

const TranslatableString &ProjectFileIO::GetLastError() const
{
   return mLastError;
}

const TranslatableString &ProjectFileIO::GetLibraryError() const
{
   return mLibraryError;
}

void ProjectFileIO::SetError(const TranslatableString &msg)
{
   mLastError = msg;
   mLibraryError = {};
}

void ProjectFileIO::SetDBError(const TranslatableString &msg)
{
   auto &currConn = CurrConn();
   mLastError = msg;
   wxLogDebug(wxT("SQLite error: %s"), mLastError.Debug());
   printf("   Lib error: %s", mLastError.Debug().mb_str().data());

   if (currConn)
   {
      mLibraryError = Verbatim(sqlite3_errmsg(currConn->DB()));
      wxLogDebug(wxT("   Lib error: %s"), mLibraryError.Debug());
      printf("   Lib error: %s", mLibraryError.Debug().mb_str().data());
   }
   abort();
   wxASSERT(false);
}

void ProjectFileIO::SetBypass()
{
   auto &currConn = CurrConn();
   if (!currConn)
      return;

   // Determine if we can bypass sample block deletes during shutdown.
   //
   // IMPORTANT:
   // If the project was compacted, then we MUST bypass further
   // deletions since the new file doesn't have the blocks that the
   // Sequences expect to be there.

   currConn->SetBypass( true );

   // Only permanent project files need cleaning at shutdown
   if (!IsTemporary() && !WasCompacted())
   {
      // If we still have unused blocks, then we must not bypass deletions
      // during shutdown.  Otherwise, we would have orphaned blocks the next time
      // the project is opened.
      //
      // An example of when dead blocks will exist is when a user opens a permanent
      // project, adds a track (with samples) to it, and chooses not to save the
      // changes.
      if (HadUnused())
      {
         currConn->SetBypass( false );
      }
   }

   return;
}

int64_t ProjectFileIO::GetBlockUsage(SampleBlockID blockid)
{
   return GetDiskUsage(CurrConn().get(), blockid);
}

int64_t ProjectFileIO::GetCurrentUsage(const std::shared_ptr<TrackList> &tracks)
{
   unsigned long long current = 0;

   InspectBlocks(*tracks, BlockSpaceUsageAccumulator(current), nullptr);

   return current;
}

int64_t ProjectFileIO::GetTotalUsage()
{
   return GetDiskUsage(CurrConn().get(), 0);
}

//
// Returns the amount of disk space used by the specified sample blockid or all
// of the sample blocks if the blockid is 0.  It does this by using the raw SQLite
// pages available from the "sqlite_dbpage" virtual table to traverse the SQLite
// table b-tree described here:  https://www.sqlite.org/fileformat.html
//
int64_t ProjectFileIO::GetDiskUsage(DBConnection *conn, SampleBlockID blockid /* = 0 */)
{
   // Information we need to track our travels through the b-tree
   typedef struct
   {
      int64_t pgno;
      int currentCell;
      int numCells;
      unsigned char data[65536];
   } page;
   std::vector<page> stack;

   int64_t total = 0;
   int64_t found = 0;
   int64_t right = 0;
   int rc;

   // Get the rootpage for the sampleblocks table.
   sqlite3_stmt *stmt =
      conn->Prepare(DBConnection::GetRootPage,
                    "SELECT rootpage FROM sqlite_master WHERE tbl_name = 'sampleblocks';");
   if (stmt == nullptr || sqlite3_step(stmt) != SQLITE_ROW)
   {
      return 0;
   }

   // And store it in our first stack frame
   stack.push_back({sqlite3_column_int64(stmt, 0)});

   // All done with the statement
   sqlite3_clear_bindings(stmt);
   sqlite3_reset(stmt);

   // Prepare/retrieve statement to read raw database page
   stmt = conn->Prepare(DBConnection::GetDBPage,
      "SELECT data FROM sqlite_dbpage WHERE pgno = ?1;");
   if (stmt == nullptr)
   {
      return 0;
   }

   // Traverse the b-tree until we've visited all of the leaf pages or until
   // we find the one corresponding to the passed in sample blockid. Because we
   // use an integer primary key for the sampleblocks table, the traversal will
   // be in ascending blockid sequence.
   do
   {
      // Acces the top stack frame
      page &pg = stack.back();

      // Read the page from the sqlite_dbpage table if it hasn't yet been loaded
      if (pg.numCells == 0)
      {
         // Bind the page number
         sqlite3_bind_int64(stmt, 1, pg.pgno);

         // And retrieve the page
         if (sqlite3_step(stmt) != SQLITE_ROW)
         {
            return 0;
         }

         // Copy the page content to the stack frame
         memcpy(&pg.data,
                sqlite3_column_blob(stmt, 0),
                sqlite3_column_bytes(stmt, 0));

         // And retrieve the total number of cells within it
         pg.numCells = get2(&pg.data[3]);

         // Reset statement for next usage
         sqlite3_clear_bindings(stmt);
         sqlite3_reset(stmt);
      }

      //wxLogDebug("%*.*spgno %lld currentCell %d numCells %d", (stack.size() - 1) * 2, (stack.size() - 1) * 2, "", pg.pgno, pg.currentCell, pg.numCells);

      // Process an interior table b-tree page
      if (pg.data[0] == 0x05)
      {
         // Process the next cell if we haven't examined all of them yet
         if (pg.currentCell < pg.numCells)
         {
            // Remember the right-most leaf page number.
            right = get4(&pg.data[8]);

            // Iterate over the cells.
            //
            // If we're not looking for a specific blockid, then we always push the
            // target page onto the stack and leave the loop after a single iteration.
            //
            // Otherwise, we match the blockid against the highest integer key contained
            // within the cell and if the blockid falls within the cell, we stack the
            // page and stop the iteration.
            //
            // In theory, we could do a binary search for a specific blockid here, but
            // because our sample blocks are always large, we will get very few cells
            // per page...usually 6 or less.
            //
            // In both cases, the stacked page can be either an internal or leaf page.
            bool stacked = false;
            while (pg.currentCell < pg.numCells)
            {
               // Get the offset to this cell using the offset in the cell pointer
               // array.
               //
               // The cell pointer array starts immediately after the page header
               // at offset 12 and the retrieved offset is from the beginning of
               // the page.
               int celloff = get2(&pg.data[12 + (pg.currentCell * 2)]);

               // Bump to the next cell for the next iteration.
               pg.currentCell++;

               // Get the page number this cell describes
               int pagenum = get4(&pg.data[celloff]);

               // And the highest integer key, which starts at offset 4 within the cell.
               int64_t intkey = 0;
               get_varint(&pg.data[celloff + 4], &intkey);

               //wxLogDebug("%*.*sinternal - right %lld celloff %d pagenum %d intkey %lld", (stack.size() - 1) * 2, (stack.size() - 1) * 2, " ", right, celloff, pagenum, intkey);

               // Stack the described page if we're not looking for a specific blockid
               // or if this page contains the given blockid.
               if (!blockid || blockid <= intkey)
               {
                  stack.push_back({pagenum, 0, 0});
                  stacked = true;
                  break;
               }
            }

            // If we pushed a new page onto the stack, we need to jump back up
            // to read the page
            if (stacked)
            {
               continue;
            }
         }

         // We've exhausted all the cells with this page, so we stack the right-most
         // leaf page.  Ensure we only process it once.
         if (right)
         {
            stack.push_back({right, 0, 0});
            right = 0;
            continue;
         }
      }
      // Process a leaf table b-tree page
      else if (pg.data[0] == 0x0d)
      {
         // Iterate over the cells
         //
         // If we're not looking for a specific blockid, then just accumulate the
         // payload sizes. We will be reading every leaf page in the sampleblocks
         // table.
         //
         // Otherwise we break out when we find the matching blockid. In this case,
         // we only ever look at 1 leaf page.
         bool stop = false;
         for (int i = 0; i < pg.numCells; i++)
         {
            // Get the offset to this cell using the offset in the cell pointer
            // array.
            //
            // The cell pointer array starts immediately after the page header
            // at offset 8 and the retrieved offset is from the beginning of
            // the page.
            int celloff = get2(&pg.data[8 + (i * 2)]);

            // Get the total payload size in bytes of the described row.
            int64_t payload = 0;
            int digits = get_varint(&pg.data[celloff], &payload);

            // Get the integer key for this row.
            int64_t intkey = 0;
            get_varint(&pg.data[celloff + digits], &intkey);

            //wxLogDebug("%*.*sleaf - celloff %4d intkey %lld payload %lld", (stack.size() - 1) * 2, (stack.size() - 1) * 2, " ", celloff, intkey, payload);

            // Add this payload size to the total if we're not looking for a specific
            // blockid
            if (!blockid)
            {
               total += payload;
            }
            // Otherwise, return the payload size for a matching row
            else if (blockid == intkey)
            {
               return payload;
            }
         }
      }

      // Done with the current branch, so pop back up to the previous one (if any)
      stack.pop_back();
   } while (!stack.empty());

   // Return the total used for all sample blocks
   return total;
}

// Retrieves a 2-byte big-endian integer from the page data
unsigned int ProjectFileIO::get2(const unsigned char *ptr)
{
   return (ptr[0] << 8) | ptr[1];
}

// Retrieves a 4-byte big-endian integer from the page data
unsigned int ProjectFileIO::get4(const unsigned char *ptr)
{
   return ((unsigned int) ptr[0] << 24) |
          ((unsigned int) ptr[1] << 16) |
          ((unsigned int) ptr[2] << 8)  |
          ((unsigned int) ptr[3]);
}

// Retrieves a variable length integer from the page data. Returns the
// number of digits used to encode the integer and the stores the
// value at the given location.
int ProjectFileIO::get_varint(const unsigned char *ptr, int64_t *out)
{
   int64_t val = 0;
   int i;

   for (i = 0; i < 8; ++i)
   {
      val = (val << 7) + (ptr[i] & 0x7f);
      if ((ptr[i] & 0x80) == 0)
      {
         *out = val;
         return i + 1;
      }
   }

   val = (val << 8) + (ptr[i] & 0xff);
   *out = val;

   return 9;
}

AutoCommitTransaction::AutoCommitTransaction(ProjectFileIO &projectFileIO,
                                             const char *name)
:  mIO(projectFileIO),
   mName(name)
{
   mInTrans = mIO.TransactionStart(mName);
   if ( !mInTrans )
      // To do, improve the message
      throw SimpleMessageBoxException( XO("Database error") );
}

AutoCommitTransaction::~AutoCommitTransaction()
{
   if (mInTrans)
   {
      if (!mIO.TransactionCommit(mName))
      {
         // Do not throw from a destructor!
         // This has to be a no-fail cleanup that does the best that it can.
      }
   }
}

bool AutoCommitTransaction::Rollback()
{
   if ( !mInTrans )
      // Misuse of this class
      THROW_INCONSISTENCY_EXCEPTION;

   mInTrans = !mIO.TransactionRollback(mName);

   return mInTrans;
}
