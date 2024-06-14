
/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include "../../tools/icgrep/grep_interface.h" // TODO: consider moving grep_interface to this library
#include <string>
#include <vector>
#include <sstream>
#include <atomic>
#include <set>
#include <boost/filesystem.hpp>
#include <re/analysis/capture-ref.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/parse/GLOB_parser.h>
#include <kernel/core/callback.h>
#include <kernel/util/linebreak_kernel.h>
#include <grep/grep_kernel.h>

namespace re { class CC; class Name; class RE; }
namespace llvm { namespace cl { class OptionCategory; } }
namespace kernel { class ProgramBuilder; }
namespace kernel { class StreamSet; }
namespace kernel { class ExternalStreamObject; }
class BaseDriver;

using ProgBuilderRef = const std::unique_ptr<kernel::ProgramBuilder> &;

namespace grep {

extern unsigned ByteCClimit;

enum class GrepRecordBreakKind {Null, LF, Unicode};

class InternalSearchEngine;
class InternalMultiSearchEngine;

enum GrepSignal : unsigned {BinaryFile};

class GrepCallBackObject : public kernel::SignallingObject {
public:
    GrepCallBackObject() : SignallingObject(), mBinaryFile(false) {}
    virtual ~GrepCallBackObject() {}
    virtual void handle_signal(unsigned signal);
    bool binaryFileSignalled() {return mBinaryFile;}
private:
    bool mBinaryFile;
};

class MatchAccumulator : public GrepCallBackObject {
public:
    MatchAccumulator() {}
    virtual ~MatchAccumulator() {}
    virtual void accumulate_match(const size_t lineNum, char * line_start, char * line_end) = 0;
    virtual void finalize_match(char * buffer_end) {}  // default: no op
    virtual unsigned getFileCount() {return 1;}  // default: return 1 for single file
    virtual size_t getFileStartPos(unsigned fileNo) {return 0;}
    virtual void setBatchLineNumber(unsigned fileNo, size_t batchLine) {}  // default: no op
};

extern "C" void accumulate_match_wrapper(intptr_t accum_addr, const size_t lineNum, char * line_start, char * line_end);

extern "C" void finalize_match_wrapper(intptr_t accum_addr, char * buffer_end);

extern "C" unsigned get_file_count_wrapper(intptr_t accum_addr);

extern "C" size_t get_file_start_pos_wrapper(intptr_t accum_addr, unsigned fileNo);

extern "C" void set_batch_line_number_wrapper(intptr_t accum_addr, unsigned fileNo, size_t batchLine);


class GrepEngine {
    enum class FileStatus {Pending, GrepComplete, PrintComplete};
    friend class InternalSearchEngine;
    friend class InternalMultiSearchEngine;
public:

    enum class EngineKind {QuietMode, MatchOnly, CountOnly, EmitMatches};

    GrepEngine(BaseDriver & driver);

    virtual ~GrepEngine() = 0;

    void setPreferMMap(bool b = true) {mPreferMMap = b;}

    void setColoring(bool b = true)  {mColoring = b;}
    void showFileNames(bool b = true) {mShowFileNames = b;}
    void setStdinLabel(std::string lbl) {mStdinLabel = lbl;}
    void showLineNumbers(bool b = true) {mShowLineNumbers = b;}
    void setContextLines(unsigned before, unsigned after) {mBeforeContext = before; mAfterContext = after;}
    void setInitialTab(bool b = true) {mInitialTab = b;}

    void setMaxCount(int m) {mMaxCount = m;}
    void setGrepStdIn(bool b = true) {mGrepStdIn = b;}
    void setInvertMatches(bool b = true) {mInvertMatches = b;}
    void setCaseInsensitive(bool b = true)  {mCaseInsensitive = b;}

    void suppressFileMessages(bool b = true) {mSuppressFileMessages = b;}
    void setBinaryFilesOption(argv::BinaryFilesMode mode) {mBinaryFilesMode = mode;}
    void setRecordBreak(GrepRecordBreakKind b);
    void initFileResult(const std::vector<boost::filesystem::path> & filenames);
    bool haveFileBatch();
    void initRE(re::RE * re);
    virtual void grepCodeGen();
    bool searchAllFiles();
    void * DoGrepThreadMethod();
    virtual void showResult(uint64_t grepResult, const std::string & fileName, std::ostringstream & strm);
    unsigned RunGrep(ProgBuilderRef P, const cc::Alphabet * a, re::RE * re, kernel::StreamSet * Matches);

protected:
    // Functional components that may be required for grep searches,
    // depending on search pattern, mode flags, external parameters and
    // implementation strategy.
    typedef uint32_t component_t;
    enum class Component : component_t {
        NoComponents = 0,
        S2P = 1,
        UTF8index = 2,
        MoveMatchesToEOL = 4,
        MatchSpans = 8,
        U21 = 64
    };
    bool hasComponent(Component compon_set, Component c);
    void setComponent(Component & compon_set, Component c);
    bool matchesToEOLrequired();

    // Transpose to basis bit streams, if required otherwise return the source byte stream.
    kernel::StreamSet * getBasis(ProgBuilderRef P, kernel::StreamSet * ByteStream);

    // Initial grep set-up.
    // Implement any required checking/processing of null characters, determine the
    // line break stream and the U8 index stream (if required).
    void grepPrologue(ProgBuilderRef P, kernel::StreamSet * SourceStream);
    // Prepare external property and GCB streams, if required.
    void prepareExternalStreams(ProgBuilderRef P, kernel::StreamSet * SourceStream);
    kernel::StreamSet * getMatchSpan(ProgBuilderRef P, re::RE * r, kernel::StreamSet * MatchResults);
    void addExternalStreams(ProgBuilderRef P, const cc::Alphabet * a, std::unique_ptr<kernel::GrepKernelOptions> & options, re::RE * regexp, kernel::StreamSet * indexMask = nullptr);
    kernel::StreamSet * initialMatches(ProgBuilderRef P, kernel::StreamSet * ByteStream);
    kernel::StreamSet * matchedLines(ProgBuilderRef P, kernel::StreamSet * ByteStream);
    kernel::StreamSet * grepPipeline(ProgBuilderRef P, kernel::StreamSet * ByteStream);
    virtual uint64_t doGrep(const std::vector<std::string> & fileNames, std::ostringstream & strm);
    int32_t openFile(const std::string & fileName, std::ostringstream & msgstrm);
    void applyColorization(const std::unique_ptr<kernel::ProgramBuilder> & E,
                                              kernel::StreamSet * SourceCoords,
                                              kernel::StreamSet * MatchSpans,
                                              kernel::StreamSet * Basis);
    std::string linePrefix(std::string fileName);

protected:

    EngineKind mEngineKind;
    bool mSuppressFileMessages;
    argv::BinaryFilesMode mBinaryFilesMode;
    bool mPreferMMap;
    bool mColoring;
    bool mShowFileNames;
    std::string mStdinLabel;
    bool mShowLineNumbers;
    unsigned mBeforeContext;
    unsigned mAfterContext;
    bool mInitialTab;
    bool mCaseInsensitive;
    bool mInvertMatches;
    int mMaxCount;
    bool mGrepStdIn;
    NullCharMode mNullMode;
    BaseDriver & mGrepDriver;
    void * mMainMethod;
    void * mBatchMethod;

    std::atomic<unsigned> mNextFileToGrep;
    std::atomic<unsigned> mNextFileToPrint;
    std::vector<boost::filesystem::path> mInputPaths;
    std::vector<std::vector<std::string>> mFileGroups;
    std::vector<std::ostringstream> mResultStrs;
    std::vector<FileStatus> mFileStatus;
    bool grepMatchFound;
    GrepRecordBreakKind mGrepRecordBreak;

    re:: RE * mRE;
    re::ReferenceInfo mRefInfo;
    std::string mFileSuffix;
    Component mExternalComponents;
    Component mInternalComponents;
    const cc::Alphabet * mIndexAlphabet;
    const cc::Alphabet * mLengthAlphabet;
    kernel::ExternalStreamTable mExternalTable;
    kernel::StreamSet * mLineBreakStream;
    kernel::StreamSet * mU8index;
    kernel::StreamSet * mU21;
    std::vector<std::string> mSpanNames;
    re::UTF8_Transformer mUTF8_Transformer;
    pthread_t mEngineThread;
};


//
// The EmitMatches engine uses an EmitMatchesAccumulator object to concatenate together
// matched lines.

class EmitMatch : public MatchAccumulator {
    friend class EmitMatchesEngine;
public:
    EmitMatch(bool showFileNames, bool showLineNumbers, bool showContext, bool initialTab)
        : mShowFileNames(showFileNames),
        mShowLineNumbers(showLineNumbers),
        mContextGroups(showContext),
        mInitialTab(initialTab),
        mCurrentFile(0),
        mLineCount(0),
        mLineNum(0),
        mTerminated(true) {}
    void prepareBatch (const std::vector<std::string> & fileNames);
    void accumulate_match(const size_t lineNum, char * line_start, char * line_end) override;
    void finalize_match(char * buffer_end) override;
    void setFileLabel(std::string fileLabel);
    void setStringStream(std::ostringstream * s);
    unsigned getFileCount() override;
    size_t getFileStartPos(unsigned fileNo) override;
    void setBatchLineNumber(unsigned fileNo, size_t batchLine) override;
protected:
    bool mShowFileNames;
    bool mShowLineNumbers;
    bool mContextGroups;
    bool mInitialTab;
    unsigned mCurrentFile;
    size_t mLineCount;
    size_t mLineNum;
    bool mTerminated;
    // An EmitMatch object may be defined to work with a single buffer for a
    // batch of files concatenated together.  The following vectors hold information
    // for each file in the batch, namely, its name, its starting code unit
    // position in the batch and its starting line number within the batch.
    std::vector<std::string> mFileNames;
    std::vector<size_t> mFileStartPositions;
    std::vector<size_t> mFileStartLineNumbers;
    std::string mLinePrefix;
    std::ostringstream * mResultStr;
    char * mBatchBuffer;
};

class EmitMatchesEngine final : public GrepEngine {
public:
    EmitMatchesEngine(BaseDriver & driver);
    void grepPipeline(ProgBuilderRef P, kernel::StreamSet * ByteStream);
    void grepCodeGen() override;
private:
    uint64_t doGrep(const std::vector<std::string> & fileNames, std::ostringstream & strm) override;
};

class CountOnlyEngine final : public GrepEngine {
public:
    CountOnlyEngine(BaseDriver & driver);
private:
    void showResult(uint64_t grepResult, const std::string & fileName, std::ostringstream & strm) override;
};

class MatchOnlyEngine final : public GrepEngine {
public:
    MatchOnlyEngine(BaseDriver & driver, bool showFilesWithoutMatch, bool useNullSeparators);
private:
    void showResult(uint64_t grepResult, const std::string & fileName, std::ostringstream & strm) override;
    unsigned mRequiredCount;
};

class QuietModeEngine final : public GrepEngine {
public:
    QuietModeEngine(BaseDriver & driver);
};



class InternalSearchEngine {
public:
    InternalSearchEngine(BaseDriver & driver);

    InternalSearchEngine(const std::unique_ptr<grep::GrepEngine> & engine);

    ~InternalSearchEngine();

    void setRecordBreak(GrepRecordBreakKind b) {mGrepRecordBreak = b;}
    void setCaseInsensitive()  {mCaseInsensitive = true;}

    void grepCodeGen(re::RE * matchingRE);

    void doGrep(const char * search_buffer, size_t bufferLength, MatchAccumulator & accum);

private:
    GrepRecordBreakKind mGrepRecordBreak;
    bool mCaseInsensitive;
    BaseDriver & mGrepDriver;
    void * mMainMethod;
};

enum class PatternKind {Include, Exclude};
class InternalMultiSearchEngine {
public:
    InternalMultiSearchEngine(BaseDriver & driver);

    InternalMultiSearchEngine(const std::unique_ptr<grep::GrepEngine> & engine);

    ~InternalMultiSearchEngine() {};

    void setRecordBreak(GrepRecordBreakKind b) {mGrepRecordBreak = b;}
    void setCaseInsensitive() {mCaseInsensitive = true;}

    void grepCodeGen(const re::PatternVector & patterns);

    void doGrep(const char * search_buffer, size_t bufferLength, MatchAccumulator & accum);

private:
    GrepRecordBreakKind mGrepRecordBreak;
    bool mCaseInsensitive;
    BaseDriver & mGrepDriver;
    void * mMainMethod;
};

/**
 * Returns which lines of a given buffer matches with a given regex pattern.
 *
 * @param pattern the regex pattern.
 * @param buffer the buffer to search for a match.
 * @param bufSize the size of the buffer.
 *
 * @return a vector with the lines that match the regex pattern.
 */
std::vector<uint64_t> lineNumGrep(re::RE * pattern, const char * buffer, size_t bufSize);

/**
 * Returns whether a given buffer matches with a given regex pattern.
 *
 * @param pattern the regex pattern.
 * @param buffer the buffer to search for a match.
 * @param bufSize the size of the buffer.
 *
 * @return true if there is any matches and false otherwise.
 */
bool matchOnlyGrep(re::RE * pattern, const char * buffer, size_t bufSize);

}

