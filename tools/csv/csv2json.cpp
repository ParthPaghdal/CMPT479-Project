/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */


#include <cstdio>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/util/linebreak_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/bitwise/bixnum_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/scan/scanmatchgen.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <string>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include "csv_util.hpp"
#ifdef ENABLE_PAPI
#include <util/papi_helper.hpp>
#endif

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P->captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P->captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P->captureByteData(#name, name)

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory CSV_Options("CSV Processing Options", "CSV Processing Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(CSV_Options));
static cl::opt<bool> HeaderSpecNamesFile("f", cl::desc("Interpret headers parameter as file name with header line"), cl::init(false), cl::cat(CSV_Options));
static cl::opt<std::string> HeaderSpec("headers", cl::desc("CSV column headers (explicit string or filename"), cl::init(""), cl::cat(CSV_Options));

static cl::opt<bool> TestDynamicRepeatingFile("dyn", cl::desc("Test Dynamic Repeating StreamSet"), cl::init(true), cl::cat(CSV_Options));
static cl::opt<bool> UseMergeByMaskKernel("merge-by-mask", cl::desc("Use MergeByMask kernel"), cl::init(false), cl::cat(CSV_Options));

typedef void (*CSVFunctionType)(uint32_t fd);

class Invert : public PabloKernel {
public:
    Invert(KernelBuilder & b, StreamSet * mask, StreamSet * inverted)
        : PabloKernel(b, "Invert",
                      {Binding{"mask", mask}},
                      {Binding{"inverted", inverted}}) {}
protected:
    void generatePabloMethod() override;
};

void Invert::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * mask = getInputStreamSet("mask")[0];
    PabloAST * inverted = pb.createInFile(pb.createNot(mask));
    Var * outVar = getOutputStreamVar("inverted");
    pb.createAssign(pb.createExtract(outVar, pb.getInteger(0)), inverted);
}

class BasisCombine : public PabloKernel {
public:
    BasisCombine(KernelBuilder & b, StreamSet * basis1, StreamSet * basis2, StreamSet * combined)
        : PabloKernel(b, "BasisCombine",
                      {Binding{"basis1", basis1}, Binding{"basis2", basis2}},
                      {Binding{"combined", combined}}) {}
protected:
    void generatePabloMethod() override;
};

void BasisCombine::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis1 = getInputStreamSet("basis1");
    std::vector<PabloAST *> basis2 = getInputStreamSet("basis2");
    Var * outVar = getOutputStreamVar("combined");
    for (unsigned i = 0; i < basis1.size(); i++) {
        PabloAST * combined = pb.createOr(basis1[i], basis2[i]);
        pb.createAssign(pb.createExtract(outVar, pb.getInteger(i)), combined);
    }
}

void MergeByMask01(const std::unique_ptr<ProgramBuilder> & P,
                 StreamSet * mask, StreamSet * a, StreamSet * b, StreamSet * merged) {
    unsigned elems = merged->getNumElements();
    if ((a->getNumElements() != elems) || (b->getNumElements() != elems)) {
        llvm::report_fatal_error("MergeByMask called with incompatible element counts");
    }
    StreamSet * expandedA = P->CreateStreamSet(elems);
    SpreadByMask(P, mask, a, expandedA);
    StreamSet * inverted = P->CreateStreamSet(1);
    P->CreateKernelCall<Invert>(mask, inverted);
    StreamSet * expandedB = P->CreateStreamSet(elems);
    SpreadByMask(P, inverted, b, expandedB);
    P->CreateKernelCall<BasisCombine>(expandedA, expandedB, merged);
}

CSVFunctionType generatePipeline(CPUDriver & pxDriver, const std::vector<std::string> & templateStrs) {
    // A Parabix program is build as a set of kernel calls called a pipeline.
    // A pipeline is construction using a Parabix driver object.
    auto & b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline({Binding{b.getInt32Ty(), "inputFileDecriptor"}}, {});
    //  The program will use a file descriptor as an input.
    Scalar * fileDescriptor = P->getInputScalar("inputFileDecriptor");
    // File data from mmap
    StreamSet * ByteStream = P->CreateStreamSet(1, 8);
    //  ReadSourceKernel is a Parabix Kernel that produces a stream of bytes
    //  from a file descriptor.
    P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
    //  S2P stands for serial-to-parallel.
    StreamSet * BasisBits = P->CreateStreamSet(8);
    Selected_S2P(P, ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
    //  We need to know which input positions are dquotes and which are not.
    StreamSet * csvCCs = P->CreateStreamSet(5);
    P->CreateKernelCall<CSVlexer>(BasisBits, csvCCs);
    StreamSet * recordSeparators = P->CreateStreamSet(1);
    StreamSet * fieldSeparators = P->CreateStreamSet(1);
    StreamSet * quoteEscape = P->CreateStreamSet(1);

    P->CreateKernelCall<CSVparser>(csvCCs, recordSeparators, fieldSeparators, quoteEscape);
    SHOW_STREAM(recordSeparators);
    SHOW_STREAM(fieldSeparators);
    SHOW_STREAM(quoteEscape);
    StreamSet * toKeep = P->CreateStreamSet(1);
    P->CreateKernelCall<CSVdataFieldMask>(csvCCs, recordSeparators, quoteEscape, toKeep, HeaderSpec == "");
    SHOW_STREAM(toKeep);
    //
    // Create a short stream which is 1-to-1 with the (field/record) separators,
    // having 0 bits for field separators and 1 bits for record separators.
    // Normally this will be a stream having exactly one bit set for every
    // N positions, where N is the number of entries per row.
    StreamSet * recordsByField = P->CreateStreamSet(1);
    FilterByMask(P, fieldSeparators, recordSeparators, recordsByField);
    SHOW_STREAM(recordsByField);

    StreamSet * translatedBasis = P->CreateStreamSet(8);
    P->CreateKernelCall<CSV_Char_Replacement>(quoteEscape, BasisBits, translatedBasis);
    SHOW_BIXNUM(translatedBasis);

    StreamSet * filteredBasis = P->CreateStreamSet(8);
    StreamSet * filteredFieldSeparators = P->CreateStreamSet(1);
    FilterByMask(P, toKeep, translatedBasis, filteredBasis);
    SHOW_BIXNUM(filteredBasis);

    FilterByMask(P, toKeep, fieldSeparators, filteredFieldSeparators);
    SHOW_STREAM(filteredFieldSeparators);

    std::vector<uint64_t> insertionAmts;
    unsigned maxInsertAmt = 0;
    for (auto & s : templateStrs) {
        unsigned insertAmt = s.size();
        insertionAmts.push_back(insertAmt);
        if (insertAmt > maxInsertAmt) maxInsertAmt = insertAmt;
    }
    const unsigned insertLengthBits = ceil_log2(maxInsertAmt+1);

    StreamSet * PrefixLgths = P->CreateRepeatingBixNum(insertLengthBits, insertionAmts, TestDynamicRepeatingFile);

    StreamSet * fieldStarts = P->CreateStreamSet(1);
    P->CreateKernelCall<LineStartsKernel>(filteredFieldSeparators, fieldStarts);
    SHOW_STREAM(fieldStarts);

    StreamSet * PrefixInsertBixNum = P->CreateStreamSet(insertLengthBits);
    SpreadByMask(P, fieldStarts, PrefixLgths, PrefixInsertBixNum);
    SHOW_BIXNUM(PrefixInsertBixNum);

    std::vector<uint64_t> fieldSuffixLgths;
    for (unsigned i = 0; i < templateStrs.size() - 1; i++) {
        // Insertion of a single quote to terminate each field.
        fieldSuffixLgths.push_back(1);
    }
    // Insertion of |"},| to terminate each record
    fieldSuffixLgths.push_back(3);

    const unsigned suffixLgthBits = 2;  // insert 1-3 characters.
    StreamSet * RepeatingSuffixLgths = P->CreateRepeatingBixNum(suffixLgthBits, fieldSuffixLgths, TestDynamicRepeatingFile);

    StreamSet * SuffixInsertBixNum = P->CreateStreamSet(suffixLgthBits);
    SpreadByMask(P, filteredFieldSeparators, RepeatingSuffixLgths, SuffixInsertBixNum);
    SHOW_BIXNUM(SuffixInsertBixNum);

    StreamSet * InsertBixNum = P->CreateStreamSet(insertLengthBits);
    P->CreateKernelCall<bixnum::Add>(PrefixInsertBixNum, SuffixInsertBixNum, InsertBixNum);
    SHOW_BIXNUM(InsertBixNum);

    StreamSet * const BasisSpreadMask = InsertionSpreadMask(P, InsertBixNum, InsertPosition::Before);
    SHOW_STREAM(BasisSpreadMask);
    
    std::vector<uint64_t> templateBytes;
    for (unsigned i = 0; i < templateStrs.size(); i++) {
        for (auto ch : templateStrs[i]) {
            templateBytes.push_back(static_cast<uint64_t>(ch));
        }
        templateBytes.push_back(static_cast<uint64_t>('"'));
        if (i == templateStrs.size() - 1) {
            templateBytes.push_back(static_cast<uint64_t>('}'));
            templateBytes.push_back(static_cast<uint64_t>(','));
        }
    }
    StreamSet * TemplateBasis = P->CreateRepeatingBixNum(8, templateBytes, TestDynamicRepeatingFile);

    StreamSet * FinalBasis = P->CreateStreamSet(8);
    if (UseMergeByMaskKernel) {
        MergeByMask(P, BasisSpreadMask, filteredBasis, TemplateBasis, FinalBasis);
    } else {
        MergeByMask01(P, BasisSpreadMask, filteredBasis, TemplateBasis, FinalBasis);
    }
    SHOW_BIXNUM(FinalBasis);
    StreamSet * Instantiated = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<P2SKernel>(FinalBasis, Instantiated);
    P->CreateKernelCall<StdOutKernel>(Instantiated);
    return reinterpret_cast<CSVFunctionType>(P->compile());
}

const unsigned MaxHeaderSize = 24;

int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&CSV_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});

    std::vector<std::string> headers;
    if (HeaderSpec == "") {
        headers = get_CSV_headers(inputFile);
    } else if (HeaderSpecNamesFile) {
        headers = get_CSV_headers(HeaderSpec);
    } else {
        headers = parse_CSV_headers(HeaderSpec);
    }
    for (auto & s : headers) {
        if (s.size() > MaxHeaderSize) {
            s = s.substr(0, MaxHeaderSize);
        }
    }
    const auto templateStrs = JSONfieldPrefixes(headers);
    //for (auto & s : templateStrs) {
    //    llvm::errs() << "template string: |" << s << "|\n";
    //}
    std::string templatePrologue = "[\n";
    std::string templateEpilogue = "\"}\n]\n";
    //  A CPU driver is capable of compiling and running Parabix programs on the CPU.
    CPUDriver driver("csv_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    CSVFunctionType fn = generatePipeline(driver, templateStrs);
    //  The compile function "fn"  can now be used.   It takes a file
    //  descriptor as an input, which is specified by the filename given by
    //  the inputFile command line option.]

    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        #ifdef REPORT_PAPI_TESTS
        papi::PapiCounter<4> jitExecution{{PAPI_L3_TCM, PAPI_L3_TCA, PAPI_TOT_INS, PAPI_TOT_CYC}};
        // papi::PapiCounter<3> jitExecution{{PAPI_FUL_ICY, PAPI_STL_CCY, PAPI_RES_STL}};
        jitExecution.start();
        #endif
        //  Run the pipeline.
        printf("%s", templatePrologue.c_str());
        fflush(stdout);
        fn(fd);
        close(fd);
        printf("%s", templateEpilogue.c_str());
        #ifdef REPORT_PAPI_TESTS
        jitExecution.stop();
        jitExecution.write(std::cerr);
        #endif
    }
    return 0;
}
