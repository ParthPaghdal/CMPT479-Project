
#include <idisa/idisa_avx_builder.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/basis/p2s_kernel.h>
#include <llvm/IR/IRBuilder.h>
#include <kernel/core/idisa_target.h>
#include <boost/filesystem.hpp>
#include <grep/grep_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/adt/adt.h>
#include <re/parse/parser.h>
#include <re/transforms/re_simplifier.h>
#include <re/unicode/resolve_properties.h>
#include <re/cc/cc_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/core/streamset.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/pablo_kernel.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <grep/grep_kernel.h>
#include <toolchain/toolchain.h>
#include <fileselect/file_select.h>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <map>

using namespace kernel;
using namespace llvm;

static cl::OptionCategory ufFlags("Command Flags", "filter options");
static cl::opt<std::string> CC_expr(cl::Positional, cl::desc("<Unicode character class expression>"), cl::Required, cl::cat(ufFlags));
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"),  cl::cat(ufFlags));

static cl::opt<bool> UseDefaultFilter("default-filter", cl::desc("Use the default byte filter by mask via S2P-FilterByMaks-P2S"), cl::cat(ufFlags));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P->captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P->captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P->captureByteData(#name, name)

class ByteFilterByMaskKernel final : public MultiBlockKernel {
public:
    ByteFilterByMaskKernel(KernelBuilder & b, StreamSet * const byteStream, StreamSet * const filter, StreamSet * const Packed);
protected:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

ByteFilterByMaskKernel::ByteFilterByMaskKernel(KernelBuilder & b, StreamSet * const byteStream, StreamSet * const filter, StreamSet * const Packed)
: MultiBlockKernel(b, "byte_filter_by_mask_kernel",
{Binding{"byteStream", byteStream, FixedRate(1)}, Binding{"filter", filter, FixedRate(1)}},
    {Binding{"output", Packed, PopcountOf("filter")}}, {}, {}, {}) {}

void ByteFilterByMaskKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * packLoop = b.CreateBasicBlock("packLoop");
    BasicBlock * packFinalize = b.CreateBasicBlock("packFinalize");
    Constant * const ZERO = b.getSize(0);

    b.CreateBr(packLoop);
    b.SetInsertPoint(packLoop);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZERO, entry);

    Value * filterVec = b.loadInputStreamBlock("filter", ZERO, blockOffsetPhi);

    VectorType * popVecTy = FixedVectorType::get(b.getIntNTy(b.getBitBlockWidth() / 8), 8);

    filterVec = b.CreateBitCast(filterVec, popVecTy);

    Value * toWritePos = b.getProducedItemCount("output");

    for (unsigned i = 0; i < 8; ++i) {
        Value * const filterElem = b.CreateExtractElement(filterVec, b.getInt32(i));

        Value * const elementPopCount = b.CreatePopcount(filterElem);

        Value * const data = b.loadInputStreamPack("byteStream", ZERO, b.getInt32(i), blockOffsetPhi);

        Value * const compressed = b.mvmd_compress(8, data, filterElem);

        Value * const toStorePtr = b.getRawOutputPointer("output", toWritePos);

        b.CreateAlignedStore(compressed, toStorePtr, 1);
        toWritePos = b.CreateAdd(toWritePos, elementPopCount);
    }

    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, packLoop);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfStrides);

    b.CreateCondBr(moreToDo, packLoop, packFinalize);
    b.SetInsertPoint(packFinalize);
}

typedef void (*FilterByMaskFunctionType)(uint32_t fd);

FilterByMaskFunctionType filterbymask_gen (CPUDriver & pxDriver, re::Name * CC_name) {

    auto & b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline(
                {Binding{b.getInt32Ty(), "fileDescriptor"}});

    Scalar * const fileDescriptor = P->getInputScalar("fileDescriptor");

    //  Create a stream set consisting of a single stream of 8-bit units (bytes).
    StreamSet * const ByteStream = P->CreateStreamSet(1, 8);
    SHOW_BYTES(ByteStream);

    //  Read the file into the ByteStream.
    P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

    //  Create a set of 8 parallel streams of 1-bit units (bits).
    StreamSet * BasisBits = P->CreateStreamSet(8, 1);
    SHOW_BIXNUM(BasisBits);

    //  Transpose the ByteSteam into parallel bit stream form.
    P->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);

    //  Create a character class bit stream.
    StreamSet * CCmask = P->CreateStreamSet(1, 1);

    std::map<std::string, StreamSet *> propertyStreamMap;
    auto nameString = CC_name->getFullName();
    propertyStreamMap.emplace(nameString, CCmask);
    P->CreateKernelFamilyCall<UnicodePropertyKernelBuilder>(CC_name, BasisBits, CCmask);
    SHOW_STREAM(CCmask);

    StreamSet * u8index = P->CreateStreamSet(1, 1);
    P->CreateKernelCall<UTF8_index>(BasisBits, u8index);
    SHOW_STREAM(u8index);

    StreamSet * CCspans = P->CreateStreamSet(1, 1);
    P->CreateKernelCall<U8Spans>(CCmask, u8index, CCspans);
    SHOW_STREAM(CCspans);

    StreamSet * const FilteredBytes = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<ByteFilterByMaskKernel>(ByteStream, CCspans, FilteredBytes);
    SHOW_BYTES(FilteredBytes);

    P->CreateKernelCall<StdOutKernel>(FilteredBytes);

    return reinterpret_cast<FilterByMaskFunctionType>(P->compile());
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&ufFlags, codegen::codegen_flags()});
    CPUDriver pxDriver("filter");

    FilterByMaskFunctionType fnPtr = nullptr;
    re::RE * CC_re = re::simplifyRE(re::RE_Parser::parse(CC_expr));
    CC_re = UCD::linkAndResolve(CC_re);
    CC_re = UCD::externalizeProperties(CC_re);
    if (re::Name * UCD_property_name = dyn_cast<re::Name>(CC_re)) {
        fnPtr = filterbymask_gen(pxDriver, UCD_property_name);
    } else if (re::CC * CC_ast = dyn_cast<re::CC>(CC_re)) {
        fnPtr = filterbymask_gen(pxDriver, makeName(CC_ast));
    } else {
        std::cerr << "Input expression must be a Unicode property or CC but found: " << CC_expr << " instead.\n";
        exit(1);
    }

    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        if (errno == EACCES) {
            std::cerr << "filter: " << inputFile << ": Permission denied.\n";
        }
        else if (errno == ENOENT) {
            std::cerr << "filter: " << inputFile << ": No such file.\n";
        }
        else {
            std::cerr << "filter: " << inputFile << ": Failed.\n";
        }
        exit(1);
    }
    struct stat sb;
    if (stat(inputFile.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        std::cerr << "filter: " << inputFile << ": Is a directory.\n";
        close(fd);
        exit(1);
    }

    fnPtr(fd);
    close(fd);
    return 0;
}
