
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
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/scan/scanmatchgen.h>
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

static cl::OptionCategory ufFlags("Command Flags", "spread options");
static cl::opt<std::string> CC_expr(cl::Positional, cl::desc("<Unicode character class expression>"), cl::Required, cl::cat(ufFlags));
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"),  cl::cat(ufFlags));

static cl::opt<bool> UseDefaultspread("default-spread", cl::desc("Use the default byte spread by mask via S2P-spreadByMaks-P2S"), cl::cat(ufFlags));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P->captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P->captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P->captureByteData(#name, name)

class BytespreadByMaskKernel final : public MultiBlockKernel {
public:
    BytespreadByMaskKernel(KernelBuilder & b, StreamSet * const byteStream, StreamSet * const spread, StreamSet * const Packed);
protected:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

BytespreadByMaskKernel::BytespreadByMaskKernel(KernelBuilder & b, StreamSet * const byteStream, StreamSet * const spread, StreamSet * const Packed)
: MultiBlockKernel(b, "byte_spread_by_mask_kernel",
{Binding{"byteStream", byteStream, FixedRate(1)}, Binding{"spread", spread, FixedRate(1)}},
    {Binding{"output", Packed, PopcountOf("spread")}}, {}, {}, {}) {}

void BytespreadByMaskKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * packLoop = b.CreateBasicBlock("packLoop");
    BasicBlock * packFinalize = b.CreateBasicBlock("packFinalize");
    Constant * const ZERO = b.getSize(0);

    b.CreateBr(packLoop);
    b.SetInsertPoint(packLoop);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZERO, entry);

    Value * spreadVec = b.loadInputStreamBlock("spread", ZERO, blockOffsetPhi);

    VectorType * popVecTy = FixedVectorType::get(b.getIntNTy(b.getBitBlockWidth() / 8), 8);

    spreadVec = b.CreateBitCast(spreadVec, popVecTy);

    Value * toWritePos = b.getProducedItemCount("output");

    for (unsigned i = 0; i < 8; ++i) {
        Value * const spreadElem = b.CreateExtractElement(spreadVec, b.getInt32(i));

        Value * const elementPopCount = b.CreatePopcount(spreadElem);

        Value * const data = b.loadInputStreamPack("byteStream", ZERO, b.getInt32(i), blockOffsetPhi);
        
        Value * const expanded = b.mvmd_expand(8, data, spreadElem);

        // Debugging statements using llvm::outs()
        llvm::outs() << "spreadElem" << i << ": " << *spreadElem << "\n";
        llvm::outs() << "data" << i << ": " << *data << "\n";
        llvm::outs() << "expanded" << i << ": " << *expanded << "\n";

        Value * const toStorePtr = b.getRawOutputPointer("output", toWritePos);

        b.CreateAlignedStore(expanded, toStorePtr, 1);
        toWritePos = b.CreateAdd(toWritePos, elementPopCount);
    }

    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, packLoop);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfStrides);

    b.CreateCondBr(moreToDo, packLoop, packFinalize);
    b.SetInsertPoint(packFinalize);
}

typedef void (*spreadByMaskFunctionType)(uint32_t fd);

spreadByMaskFunctionType spreadbymask_gen (CPUDriver & pxDriver, re::Name * CC_name) {

    auto & b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline(
                {Binding{b.getInt32Ty(), "fileDescriptor"}});

    Scalar * const fileDescriptor = P->getInputScalar("fileDescriptor");

    //  Create a stream set consisting of a single stream of 8-bit units (bytes).
    StreamSet * const ByteStream = P->CreateStreamSet(1, 8);

    //  Read the file into the ByteStream.
    P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    SHOW_BYTES(ByteStream);

    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
    //  S2P stands for serial-to-parallel.
    StreamSet * BasisBits = P->CreateStreamSet(8);
    P->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);

    //  We need to know which input positions are LFs and which are not.
    //  The nonLF positions need to be expanded to generate two hex digits each.
    //  The Parabix CharacterClassKernelBuilder can create any desired stream for
    //  characters.   Note that the input is the set of byte values in the range
    //  [\x{00}-x{09}\x{0B}-\x{FF}] that is, all byte values except \x{0A}.
    //  For our example input "Wolf!\b", the nonLF stream is "11111."
    StreamSet * nonLF = P->CreateStreamSet(1);
    std::vector<re::CC *> nonLF_CC = {re::makeCC(re::makeByte(0,9), re::makeByte(0xB, 0xff))};
    P->CreateKernelCall<CharacterClassKernelBuilder>(nonLF_CC, BasisBits, nonLF);
    SHOW_STREAM(nonLF);

    //  We need to spread out the basis bits to make room for two positions for
    //  each non LF in the input.   The Parabix function UnitInsertionSpreadMask
    //  takes care of this using a mask of positions for insertion of one position.
    //  We insert one position for eacn nonLF character.    Given the
    //  nonLF stream "11111", the hexInsertMask is "1.1.1.1.1.1"
    StreamSet * insertMask = UnitInsertionSpreadMask(P, nonLF, InsertPosition::After);
    SHOW_STREAM(insertMask);

    // The parabix SpreadByMask function copies bits from an input stream
    // set to an output stream set, to positions marked by 1s in the first
    // argument (the spread mask).   Zeroes are inserted everywhere else.
    // This function performs STEP 1 in the comments above.
    StreamSet * spreadBasis = P->CreateStreamSet(8);
    SpreadByMask(P, insertMask, BasisBits, spreadBasis);
    SHOW_BIXNUM(spreadBasis);
    
    StreamSet * const spreadedBytes = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<BytespreadByMaskKernel>(ByteStream, insertMask, spreadedBytes);
    P->CreateKernelCall<StdOutKernel>(spreadedBytes);
    SHOW_BYTES(spreadedBytes);

    

    return reinterpret_cast<spreadByMaskFunctionType>(P->compile());
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&ufFlags, codegen::codegen_flags()});
    CPUDriver pxDriver("spread");

    spreadByMaskFunctionType fnPtr = nullptr;
    re::RE * CC_re = re::simplifyRE(re::RE_Parser::parse(CC_expr));
    CC_re = UCD::linkAndResolve(CC_re);
    CC_re = UCD::externalizeProperties(CC_re);
    if (re::Name * UCD_property_name = dyn_cast<re::Name>(CC_re)) {
        fnPtr = spreadbymask_gen(pxDriver, UCD_property_name);
    } else if (re::CC * CC_ast = dyn_cast<re::CC>(CC_re)) {
        fnPtr = spreadbymask_gen(pxDriver, makeName(CC_ast));
    } else {
        std::cerr << "Input expression must be a Unicode property or CC but found: " << CC_expr << " instead.\n";
        exit(1);
    }

    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        if (errno == EACCES) {
            std::cerr << "spread: " << inputFile << ": Permission denied.\n";
        }
        else if (errno == ENOENT) {
            std::cerr << "spread: " << inputFile << ": No such file.\n";
        }
        else {
            std::cerr << "spread: " << inputFile << ": Failed.\n";
        }
        exit(1);
    }
    struct stat sb;
    if (stat(inputFile.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        std::cerr << "spread: " << inputFile << ": Is a directory.\n";
        close(fd);
        exit(1);
    }

    fnPtr(fd);
    close(fd);
    return 0;
}
