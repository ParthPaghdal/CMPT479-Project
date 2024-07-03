#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/core/streamset.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <kernel/io/stdout_kernel.h>
#include <idisa/idisa_sse_builder.h>
#include <idisa/idisa_avx_builder.h>

using namespace kernel;
using namespace llvm;
using namespace codegen;

static cl::OptionCategory ByteFilterByMaskOptions("Byte Filter By Mask Options", "Options for Byte Filter By Mask.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(ByteFilterByMaskOptions));

class ByteFilterByMaskKernel final : public MultiBlockKernel {
public:
    ByteFilterByMaskKernel(KernelBuilder &b, StreamSet *inputStream, StreamSet *maskStream, StreamSet *outputStream);
    static constexpr unsigned fw = 8; // Field width
protected:
    void generateMultiBlockLogic(KernelBuilder &b, llvm::Value * const numOfStrides) override;
};

constexpr unsigned ByteFilterByMaskKernel::fw;

ByteFilterByMaskKernel::ByteFilterByMaskKernel(KernelBuilder &b, StreamSet *inputStream, StreamSet *maskStream, StreamSet *outputStream)
    : MultiBlockKernel(b, "byte_filter_by_mask_kernel",
        {Binding{"inputStream", inputStream, FixedRate(1)}, Binding{"maskStream", maskStream, FixedRate(1)}},
        {Binding{"outputStream", outputStream, FixedRate(1)}}, {}, {}, {}) {}

void ByteFilterByMaskKernel::generateMultiBlockLogic(KernelBuilder &b, Value * const numOfStrides) {
    const unsigned inputPacksPerStride = fw;
    const unsigned outputPacksPerStride = fw;
    BasicBlock *entry = b.GetInsertBlock();
    BasicBlock *filterLoop = b.CreateBasicBlock("filterLoop");
    BasicBlock *filterFinalize = b.CreateBasicBlock("filterFinalize");
    Constant * const ZERO = b.getSize(0);

    Value *numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(std::log2(getStride() / b.getBitBlockWidth())));
    }

    b.CreateBr(filterLoop);
    b.SetInsertPoint(filterLoop);

    PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZERO, entry);

    Value *bytepack[inputPacksPerStride];
    Value *maskpack[inputPacksPerStride];
    for (unsigned i = 0; i < inputPacksPerStride; i++) {
        bytepack[i] = b.loadInputStreamPack("inputStream", ZERO, b.getInt32(i), blockOffsetPhi);
        maskpack[i] = b.loadInputStreamPack("maskStream", ZERO, b.getInt32(i), blockOffsetPhi);
    }

    auto &context = b.getContext();
    IDISA::IDISA_AVX512F_Builder idisaBuilder(context, 512, 8);

    Value *compressed[outputPacksPerStride];

    for(unsigned i = 0; i < inputPacksPerStride; i++){
        compressed[i] = idisaBuilder.mvmd_compress(fw, bytepack[i], maskpack[i]);
    }

    for(unsigned i = 0; i < 1; i++){
        b.storeOutputStreamPack("outputStream", ZERO, blockOffsetPhi, compressed[i]);
    }

    Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, filterLoop);
    Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
    b.CreateCondBr(moreToDo, filterLoop, filterFinalize);

    b.SetInsertPoint(filterFinalize);
}

typedef void (*ByteFilterByMaskFunctionType)(uint32_t fd);

ByteFilterByMaskFunctionType byteFilterByMaskGen(CPUDriver &driver) {
    auto &b = driver.getBuilder();
    auto P = driver.makePipeline({Binding{b.getInt32Ty(), "inputFileDescriptor"}}, {});

    Scalar *fileDescriptor = P->getInputScalar("inputFileDescriptor");

    StreamSet *inputStream = P->CreateStreamSet(1, 8);
    StreamSet *maskStream = P->CreateStreamSet(1, 8);
    StreamSet *outputStream = P->CreateStreamSet(1, 8);

    P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, inputStream);
    P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, maskStream);
    P->CreateKernelCall<ByteFilterByMaskKernel>(inputStream, maskStream, outputStream);
    P->CreateKernelCall<StdOutKernel>(outputStream);

    return reinterpret_cast<ByteFilterByMaskFunctionType>(P->compile());
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&ByteFilterByMaskOptions, codegen::codegen_flags()});
    CPUDriver pxDriver("bytefilterbymask");

    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
        return 1;
    } else {
        ByteFilterByMaskFunctionType func = nullptr;
        func = byteFilterByMaskGen(pxDriver);
        func(fd);
        close(fd);
    }
    return 0;
}