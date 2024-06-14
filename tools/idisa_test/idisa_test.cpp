/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <vector>
#include <string>
#include <toolchain/toolchain.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/idisa_target.h>
#include <kernel/core/streamset.h>
#include <kernel/io/source_kernel.h>
#include <kernel/util/hex_convert.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

using namespace llvm;
using namespace kernel;

static cl::OptionCategory testFlags("Command Flags", "test options");

static cl::opt<std::string> TestOperation(cl::Positional, cl::desc("Operation to test"), cl::Required, cl::cat(testFlags));

static cl::opt<int> TestFieldWidth(cl::Positional, cl::desc("Test field width (default 64)."), cl::init(64), cl::Required, cl::cat(testFlags));

static cl::opt<std::string> Operand1TestFile(cl::Positional, cl::desc("Operand 1 data file."), cl::Required, cl::cat(testFlags));
static cl::opt<std::string> Operand2TestFile(cl::Positional, cl::desc("Operand 2 data file."), cl::Required, cl::cat(testFlags));
static cl::opt<std::string> TestOutputFile("o", cl::desc("Test output file."), cl::cat(testFlags));
static cl::opt<bool> QuietMode("q", cl::desc("Suppress output, set the return code only."), cl::cat(testFlags));
static cl::opt<int> ShiftMask("ShiftMask", cl::desc("Mask applied to the shift operand (2nd operand) of simd_sllv, srlv, srav, rotl, rotr"), cl::init(0));
static cl::opt<int> Immediate("i", cl::desc("Immediate value for mvmd_dslli"), cl::init(1));

class ShiftMaskKernel : public BlockOrientedKernel {
public:
    ShiftMaskKernel(KernelBuilder & b, unsigned fw, unsigned limit, StreamSet * input, StreamSet * output);
protected:
    void generateDoBlockMethod(KernelBuilder & kb) override;
private:
    const unsigned mTestFw;
    const unsigned mShiftMask;
};

ShiftMaskKernel::ShiftMaskKernel(KernelBuilder & b, unsigned fw, unsigned mask, StreamSet *input, StreamSet *output)
: BlockOrientedKernel(b, "shiftMask" + std::to_string(fw) + "_" + std::to_string(mask),
                              {Binding{"shiftOperand", input}},
                              {Binding{"limitedShift", output}},
                              {}, {}, {}),
mTestFw(fw), mShiftMask(mask) {}

void ShiftMaskKernel::generateDoBlockMethod(KernelBuilder & b) {
    Type * fwTy = b.getIntNTy(mTestFw);
    Constant * const ZeroConst = b.getSize(0);
    Value * shiftOperand = b.loadInputStreamBlock("shiftOperand", ZeroConst);
    unsigned fieldCount = b.getBitBlockWidth()/mTestFw;
    Value * masked = b.simd_and(shiftOperand, b.getSplat(fieldCount, ConstantInt::get(fwTy, mShiftMask)));
    b.storeOutputStreamBlock("limitedShift", ZeroConst, masked);
}

class IdisaBinaryOpTestKernel : public MultiBlockKernel {
public:
    IdisaBinaryOpTestKernel(KernelBuilder & b, std::string idisa_op, unsigned fw, unsigned imm,
                            StreamSet * Operand1, StreamSet * Operand2, StreamSet * result);
protected:
    void generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfStrides) override;
private:
    const std::string mIdisaOperation;
    const unsigned mTestFw;
    const unsigned mImmediateShift;
};

IdisaBinaryOpTestKernel::IdisaBinaryOpTestKernel(KernelBuilder & b, std::string idisa_op, unsigned fw, unsigned imm,
                                                 StreamSet *Operand1, StreamSet *Operand2, StreamSet *result)
: MultiBlockKernel(b, idisa_op + std::to_string(fw) + "_test",
     {Binding{"operand1", Operand1}, Binding{"operand2", Operand2}},
     {Binding{"result", result}},
     {}, {}, {}),
mIdisaOperation(std::move(idisa_op)), mTestFw(fw), mImmediateShift(imm) {}

void IdisaBinaryOpTestKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfBlocks) {
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * processBlock = b.CreateBasicBlock("processBlock");
    BasicBlock * done = b.CreateBasicBlock("done");
    Constant * const ZeroConst = b.getSize(0);
    b.CreateBr(processBlock);
    b.SetInsertPoint(processBlock);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZeroConst, entry);
    Value * operand1 = b.loadInputStreamBlock("operand1", ZeroConst, blockOffsetPhi);
    Value * operand2 = b.loadInputStreamBlock("operand2", ZeroConst, blockOffsetPhi);
    Value * result = nullptr;
    if (mIdisaOperation == "simd_add") {
        result = b.simd_add(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_sub") {
        result = b.simd_sub(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_mult") {
        result = b.simd_mult(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_eq") {
        result = b.simd_eq(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_ne") {
        result = b.simd_ne(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_gt") {
        result = b.simd_gt(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_ugt") {
        result = b.simd_ugt(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_ge") {
        result = b.simd_ge(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_uge") {
        result = b.simd_uge(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_lt") {
        result = b.simd_lt(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_le") {
        result = b.simd_le(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_ult") {
        result = b.simd_ult(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_ule") {
        result = b.simd_ule(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_max") {
        result = b.simd_max(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_min") {
        result = b.simd_min(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_umax") {
        result = b.simd_umax(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_umin") {
        result = b.simd_umin(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_sllv") {
        result = b.simd_sllv(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_srlv") {
        result = b.simd_srlv(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_rotl") {
        result = b.simd_rotl(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_rotr") {
        result = b.simd_rotr(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_pext") {
        result = b.simd_pext(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "simd_pdep") {
        result = b.simd_pdep(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "hsimd_packh") {
        result = b.hsimd_packh(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "hsimd_packl") {
        result = b.hsimd_packl(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "hsimd_packus") {
        result = b.hsimd_packus(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "hsimd_packss") {
        result = b.hsimd_packss(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "esimd_mergeh") {
        result = b.esimd_mergeh(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "esimd_mergel") {
        result = b.esimd_mergel(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "mvmd_shuffle") {
        result = b.mvmd_shuffle(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "mvmd_compress") {
        result = b.mvmd_compress(mTestFw, operand1, operand2);
    } else if (mIdisaOperation == "mvmd_dslli") {
        result = b.mvmd_dslli(mTestFw, operand1, operand2, mImmediateShift);
    } else {
        llvm::report_fatal_error(llvm::StringRef("Binary operation ") + mIdisaOperation + " is unknown to the IdisaBinaryOpTestKernel kernel.");
    }
    b.storeOutputStreamBlock("result", ZeroConst, blockOffsetPhi, b.bitCast(result));
    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, processBlock);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
    b.CreateCondBr(moreToDo, processBlock, done);
    b.SetInsertPoint(done);
}

class IdisaBinaryOpCheckKernel : public BlockOrientedKernel {
public:
    IdisaBinaryOpCheckKernel(KernelBuilder & b, std::string idisa_op, unsigned fw, unsigned imm,
                             StreamSet * Operand1, StreamSet * Operand2, StreamSet * result,
                             StreamSet * expected, Scalar * failures);
protected:
    void generateDoBlockMethod(KernelBuilder & kb) override;
private:
    const std::string mIdisaOperation;
    const unsigned mTestFw;
    const unsigned mImmediateShift;
};

IdisaBinaryOpCheckKernel::IdisaBinaryOpCheckKernel(KernelBuilder & b, std::string idisa_op, unsigned fw, unsigned imm,
                                                   StreamSet *Operand1, StreamSet *Operand2, StreamSet *result,
                                                   StreamSet *expected, Scalar *failures)
: BlockOrientedKernel(b, idisa_op + std::to_string(fw) + "_check" + std::to_string(QuietMode),
                           {Binding{"operand1", Operand1},
                            Binding{"operand2", Operand2},
                            Binding{"test_result", result}},
                           {Binding{"expected_result", expected}},
                           {}, {Binding{"totalFailures", failures}}, {}),
mIdisaOperation(idisa_op), mTestFw(fw), mImmediateShift(imm) {}

void IdisaBinaryOpCheckKernel::generateDoBlockMethod(KernelBuilder & b) {
    Type * fwTy = b.getIntNTy(mTestFw);
    BasicBlock * reportFailure = b.CreateBasicBlock("reportFailure");
    BasicBlock * continueTest = b.CreateBasicBlock("continueTest");
    Constant * const ZeroConst = b.getSize(0);
    Value * operand1Block = b.loadInputStreamBlock("operand1", ZeroConst);
    Value * operand2Block = b.loadInputStreamBlock("operand2", ZeroConst);
    Value * resultBlock = b.loadInputStreamBlock("test_result", ZeroConst);
    unsigned fieldCount = b.getBitBlockWidth()/mTestFw;
    Value * expectedBlock = b.allZeroes();
    if (mIdisaOperation == "mvmd_shuffle") {
        for (unsigned i = 0; i < fieldCount; i++) {
            Value * idx = b.CreateURem(b.mvmd_extract(mTestFw, operand2Block, i), ConstantInt::get(fwTy, fieldCount));
            Value * elt = b.CreateExtractElement(b.fwCast(mTestFw, operand1Block), b.CreateZExtOrTrunc(idx, b.getInt32Ty()));
            expectedBlock = b.mvmd_insert(mTestFw, expectedBlock, elt, i);
        }
    } else if (mIdisaOperation == "mvmd_dslli") {
        for (unsigned i = 0; i < fieldCount; i++) {
            Value * elt = nullptr;
            if (i < mImmediateShift) elt = b.mvmd_extract(mTestFw, operand2Block, fieldCount - mImmediateShift + i);
            else elt = b.mvmd_extract(mTestFw, operand1Block, i - mImmediateShift);
            expectedBlock = b.mvmd_insert(mTestFw, expectedBlock, elt, i);
        }
    } else {
        for (unsigned i = 0; i < fieldCount; i++) {
            Value * operand1 = b.mvmd_extract(mTestFw, operand1Block, i);
            Value * operand2 = b.mvmd_extract(mTestFw, operand2Block, i);
            Value * expected = nullptr;
            if (mIdisaOperation.substr(0,5) == "simd_") {
                if (mIdisaOperation == "simd_add") {
                    expected = b.CreateAdd(operand1, operand2);
                } else if (mIdisaOperation == "simd_sub") {
                    expected = b.CreateSub(operand1, operand2);
                } else if (mIdisaOperation == "simd_mult") {
                    expected = b.CreateMul(operand1, operand2);
                } else if (mIdisaOperation == "simd_eq") {
                    expected = b.CreateSExt(b.CreateICmpEQ(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_ne") {
                    expected = b.CreateSExt(b.CreateICmpNE(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_gt") {
                    expected = b.CreateSExt(b.CreateICmpSGT(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_ge") {
                    expected = b.CreateSExt(b.CreateICmpSGE(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_ugt") {
                    expected = b.CreateSExt(b.CreateICmpUGT(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_uge") {
                    expected = b.CreateSExt(b.CreateICmpUGE(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_lt") {
                    expected = b.CreateSExt(b.CreateICmpSLT(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_le") {
                    expected = b.CreateSExt(b.CreateICmpSLE(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_ult") {
                    expected = b.CreateSExt(b.CreateICmpULT(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_ule") {
                    expected = b.CreateSExt(b.CreateICmpULE(operand1, operand2), fwTy);
                } else if (mIdisaOperation == "simd_max") {
                    expected = b.CreateSelect(b.CreateICmpSGT(operand1, operand2), operand1, operand2);
                } else if (mIdisaOperation == "simd_min") {
                    expected = b.CreateSelect(b.CreateICmpSLT(operand1, operand2), operand1, operand2);
                } else if (mIdisaOperation == "simd_umax") {
                    expected = b.CreateSelect(b.CreateICmpUGT(operand1, operand2), operand1, operand2);
                } else if (mIdisaOperation == "simd_umin") {
                    expected = b.CreateSelect(b.CreateICmpULT(operand1, operand2), operand1, operand2);
                } else if (mIdisaOperation == "simd_sllv") {
                    expected = b.CreateShl(operand1, operand2);
                } else if (mIdisaOperation == "simd_srlv") {
                    expected = b.CreateLShr(operand1, operand2);
                } else if (mIdisaOperation == "simd_pext") {
                    Constant * zeroConst = ConstantInt::getNullValue(fwTy);
                    Constant * oneConst = ConstantInt::get(fwTy, 1);
                    expected = zeroConst;
                    Value * out_bit = oneConst;
                    for (unsigned i = 0; i < mTestFw; i++) {
                        Value * i_bit = Constant::getIntegerValue(fwTy, APInt::getOneBitSet(mTestFw, i));
                        Value * operand_i_isSet = b.CreateICmpEQ(b.CreateAnd(operand1, i_bit), i_bit);
                        Value * mask_i_isSet = b.CreateICmpEQ(b.CreateAnd(operand2, i_bit), i_bit);
                        expected = b.CreateSelect(b.CreateAnd(operand_i_isSet, mask_i_isSet), b.CreateOr(expected, out_bit), expected);
                        out_bit = b.CreateSelect(mask_i_isSet, b.CreateAdd(out_bit, out_bit), out_bit);
                    }
                } else if (mIdisaOperation == "simd_rotl") {
                    Constant * fwConst = ConstantInt::get(fwTy, mTestFw);
                    Constant * fwMaskConst = ConstantInt::get(fwTy, mTestFw - 1);
                    Value * shl = b.CreateShl(operand1, b.CreateAnd(operand2, fwMaskConst));
                    Value * shr = b.CreateLShr(operand1, b.CreateAnd(b.CreateSub(fwConst, operand2), fwMaskConst));
                    expected = b.CreateOr(shl, shr);
                } else if (mIdisaOperation == "simd_rotr") {
                    Constant * fwConst = ConstantInt::get(fwTy, mTestFw);
                    Constant * fwMaskConst = ConstantInt::get(fwTy, mTestFw - 1);
                    Value * shl = b.CreateShl(operand1, b.CreateAnd(b.CreateSub(fwConst, operand2), fwMaskConst));
                    Value * shr = b.CreateLShr(operand1, b.CreateAnd(operand2, fwMaskConst));
                    expected = b.CreateOr(shl, shr);
                } else if (mIdisaOperation == "simd_pdep") {
                    Constant * zeroConst = ConstantInt::getNullValue(fwTy);
                    Constant * oneConst = ConstantInt::get(fwTy, 1);
                    expected = zeroConst;
                    Value * shft = zeroConst;
                    Value * select_bit = oneConst;
                    for (unsigned i = 0; i < mTestFw; i++) {
                        expected = b.CreateOr(b.CreateAnd(operand2, b.CreateShl(b.CreateAnd(operand1, select_bit), shft)), expected);
                        Value * i_bit = Constant::getIntegerValue(fwTy, APInt::getOneBitSet(mTestFw, i));
                        Value * mask_i_isSet = b.CreateICmpEQ(b.CreateAnd(operand2, i_bit), i_bit);
                        select_bit = b.CreateSelect(mask_i_isSet, b.CreateAdd(select_bit, select_bit), select_bit);
                        shft = b.CreateSelect(mask_i_isSet, shft, b.CreateAdd(shft, oneConst));
                    }
                } else {
                    llvm::report_fatal_error(llvm::StringRef("Unknown SIMD vertical operation: ") + mIdisaOperation);
                }
                expectedBlock = b.bitCast(b.mvmd_insert(mTestFw, expectedBlock, expected, i));
            } else if (mIdisaOperation == "hsimd_packh") {
                operand1 = b.CreateTrunc(b.CreateLShr(operand1, mTestFw/2), b.getIntNTy(mTestFw/2));
                operand2 = b.CreateTrunc(b.CreateLShr(operand2, mTestFw/2), b.getIntNTy(mTestFw/2));
                expectedBlock = b.mvmd_insert(mTestFw/2, expectedBlock, operand1, i);
                expectedBlock = b.bitCast(b.mvmd_insert(mTestFw/2, expectedBlock, operand2, fieldCount + i));
            } else if (mIdisaOperation == "hsimd_packl") {
                operand1 = b.CreateTrunc(operand1, b.getIntNTy(mTestFw/2));
                operand2 = b.CreateTrunc(operand2, b.getIntNTy(mTestFw/2));
                expectedBlock = b.mvmd_insert(mTestFw/2, expectedBlock, operand1, i);
                expectedBlock = b.bitCast(b.mvmd_insert(mTestFw/2, expectedBlock, operand2, fieldCount + i));
            } else if (mIdisaOperation == "hsimd_packus") {
                Value * zeroes = ConstantInt::getNullValue(operand1->getType());
                operand1 = b.CreateSelect(b.CreateICmpSLT(operand1, zeroes), zeroes, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSLT(operand2, zeroes), zeroes, operand2);
                Value * testVal = ConstantInt::get(b.getContext(), APInt::getLowBitsSet(mTestFw, mTestFw/2));
                operand1 = b.CreateSelect(b.CreateICmpSGT(operand1, testVal), testVal, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSGT(operand2, testVal), testVal, operand2);
                expectedBlock = b.mvmd_insert(mTestFw/2, expectedBlock, operand1, i);
                expectedBlock = b.bitCast(b.mvmd_insert(mTestFw/2, expectedBlock, operand2, fieldCount + i));
            } else if (mIdisaOperation == "hsimd_packss") {
                Value * testVal = ConstantInt::get(b.getIntNTy(mTestFw), (1 << (mTestFw/2 - 1)) - 1);
                operand1 = b.CreateSelect(b.CreateICmpSGT(operand1, testVal), testVal, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSGT(operand2, testVal), testVal, operand2);
                testVal = b.CreateNot(testVal);
                operand1 = b.CreateSelect(b.CreateICmpSLT(operand1, testVal), testVal, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSLT(operand2, testVal), testVal, operand2);
                expectedBlock = b.mvmd_insert(mTestFw/2, expectedBlock, operand1, i);
                expectedBlock = b.bitCast(b.mvmd_insert(mTestFw/2, expectedBlock, operand2, fieldCount + i));
            } else if (mIdisaOperation == "esimd_mergeh") {
                if (i >= fieldCount/2) {
                    expectedBlock = b.mvmd_insert(mTestFw, expectedBlock, operand1, 2*(i - fieldCount/2));
                    expectedBlock = b.bitCast(b.mvmd_insert(mTestFw, expectedBlock, operand2, 2*(i - fieldCount/2) + 1));
                }
            } else if (mIdisaOperation == "esimd_mergel") {
                if (i < fieldCount/2) {
                    expectedBlock = b.mvmd_insert(mTestFw, expectedBlock, operand1, 2*i);
                    expectedBlock = b.bitCast(b.mvmd_insert(mTestFw, expectedBlock, operand2, 2*i + 1));
                }
            }
        }
    }
    b.storeOutputStreamBlock("expected_result", ZeroConst, expectedBlock);
    Value * failures = b.simd_ugt(mTestFw, b.simd_xor(resultBlock, expectedBlock), b.allZeroes());
    Value * anyFailure = b.bitblock_any(failures);
    Value * failure_count = b.CreateUDiv(b.bitblock_popcount(failures), b.getSize(mTestFw));
    b.setScalarField("totalFailures", b.CreateAdd(b.getScalarField("totalFailures"), failure_count));
    if (!QuietMode) {
        b.CreateCondBr(anyFailure, reportFailure, continueTest);
        b.SetInsertPoint(reportFailure);
        b.CallPrintRegister("operand1", b.bitCast(operand1Block));
        b.CallPrintRegister("operand2", b.bitCast(operand2Block));
        b.CallPrintRegister(mIdisaOperation + "(" + std::to_string(mTestFw) + ", operand1, operand2)", resultBlock);
        b.CallPrintRegister("expecting", expectedBlock);
        b.CreateBr(continueTest);
        b.SetInsertPoint(continueTest);
    }
}

// Open a file and return its file desciptor.
int32_t openFile(const std::string & fileName, llvm::raw_ostream & msgstrm) {
    if (fileName == "-") {
        return STDIN_FILENO;
    }
    else {
        struct stat sb;
        int32_t fileDescriptor = open(fileName.c_str(), O_RDONLY);
        if (LLVM_UNLIKELY(fileDescriptor == -1)) {
            if (errno == EACCES) {
                msgstrm << "idisa_test: " << fileName << ": Permission denied.\n";
            }
            else if (errno == ENOENT) {
                msgstrm << "idisa_test: " << fileName << ": No such file.\n";
            }
            else {
                msgstrm << "idisa_test: " << fileName << ": Failed.\n";
            }
            return fileDescriptor;
        }
        if (stat(fileName.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
            msgstrm << "idisa_test: " << fileName << ": Is a directory.\n";
            close(fileDescriptor);
            return -1;
        }
        return fileDescriptor;
    }
}

typedef size_t (*IDISAtestFunctionType)(int32_t fd1, int32_t fd2);

StreamSet * readHexToBinary(std::unique_ptr<ProgramBuilder> & P, const std::string & fd) {
    StreamSet * const hexStream = P->CreateStreamSet(1, 8);
    Scalar * const fileDecriptor = P->getInputScalar(fd);
    P->CreateKernelCall<ReadSourceKernel>(fileDecriptor, hexStream);
    StreamSet * const bitStream = P->CreateStreamSet(1, 1);
    P->CreateKernelCall<HexToBinary>(hexStream, bitStream);
    return bitStream;
}

inline StreamSet * applyShiftMask(std::unique_ptr<ProgramBuilder> & P, StreamSet * input) {
    if (ShiftMask > 0) {
        StreamSet * output = P->CreateStreamSet(1, 1);
        P->CreateKernelCall<ShiftMaskKernel>(TestFieldWidth, ShiftMask, input, output);
        return output;
    }
    return input;
}

IDISAtestFunctionType pipelineGen(CPUDriver & pxDriver) {

    auto & b = pxDriver.getBuilder();

    Type * const sizeTy = b.getSizeTy();
    Type * const int32Ty = b.getInt32Ty();

    Bindings inputs;
    inputs.emplace_back(int32Ty, "operand1FileDecriptor");
    inputs.emplace_back(int32Ty, "operand2FileDecriptor");
    if (!TestOutputFile.empty()) {
        inputs.emplace_back(b.getInt8PtrTy(), "outputFileName");
    }

    auto P = pxDriver.makePipeline(std::move(inputs), {Binding{sizeTy, "totalFailures"}});


    StreamSet * Operand1BitStream = readHexToBinary(P, "operand1FileDecriptor");
    StreamSet * Operand2BitStream = applyShiftMask(P, readHexToBinary(P, "operand2FileDecriptor"));

    StreamSet * ResultBitStream = P->CreateStreamSet(1, 1);

    P->CreateKernelCall<IdisaBinaryOpTestKernel>(TestOperation, TestFieldWidth, Immediate
                                                 , Operand1BitStream, Operand2BitStream
                                                 , ResultBitStream);

    StreamSet * ExpectedResultBitStream = P->CreateStreamSet(1, 1);

    P->CreateKernelCall<IdisaBinaryOpCheckKernel>(TestOperation, TestFieldWidth, Immediate
                                                 , Operand1BitStream, Operand2BitStream, ResultBitStream
                                                 , ExpectedResultBitStream, P->getOutputScalar("totalFailures"));

    if (!TestOutputFile.empty()) {
        StreamSet * ResultHexStream = P->CreateStreamSet(1, 8);
        P->CreateKernelCall<BinaryToHex>(ResultBitStream, ResultHexStream);
        Scalar * outputFileName = P->getInputScalar("outputFileName");
        P->CreateKernelCall<FileSink>(outputFileName, ResultHexStream);
    }

    return reinterpret_cast<IDISAtestFunctionType>(P->compile());
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&testFlags, codegen::codegen_flags()});
    CPUDriver pxDriver("idisa_test");
    if (ShiftMask == 0) {
        ShiftMask = TestFieldWidth - 1;
    }
    auto idisaTestFunction = pipelineGen(pxDriver);

    const int32_t fd1 = openFile(Operand1TestFile, llvm::outs());
    const int32_t fd2 = openFile(Operand2TestFile, llvm::outs());
    const size_t failure_count = idisaTestFunction(fd1, fd2);
    if (!QuietMode) {
        if (failure_count == 0) {
            llvm::outs() << "Test success: " << TestOperation << "<" << TestFieldWidth << ">\n";
        } else {
            llvm::outs() << "Test failure: " << TestOperation << "<" << TestFieldWidth << "> failed " << failure_count << " tests!\n";
        }
    }
    close(fd1);
    close(fd2);
    return failure_count > 0;
}
