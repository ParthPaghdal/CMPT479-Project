#include "IDISABuilder.h"

// Constructor for IDISABuilder class, initializing LLVM context, IRBuilder, and a module
IDISABuilder::IDISABuilder(llvm::LLVMContext &context)
        : context(context), builder(context), module("ByteFilterByMaskModule", context) {}

// Method to perform masked vector compression
llvm::Value *IDISABuilder::mvmd_compress(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
    // Define the types for 8-bit integer and a vector of 64 8-bit integers
    llvm::Type *int8Ty = llvm::Type::getInt8Ty(context);
    llvm::Type *int8x64Ty = llvm::VectorType::get(int8Ty, 64);

    // Load the input and mask vectors
    llvm::Value *inputVec = builder.CreateLoad(int8x64Ty, a);
    llvm::Value *maskVec = builder.CreateLoad(int8x64Ty, select_mask);

    // Create a constant zero value of 8-bit integer type
    llvm::Value *zero = llvm::ConstantInt::get(int8Ty, 0);

    // Create a result mask by comparing the mask vector with zero (non-zero elements)
    llvm::Value *resultMask = builder.CreateICmpNE(maskVec, zero);

    // Use the AVX-512 intrinsic to compress the input vector based on the result mask
    llvm::Value *compressedVec = builder.CreateIntrinsic(llvm::Intrinsic::x86_avx512_mask_compress_epi8, {int8x64Ty}, {inputVec, resultMask});

    // Return the compressed vector
    return compressedVec;
}

// Method to create a function for byte-oriented filter by mask
llvm::Function *IDISABuilder::createByteFilterByMaskFunction() {
    // Define the types for 32-bit integer and pointer to 8-bit integer
    llvm::Type *int32Ty = llvm::Type::getInt32Ty(context);
    llvm::Type *int8PtrTy = llvm::Type::getInt8PtrTy(context);

    // Define the function type with input, mask, output pointers and size as arguments
    llvm::FunctionType *funcType = llvm::FunctionType::get(llvm::Type::getVoidTy(context), {int8PtrTy, int8PtrTy, int8PtrTy, int32Ty}, false);

    // Create the function and add it to the module
    llvm::Function *function = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, "byteFilterByMask", module);

    // Create an entry basic block and set the insertion point to it
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", function);
    builder.SetInsertPoint(entry);

    // Retrieve the function arguments
    auto args = function->arg_begin();
    llvm::Value *input = args++;
    llvm::Value *mask = args++;
    llvm::Value *output = args++;
    llvm::Value *size = args;

    // Apply the compression operation twice to filter the bytes
    llvm::Value *select_mask = mvmd_compress(8, input, mask);
    llvm::Value *compressed = mvmd_compress(8, input, select_mask);

    // Store the compressed result into the output
    builder.CreateStore(compressed, output);
    // Return from the function
    builder.CreateRetVoid();

    return function;
}

// Method to get the module containing the function definitions
llvm::Module &IDISABuilder::getModule() {
    return module;
}




