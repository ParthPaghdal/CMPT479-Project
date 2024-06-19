#i#ifndef IDISA_BUILDER_H
#define IDISABuilder_h

// Include necessary LLVM headers
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Module.h>

// Class definition for IDISABuilder
// her this this class is responsible for constructing LLVM IR (Intermediate Representation)
// for operations needed in the byte-oriented FilterByMask function
class IDISABuilder {
public:
    // Constructor that initializes the builder with a given LLVM context
    IDISABuilder(llvm::LLVMContext &context);

    // So here is Method to perform masked vector compression
    // Parameters:
    // fww stands fro  Field width
    //  a: Input vector
    //  select_mask: Mask vector to select elements
    // Returns: Compressed vector based on the mask
    llvm::Value *mvmd_compress(unsigned fw, llvm::Value *a, llvm::Value *select_mask);

    // Method to create the byte-oriented FilterByMask function
    // Returns: A pointer to the created function
    llvm::Function *createByteFilterByMaskFunction();

    // Method to get the LLVM module containing the function definitions
    // Returns: A reference to the module
    llvm::Module &getModule();

private:
    // Member variables
    llvm::LLVMContext &context;  // Reference to the LLVM context
    llvm::IRBuilder<> builder;   // IRBuilder to generate LLVM IR instructions
    llvm::Module module;         // Module to hold the function definitions
};

#endif // IDISA_BUILDER_H


