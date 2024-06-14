#pragma once

/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <idisa/idisa_builder.h>

namespace IDISA {
    const unsigned I64_width = 64;

class IDISA_I64_Builder : public virtual IDISA_Builder {
public:
    static const unsigned NativeBitBlockWidth = I64_width;
  
    IDISA_I64_Builder(llvm::LLVMContext & C, unsigned bitBlockWidth, unsigned laneWidth)
    : IDISA_Builder(C, I64_width, bitBlockWidth, laneWidth) {

    } 

    virtual std::string getBuilderUniqueName() override;

    llvm::Value * hsimd_packh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packl(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    ~IDISA_I64_Builder() {}

};

}

