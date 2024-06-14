/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <pablo/pabloAST.h>
#include <llvm/ADT/StringMap.h>
#include <boost/container/flat_map.hpp>
#include <memory>

namespace IDISA { class IDISA_Builder; }
namespace pablo { class String; }
namespace pablo { class Integer; }
namespace llvm { class LLVMContext; }

namespace pablo {

class SymbolGenerator {
    friend class PabloKernel;
    using Allocator = PabloAST::Allocator;
public:
    using IntTy = uint64_t;
    String * makeString(const llvm::StringRef prefix) noexcept;
    Integer * getInteger(const IntTy value, unsigned intWidth = 64) noexcept;
    ~SymbolGenerator() { }
protected:
    SymbolGenerator(llvm::LLVMContext & C, Allocator & allocator)
    : mContext(C)
    , mAllocator(allocator) {

    }
private:
    llvm::LLVMContext &                          mContext;
    Allocator &                                  mAllocator;
    llvm::StringMap<IntTy>                       mPrefixMap;
    llvm::StringMap<String *>                    mStringMap;
    boost::container::flat_map<std::pair<IntTy, unsigned>, Integer *> mIntegerMap;
};


}

