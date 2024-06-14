/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>

namespace codegen {

class VirtualDriver {
public:

    virtual bool hasExternalFunction(const llvm::StringRef functionName) const = 0;

    virtual llvm::Function * addLinkFunction(llvm::Module * mod, llvm::StringRef name, llvm::FunctionType * type, void * functionPtr) const = 0;

};

}
