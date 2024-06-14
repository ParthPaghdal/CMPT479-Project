#include <pablo/pablo_illustratorpass.h>
#include <pablo/pe_illustrator.h>
#include <pablo/pablo_kernel.h>
#include <pablo/pabloAST.h>
#include <pablo/ps_assign.h>
#include <pablo/pe_var.h>
#include <pablo/branch.h>
#include <pablo/codegenstate.h>
#include <pablo/pablo_toolchain.h>
#include <boost/regex.hpp>

using namespace llvm;

namespace pablo {

void runIllustratorPass(PabloKernel * const kernel) {

    if (pablo::PabloIllustrateBitstreamRegEx.empty()) {
        return;
    }

    const boost::regex ex(pablo::PabloIllustrateBitstreamRegEx);

    SmallVector<char, 1024> tmp;

    std::function<void(PabloBlock *)> run = [&](PabloBlock * const scope) {
        Statement * stmt = scope->front();
        while (stmt) {
            Statement * const next = stmt->getNextNode();
            if (isa<Branch>(stmt)) {
                run(cast<Branch>(stmt)->getBody());
            } else if (LLVM_UNLIKELY(isa<Illustrate>(stmt))) {
                /* do nothing */
            } else {
                // TODO: should the string we compare the regex to also include the kernel name?
                const pablo::String * str = nullptr;
                PabloAST * value = nullptr;
                if (isa<Assign>(stmt)) {
                    const Var * var = cast<Assign>(stmt)->getVariable();
                    while (LLVM_UNLIKELY(isa<Extract>(var))) {
                        var = cast<Extract>(var)->getArray();
                    }
                    str = &var->getName();
                    value = cast<Assign>(stmt)->getValue();
                } else {
                    str = &stmt->getName();
                    value = stmt;
                }
                if (LLVM_UNLIKELY(boost::regex_match(str->str(), ex))) {
                    scope->setInsertPoint(stmt);
                    scope->createIllustrateBitstream(value, str);
                }
            }
            stmt = next;
        }
    };

    run(kernel->getEntryScope());

}

}
