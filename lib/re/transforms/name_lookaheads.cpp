/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/transforms/name_lookaheads.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <re/adt/adt.h>
#include <re/transforms/re_transformer.h>

using namespace llvm;

namespace re {

class LookAheadNamer final : public RE_Transformer {
public:
    LookAheadNamer() : RE_Transformer("LookAheadNamer") {}
    RE * transformAssertion (Assertion * a) override;
private:
};

RE * LookAheadNamer::transformAssertion (Assertion * a) {
    RE * x0 = a->getAsserted();
    RE * x = transform(x0);
    if ((a->getKind() == Assertion::Kind::LookAhead) && !isa<Name>(x)) {
        std::string name = Printer_RE::PrintRE(x);
        return makeAssertion(makeName(name, x), Assertion::Kind::LookAhead, a->getSense());
    } else if (x == x0) {
        return a;
    } else {
        return makeAssertion(x, a->getKind(), a->getSense());
    }
}

RE * name_lookaheads(RE * re) {
    return LookAheadNamer().transformRE(re);
}
}
