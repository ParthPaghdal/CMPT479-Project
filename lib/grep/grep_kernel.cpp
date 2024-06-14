/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <grep/grep_kernel.h>

#include <grep/grep_engine.h>
#include <grep/grep_toolchain.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/codegenstate.h>
#include <toolchain/toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>          // for Ones
#include <pablo/pe_var.h>           // for Var
#include <pablo/pe_zeroes.h>        // for Zeroes
#include <pablo/pe_infile.h>
#include <pablo/pe_advance.h>
#include <pablo/boolean.h>
#include <pablo/pe_count.h>
#include <pablo/pe_matchstar.h>
#include <pablo/pe_pack.h>
#include <pablo/pe_debugprint.h>
#include <re/printer/re_printer.h>
#include <re/adt/re_cc.h>
#include <re/adt/re_name.h>
#include <re/alphabet/alphabet.h>
#include <re/analysis/re_analysis.h>
#include <re/toolchain/toolchain.h>
#include <re/transforms/re_reverse.h>
#include <re/transforms/re_transformer.h>
#include <re/transforms/to_utf8.h>
#include <re/analysis/collect_ccs.h>
#include <re/transforms/exclude_CC.h>
#include <re/transforms/re_multiplex.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/streams_merge.h>
#include <kernel/unicode/boundary_kernels.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <re/analysis/re_name_gather.h>
#include <re/unicode/boundaries.h>
#include <re/unicode/re_name_resolve.h>
#include <re/unicode/resolve_properties.h>
#include <kernel/unicode/charclasses.h>
#include <re/cc/cc_compiler.h>         // for CC_Compiler
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_kernel.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/compile/re_compiler.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/utf/utf_compiler.h>

using namespace kernel;
using namespace pablo;
using namespace re;
using namespace llvm;

StreamIndexCode ExternalStreamTable::declareStreamIndex(std::string name, StreamIndexCode base, std::string indexStreamName) {
    StreamIndexCode newCode = mStreamIndices.size();
    mStreamIndices.push_back({name, base, indexStreamName});
    mExternalMap.resize(mStreamIndices.size());
    return newCode;
}

StreamIndexCode ExternalStreamTable::getStreamIndex(std::string indexName) {
    for (unsigned i = 0; i < mStreamIndices.size(); i++) {
        if (mStreamIndices[i].name == indexName) return i;
    }
    report_fatal_error(StringRef("Undeclared stream index") + indexName);
}

void ExternalStreamTable::declareExternal(StreamIndexCode c, std::string externalName, ExternalStreamObject * ext) {
    if (grep::ShowExternals) {
        errs() << "declareExternal: " << mStreamIndices[c].name << "_" << externalName << "(";
        auto parms = ext->getParameters();
        bool at_start = true;
        for (auto & p : parms) {
            errs() << (!at_start ? ", " : "") << p;
            at_start = false;
        }
        errs() << ")\n";
    }

    auto & E = mExternalMap[c];
    auto f = E.find(externalName);
    if (LLVM_UNLIKELY(f != E.end())) {
        if (grep::ShowExternals) {
            errs() << "  redeclaration!  Discarding previous declaration.\n";
        }
        const auto curr = f->second;
        if (LLVM_LIKELY(curr != ext)) {
            delete curr;
            f->second = ext;
        }
    } else {
        E.emplace(externalName, ext);
    }
}

ExternalStreamObject * ExternalStreamTable::lookup(StreamIndexCode c, std::string ssname) {
    auto f = mExternalMap[c].find(ssname);
    if (f == mExternalMap[c].end()) {
        report_fatal_error(StringRef("Cannot get external stream object ") +
                           mStreamIndices[c].name + "_" + ssname);
    }
    return f->second;
}

bool ExternalStreamTable::isDeclared(StreamIndexCode c, std::string ssname) {
    return mExternalMap[c].find(ssname) != mExternalMap[c].end();
}

bool ExternalStreamTable::hasReferenceTo(StreamIndexCode c, std::string ssname) {
    for (auto & entry : mExternalMap[c]) {
        auto params = entry.second->getParameters();
        for (auto p : params) {
            if (p == ssname) return true;
        }
    }
    return false;
}

StreamSet * ExternalStreamTable::getStreamSet(ProgBuilderRef b, StreamIndexCode c, std::string ssname) {
    const auto & ext = lookup(c, ssname);
    if (!ext->isResolved()) {
        auto paramNames = ext->getParameters();
        if (grep::ShowExternals) {
            errs() << "resolving External: " << mStreamIndices[c].name << "_" << ssname << "(";
            bool at_start = true;
            for (auto & p : paramNames) {
                errs() << (!at_start ? ", " : "") << p;
                at_start = false;
            }
            errs() << ")\n";
        }
        StreamIndexCode code = isa<FilterByMaskExternal>(ext) ? mStreamIndices[c].base : c;
        bool all_found = true;
        for (auto & p : paramNames) {
            if ((code == c) && (p == ssname)) {
                report_fatal_error(StringRef("Recursion in external resolution: ") + ssname);
            }
            auto f = mExternalMap[code].find(p);
            if (f == mExternalMap[code].end()) {
                all_found = false;
            }
        }
        if (all_found) {
            std::vector<StreamSet *> paramStreams;
            for (auto pName : paramNames) {
                paramStreams.push_back(getStreamSet(b, code, pName));
            }
            ext->resolveStreamSet(b, paramStreams);
        } else {
            auto base = mStreamIndices[c].base;
            if (base == c) {
                report_fatal_error(StringRef("Cannot resolve ") + mStreamIndices[c].name + "_" + ssname);
            }
            mExternalMap[base].emplace(ssname, ext);
            StreamSet * baseSet = getStreamSet(b, base, ssname);
            StreamSet * mask = getStreamSet(b, base, mStreamIndices[c].indexStreamName);
            StreamSet * filtered = b->CreateStreamSet(baseSet->getNumElements());
            FilterByMask(b, mask, baseSet, filtered);
            ext->installStreamSet(filtered);
        }
        StreamSet * s = ext->getStreamSet();
        if (codegen::EnableIllustrator) {
            if (s->getNumElements() == 1) {
                b->captureBitstream(mStreamIndices[c].name + "_" + ssname, s);
            } else {
                b->captureBixNum(mStreamIndices[c].name + "_" + ssname, s);
            }
        }
    }
    return ext->getStreamSet();
}

void ExternalStreamTable::resetExternals() {
    for (unsigned i = 0; i < mExternalMap.size(); i++) {
        for (auto & entry : mExternalMap[i]) {
            entry.second->mStreamSet = nullptr;
        }
    }
}

void ExternalStreamTable::resolveExternals(ProgBuilderRef b) {
    for (unsigned i = 0; i < mExternalMap.size(); i++) {
        for (auto & entry : mExternalMap[i]) {
            if (!entry.second->isResolved()) {
                getStreamSet(b, i, entry.first);
            }
        }
    }
}

ExternalStreamTable::~ExternalStreamTable() {
//    for (std::map<std::string, ExternalStreamObject *> & M : mExternalMap) {
//        for (auto & m : M) {
//            delete m.second;
//        }
//    }
}


void ExternalStreamObject::installStreamSet(StreamSet * s) {
    mStreamSet = s;
}

const std::vector<std::string> LineStartsExternal::getParameters() {
    return mParms;
}

void LineStartsExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * linebreaks = inputs[0];
    StreamSet * linestarts  = b->CreateStreamSet(1);
    if (mParms.size() == 1) {
        b->CreateKernelCall<LineStartsKernel>(linebreaks, linestarts);
    } else {
        StreamSet * index = inputs[1];
        b->CreateKernelCall<LineStartsKernel>(linebreaks, linestarts, index);
    }
    installStreamSet(linestarts);
}

void U21_External::resolveStreamSet(ProgBuilderRef P, std::vector<StreamSet *> inputs) {
    StreamSet * U21 = P->CreateStreamSet(21, 1);
    P->CreateKernelCall<UTF8_Decoder>(inputs[0], U21);
    installStreamSet(U21);
}

void PropertyExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * pStrm  = b->CreateStreamSet(1);
    b->CreateKernelFamilyCall<UnicodePropertyKernelBuilder>(mName, inputs[0], pStrm);
    installStreamSet(pStrm);
}

const std::vector<std::string> PropertyBoundaryExternal::getParameters() {
    std::string basis_name = UCD::getPropertyFullName(mProperty) + "_basis";
    return std::vector<std::string>{basis_name, "u8index"};
}

void PropertyBoundaryExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * basis = inputs[0];
    StreamSet * index = inputs[1];
    StreamSet * bStrm  = b->CreateStreamSet(1);
    b->CreateKernelCall<BoundaryKernel>(basis, index, bStrm);
    installStreamSet(bStrm);
}

void CC_External::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * ccStrm = b->CreateStreamSet(1);
    std::vector<re::CC *> ccs = {mCharClass};
    b->CreateKernelFamilyCall<CharClassesKernel>(ccs, inputs[0], ccStrm);
    installStreamSet(ccStrm);
}

void RE_External::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * reStrm  = b->CreateStreamSet(1);
    #ifndef NDEBUG
    const auto offset =
    #endif
    mGrepEngine->RunGrep(b, mIndexAlphabet, mRE, reStrm);
    assert(offset == static_cast<unsigned>(mOffset));
    installStreamSet(reStrm);
}

void PropertyDistanceExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * propertyBasis = inputs[0];
    StreamSet * distStrm = b->CreateStreamSet(1);
    UCD::PropertyObject * propObj = UCD::getPropertyObject(mProperty);
    if (isa<UCD::CodePointPropertyObject>(propObj)) {
        b->CreateKernelCall<CodePointMatchKernel>(mProperty, mDistance, propertyBasis, distStrm);
    } else {
        b->CreateKernelCall<FixedDistanceMatchesKernel>(mDistance, propertyBasis, distStrm);
    }
    installStreamSet(distStrm);
}

const std::vector<std::string> PropertyDistanceExternal::getParameters() {
    UCD::PropertyObject * propObj = UCD::getPropertyObject(mProperty);
    if (isa<UCD::EnumeratedPropertyObject>(propObj)) {
        return {UCD::getPropertyFullName(mProperty) + "_basis"};
    }
    return {"basis"};
}

void PropertyBasisExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    UCD::PropertyObject * propObj = UCD::getPropertyObject(mProperty);
    if (auto * obj = dyn_cast<UCD::EnumeratedPropertyObject>(propObj)) {
        std::vector<UCD::UnicodeSet> & bases = obj->GetEnumerationBasisSets();
        std::vector<re::CC *> ccs;
        for (auto & b : bases) ccs.push_back(makeCC(b, &cc::Unicode));
        StreamSet * basis = b->CreateStreamSet(ccs.size());
        b->CreateKernelFamilyCall<CharClassesKernel>(ccs, inputs[0], basis);
        installStreamSet(basis);
    } else {
        StreamSet * u21 = b->CreateStreamSet(21);
        b->CreateKernelCall<UTF8_Decoder>(inputs[0], u21);
        installStreamSet(u21);
    }
}

void MultiplexedExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    auto mpx_basis = mAlphabet->getMultiplexedCCs();
    StreamSet * const u8CharClasses = b->CreateStreamSet(mpx_basis.size());
    b->CreateKernelFamilyCall<CharClassesKernel>(mpx_basis, inputs[0], u8CharClasses);
    installStreamSet(u8CharClasses);
}

const std::vector<std::string> GraphemeClusterBreak::getParameters() {
    return std::vector<std::string>{"UCD:" + getPropertyFullName(UCD::GCB) + "_basis", "Extended_Pictographic"};
}

void GraphemeClusterBreak::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * GCBstream = b->CreateStreamSet(1);
    re::RE * GCB_RE = re::generateGraphemeClusterBoundaryRule();
    GCB_RE = UCD::enumeratedPropertiesToCCs(std::set<UCD::property_t>{UCD::GCB}, GCB_RE);
    GCB_RE = UCD::externalizeProperties(GCB_RE);
    //GCB_RE = toUTF8(GCB_RE);
    //StreamSet * idxStrm = (mIndexAlphabet == &cc::UTF8) ? mGrepEngine->mU8index : nullptr;
    mGrepEngine->RunGrep(b, mIndexAlphabet, GCB_RE, GCBstream);
    installStreamSet(GCBstream);
}

const std::vector<std::string> WordBoundaryExternal::getParameters() {
    return std::vector<std::string>{"basis", "u8index"};
}

void WordBoundaryExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * wb = b->CreateStreamSet(1);
    WordBoundaryLogic(b, inputs[0], inputs[1], wb);
    installStreamSet(wb);
}

void FilterByMaskExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * mask = inputs[0];
    StreamSet * toFilter = inputs[1];
    StreamSet * filtered = b->CreateStreamSet(toFilter->getNumElements());
    FilterByMask(b, mask, toFilter, filtered);
    installStreamSet(filtered);
}

const std::vector<std::string> FixedSpanExternal::getParameters() {
    return std::vector<std::string>{mMatchMarks};
}

void FixedSpanExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * matchMarks = inputs[0];
    StreamSet * spans = b->CreateStreamSet(1, 1);
    b->CreateKernelFamilyCall<FixedMatchSpansKernel>(mLengthRange.first, mOffset, matchMarks, spans);
    installStreamSet(spans);
}

const std::vector<std::string> MarkedSpanExternal::getParameters() {
    return std::vector<std::string>{mPrefixMarks, mMatchMarks};
}

void MarkedSpanExternal::resolveStreamSet(ProgBuilderRef P, std::vector<StreamSet *> inputs) {
    //StreamSet * prefixMarks = inputs[0];
    StreamSet * suffixMarks = inputs[1];
    StreamSet * mask = P->CreateStreamSet(1);
    P->CreateKernelCall<StreamsMerge>(inputs, mask);
    StreamSet * filteredSuffix = P->CreateStreamSet(1);
    FilterByMask(P, mask, suffixMarks, filteredSuffix);
    StreamSet * filteredSpanMarks = P->CreateStreamSet(2);
    P->CreateKernelCall<LongestMatchMarks>(filteredSuffix, filteredSpanMarks);
    StreamSet * spanMarks = P->CreateStreamSet(2);
    SpreadByMask(P, mask, filteredSpanMarks, spanMarks);
    StreamSet * spans = P->CreateStreamSet(1);
    P->CreateKernelCall<InclusiveSpans>(mPrefixLength - 1, mOffset, spanMarks, spans);
    installStreamSet(spans);
}

const std::vector<std::string> CCmask::getParameters() {
    //if (mIndexAlphabet != &cc::Unicode) return std::vector<std::string>{"basis", "u8index"};
    return std::vector<std::string>{"basis"};
}

void CCmask::resolveStreamSet(ProgBuilderRef P, std::vector<StreamSet *> inputs) {
    StreamSet * basis = inputs[0];
    StreamSet * mask = P->CreateStreamSet(1);
    StreamSet * index = nullptr;
    if (mIndexAlphabet != &cc::Unicode) {
        //index = inputs[1];
    }
    P->CreateKernelCall<MaskCC>(mCC_to_mask, basis, mask, index);
    installStreamSet(mask);
}

const std::vector<std::string> CCselfTransitionMask::getParameters() {
    return std::vector<std::string>{"basis", "index"};
}

void CCselfTransitionMask::resolveStreamSet(ProgBuilderRef P, std::vector<StreamSet *> inputs) {
    StreamSet * basis = inputs[0];
    StreamSet * index = inputs[1];
    StreamSet * mask = P->CreateStreamSet(1);
    P->CreateKernelCall<MaskSelfTransitions>(mTransitionCCs, basis, mask, index);
    installStreamSet(mask);
}

const std::vector<std::string> MaskedFixedSpanExternal::getParameters() {
    return std::vector<std::string>{mMask, mMatches};
}

void MaskedFixedSpanExternal::resolveStreamSet(ProgBuilderRef P, std::vector<StreamSet *> inputs) {
    StreamSet * positions_mask = inputs[0];
    StreamSet * matches = inputs[1];
    StreamSet * filteredMatches = P->CreateStreamSet(1);
    FilterByMask(P, positions_mask, matches, filteredMatches);
    StreamSet * filteredMatchStarts = P->CreateStreamSet(1);
    P->CreateKernelCall<ShiftBack>(filteredMatches, filteredMatchStarts, mLengthRange.first);
    StreamSet * matchStarts = P->CreateStreamSet(1);
    SpreadByMask(P, positions_mask, filteredMatchStarts, matchStarts);
    StreamSet * spans = P->CreateStreamSet(1);
    P->CreateKernelCall<InclusiveSpans>(0, mOffset, streamutils::Select(P, std::vector<StreamSet *>{matchStarts, matches}), spans);
    installStreamSet(spans);
}

void UTF8_index::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::unique_ptr<cc::CC_Compiler> ccc;
    bool useDirectCC = getInput(0)->getType()->getArrayNumElements() == 1;
    if (useDirectCC) {
        ccc = std::make_unique<cc::Direct_CC_Compiler>(getEntryScope(), pb.createExtract(getInput(0), pb.getInteger(0)));
    } else {
        ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(getEntryScope(), getInputStreamSet("source"));
    }

    Zeroes * const ZEROES = pb.createZeroes();
    PabloAST * const u8pfx = ccc->compileCC(makeByte(0xC0, 0xFF));


    Var * const nonFinal = pb.createVar("nonFinal", u8pfx);
    Var * const u8invalid = pb.createVar("u8invalid", ZEROES);
    Var * const valid_pfx = pb.createVar("valid_pfx", u8pfx);

    auto it = pb.createScope();
    pb.createIf(u8pfx, it);
    PabloAST * const u8pfx2 = ccc->compileCC(makeByte(0xC2, 0xDF), it);
    PabloAST * const u8pfx3 = ccc->compileCC(makeByte(0xE0, 0xEF), it);
    PabloAST * const u8pfx4 = ccc->compileCC(makeByte(0xF0, 0xF4), it);

    //
    // Two-byte sequences
    Var * const anyscope = it.createVar("anyscope", ZEROES);
    auto it2 = it.createScope();
    it.createIf(u8pfx2, it2);
    it2.createAssign(anyscope, it2.createAdvance(u8pfx2, 1));


    //
    // Three-byte sequences
    Var * const EF_invalid = it.createVar("EF_invalid", ZEROES);
    auto it3 = it.createScope();
    it.createIf(u8pfx3, it3);
    PabloAST * const u8scope32 = it3.createAdvance(u8pfx3, 1);
    it3.createAssign(nonFinal, it3.createOr(nonFinal, u8scope32));
    PabloAST * const u8scope33 = it3.createAdvance(u8pfx3, 2);
    PabloAST * const u8scope3X = it3.createOr(u8scope32, u8scope33);
    it3.createAssign(anyscope, it3.createOr(anyscope, u8scope3X));

    PabloAST * const advE0 = it3.createAdvance(ccc->compileCC(makeByte(0xE0), it3), 1, "advEO");
    PabloAST * const range80_9F = ccc->compileCC(makeByte(0x80, 0x9F), it3);
    PabloAST * const E0_invalid = it3.createAnd(advE0, range80_9F, "E0_invalid");

    PabloAST * const advED = it3.createAdvance(ccc->compileCC(makeByte(0xED), it3), 1, "advED");
    PabloAST * const rangeA0_BF = ccc->compileCC(makeByte(0xA0, 0xBF), it3);
    PabloAST * const ED_invalid = it3.createAnd(advED, rangeA0_BF, "ED_invalid");

    PabloAST * const EX_invalid = it3.createOr(E0_invalid, ED_invalid);
    it3.createAssign(EF_invalid, EX_invalid);

    //
    // Four-byte sequences
    auto it4 = it.createScope();
    it.createIf(u8pfx4, it4);
    PabloAST * const u8scope42 = it4.createAdvance(u8pfx4, 1, "u8scope42");
    PabloAST * const u8scope43 = it4.createAdvance(u8scope42, 1, "u8scope43");
    PabloAST * const u8scope44 = it4.createAdvance(u8scope43, 1, "u8scope44");
    PabloAST * const u8scope4nonfinal = it4.createOr(u8scope42, u8scope43);
    it4.createAssign(nonFinal, it4.createOr(nonFinal, u8scope4nonfinal));
    PabloAST * const u8scope4X = it4.createOr(u8scope4nonfinal, u8scope44);
    it4.createAssign(anyscope, it4.createOr(anyscope, u8scope4X));
    PabloAST * const F0_invalid = it4.createAnd(it4.createAdvance(ccc->compileCC(makeByte(0xF0), it4), 1), ccc->compileCC(makeByte(0x80, 0x8F), it4));
    PabloAST * const F4_invalid = it4.createAnd(it4.createAdvance(ccc->compileCC(makeByte(0xF4), it4), 1), ccc->compileCC(makeByte(0x90, 0xBF), it4));
    PabloAST * const FX_invalid = it4.createOr(F0_invalid, F4_invalid);
    it4.createAssign(EF_invalid, it4.createOr(EF_invalid, FX_invalid));

    //
    // Invalid cases
    PabloAST * const legalpfx = it.createOr(it.createOr(u8pfx2, u8pfx3), u8pfx4);
    //  Any scope that does not have a suffix byte, and any suffix byte that is not in
    //  a scope is a mismatch, i.e., invalid UTF-8.
    PabloAST * const u8suffix = ccc->compileCC("u8suffix", makeByte(0x80, 0xBF), it);
    PabloAST * const mismatch = it.createXor(anyscope, u8suffix);
    //
    PabloAST * const pfx_invalid = it.createXor(valid_pfx, legalpfx);
    it.createAssign(u8invalid, it.createOr(pfx_invalid, it.createOr(mismatch, EF_invalid)));
    PabloAST * const u8valid = it.createNot(u8invalid, "u8valid");
    //
    it.createAssign(nonFinal, it.createAnd(nonFinal, u8valid));

    Var * const u8index = getOutputStreamVar("u8index");
    PabloAST * u8final = pb.createInFile(pb.createNot(nonFinal));
    if (getNumOfStreamInputs() > 1) {
        u8final = pb.createOr(u8final, getInputStreamSet("u8_LB")[0]);
    }
    pb.createAssign(pb.createExtract(u8index, pb.getInteger(0)), u8final);
}

UTF8_index::UTF8_index(KernelBuilder & kb, StreamSet * Source, StreamSet * u8index, StreamSet * u8_LB)
: PabloKernel(kb, [&]() -> std::string {
    std::stringstream s;
    s << "UTF8_index_";
    s << Source->getNumElements() << "x" << Source->getFieldWidth();
    if (u8_LB) {
        s << "_LB";
    }
    return s.str();}(),
{}, {Binding{"u8index", u8index}}) {
    mInputStreamSets.push_back(Binding{"source", Source});
    if (u8_LB) {
        mInputStreamSets.push_back(Binding{"u8_LB", u8_LB, FixedRate(), Principal()});
    }
}

void GrepKernelOptions::setBarrier(StreamSet * b) {
    mBarrierStream = b;
}

void GrepKernelOptions::setIndexing(StreamSet * idx) {
    mIndexStream = idx;
}

void GrepKernelOptions::setRE(RE * e) {mRE = e;}
void GrepKernelOptions::setCombiningStream(GrepCombiningType t, StreamSet * toCombine){
    mCombiningType = t;
    mCombiningStream = toCombine;
}
void GrepKernelOptions::setResults(StreamSet * r) {mResults = r;}

void GrepKernelOptions::addAlphabet(const cc::Alphabet * a, StreamSet * basis) {
    mAlphabets.emplace_back(a, basis);
}

unsigned round_up_to_blocksize(unsigned offset) {
    unsigned lookahead_blocks = (codegen::BlockSize - 1 + offset)/codegen::BlockSize;
    return lookahead_blocks * codegen::BlockSize;
}

void GrepKernelOptions::addExternal(std::string name, StreamSet * strm, unsigned offset, std::pair<int, int> lengthRange) {
    if (offset == 0) {
        mExternalBindings.emplace_back(name, strm);
    } else {
        unsigned ahead = round_up_to_blocksize(offset);
        mExternalBindings.emplace_back(name, strm, FixedRate(), LookAhead(ahead));
    }
    mExternalOffsets.push_back(offset);
    mExternalLengths.push_back(lengthRange);
}

Bindings GrepKernelOptions::streamSetInputBindings() {
    Bindings inputs;
    if (mBarrierStream) {
        inputs.emplace_back("mBarrier", mBarrierStream);
    }
    for (const auto & a : mAlphabets) {
        inputs.emplace_back(a.first->getName() + "_basis", a.second);
    }
    for (const auto & a : mExternalBindings) {
        inputs.emplace_back(a);
    }
    if (mIndexStream) {
        inputs.emplace_back("mIndexing", mIndexStream);
    }
    if (mCombiningType != GrepCombiningType::None) {
        inputs.emplace_back("toCombine", mCombiningStream, FixedRate(), Add1());
    }
    return inputs;
}

Bindings GrepKernelOptions::streamSetOutputBindings() {
    return {Binding{"matches", mResults, FixedRate(), Add1()}};
}

Bindings GrepKernelOptions::scalarInputBindings() {
    return {};
}

Bindings GrepKernelOptions::scalarOutputBindings() {
    return {};
}

GrepKernelOptions::GrepKernelOptions(const cc::Alphabet * codeUnitAlphabet)
: mCodeUnitAlphabet(codeUnitAlphabet) {

}

std::string GrepKernelOptions::makeSignature() {
    std::string tmp;
    std::vector<std::string> externals;
    std::set<std::string> canon_externals;
    raw_string_ostream sig(mSignature);
    std::string alpha_prefix = "";
    for (const auto & a: mAlphabets) {
        sig << alpha_prefix << a.second->getNumElements() << "xi" << a.second->getFieldWidth();
        alpha_prefix = "!";
    }
    if (mBarrierStream == nullptr) sig << "-barrier";
    if (mIndexStream) sig << "+ix";
    for (unsigned i = 0; i < mExternalBindings.size(); i++) {
        auto & e = mExternalBindings[i];
        std::string canon = "@" + std::to_string(i);
        if (e.hasLookahead()) {
            canon += std::to_string(round_up_to_blocksize(e.getLookahead()));
        }
        externals.push_back(e.getName());
        canon_externals.insert(canon);
    }
    if (mCombiningType == GrepCombiningType::Exclude) {
        sig << "&~";
    } else if (mCombiningType == GrepCombiningType::Include) {
        sig << "|=";
    }
    RE * canonRE = canonicalizeExternals(mRE, externals);
    sig << ':' << Printer_RE::PrintRE(canonRE, canon_externals);
    sig.flush();
    return mSignature;
}

ICGrepKernel::ICGrepKernel(KernelBuilder & b, std::unique_ptr<GrepKernelOptions> && options)
: PabloKernel(b, AnnotateWithREflags("ic") + getStringHash(options->makeSignature()),
options->streamSetInputBindings(),
options->streamSetOutputBindings(),
options->scalarInputBindings(),
options->scalarOutputBindings()),
mOptions(std::move(options)),
mSignature(mOptions->getSignature()) {
    addAttribute(InfrequentlyUsed());
    mOffset = grepOffset(mOptions->mRE);
    if (grep::ShowExternals) {
        errs() << "ICGrep signature: " << mSignature << "\n";
        errs() << "signature hash:" << getStringHash(mSignature) << "\n";
    }
}

StringRef ICGrepKernel::getSignature() const {
    return mSignature;
}

void ICGrepKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * barrier = nullptr;
    if (mOptions->mBarrierStream) {
        barrier = pb.createExtract(getInputStreamVar("mBarrier"), pb.getInteger(0));
    }
    RE_Compiler re_compiler(getEntryScope(), barrier, mOptions->mCodeUnitAlphabet);
    for (unsigned i = 0; i < mOptions->mAlphabets.size(); i++) {
        auto & alpha = mOptions->mAlphabets[i].first;
        auto basis = getInputStreamSet(alpha->getName() + "_basis");
        re_compiler.addAlphabet(alpha, basis);
    }
    if (mOptions->mIndexStream) {
        PabloAST * idxStrm = pb.createExtract(getInputStreamVar("mIndexing"), pb.getInteger(0));
        re_compiler.setIndexing(&cc::Unicode, idxStrm);
    }
    for (unsigned i = 0; i < mOptions->mExternalBindings.size(); i++) {
        auto extName = mOptions->mExternalBindings[i].getName();
        PabloAST * extStrm = pb.createExtract(getInputStreamVar(extName), pb.getInteger(0));
        unsigned offset = mOptions->mExternalOffsets[i];
        std::pair<int, int> lgthRange = mOptions->mExternalLengths[i];
        re_compiler.addPrecompiled(extName, RE_Compiler::ExternalStream(RE_Compiler::Marker(extStrm, offset), lgthRange));
    }
    Var * const final_matches = pb.createVar("final_matches", pb.createZeroes());
    RE_Compiler::Marker matches = re_compiler.compileRE(mOptions->mRE);
    PabloAST * matchResult = matches.stream();
    if (matches.offset() != mOffset) {
        //errs() << Printer_RE::PrintRE(mOptions->mRE) <<"\n mOffset = " << mOffset << "\n";
        //report_fatal_error("matches.offset() != mOffset");
    }
    pb.createAssign(final_matches, matchResult);
    Var * const output = pb.createExtract(getOutputStreamVar("matches"), pb.getInteger(0));
    PabloAST * value = nullptr;
    if (mOptions->mCombiningType == GrepCombiningType::None) {
        value = final_matches;
    } else {
        PabloAST * toCombine = pb.createExtract(getInputStreamVar("toCombine"), pb.getInteger(0));
        if (mOptions->mCombiningType == GrepCombiningType::Exclude) {
            value = pb.createAnd(toCombine, pb.createNot(final_matches), "toCombine");
        } else {
            value = pb.createOr(toCombine, final_matches, "toCombine");
        }
    }
    pb.createAssign(output, value);
}

void MatchedLinesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    auto matchResults = getInputStreamSet("matchResults");
    PabloAST * lineBreaks = pb.createExtract(getInputStreamVar("lineBreaks"), pb.getInteger(0));
    PabloAST * notLB = pb.createNot(lineBreaks);
    PabloAST * match_follow = pb.createMatchStar(matchResults.back(), notLB);
    Var * const matchedLines = getOutputStreamVar("matchedLines");
    pb.createAssign(pb.createExtract(matchedLines, pb.getInteger(0)), pb.createAnd(match_follow, lineBreaks, "matchedLines"));
}

MatchedLinesKernel::MatchedLinesKernel (KernelBuilder & b, StreamSet * Matches, StreamSet * LineBreakStream, StreamSet * MatchedLines)
: PabloKernel(b, "MatchedLines" + std::to_string(Matches->getNumElements()),
// inputs
{Binding{"matchResults", Matches}
,Binding{"lineBreaks", LineBreakStream, FixedRate()}},
// output
{Binding{"matchedLines", MatchedLines}}) {

}

void InvertMatchesKernel::generateDoBlockMethod(KernelBuilder & b) {
    Value * input = b.loadInputStreamBlock("matchedLines", b.getInt32(0));
    Value * lbs = b.loadInputStreamBlock("lineBreaks", b.getInt32(0));
    Value * inverted = b.CreateAnd(b.CreateNot(input), lbs, "inverted");
    b.storeOutputStreamBlock("nonMatches", b.getInt32(0), inverted);
}

InvertMatchesKernel::InvertMatchesKernel(KernelBuilder & b, StreamSet * Matches, StreamSet * LineBreakStream, StreamSet * InvertedMatches)
: BlockOrientedKernel(b, "Invert" + std::to_string(Matches->getNumElements()),
// Inputs
{Binding{"matchedLines", Matches},
 Binding{"lineBreaks", LineBreakStream}},
// Outputs
{Binding{"nonMatches", InvertedMatches}},
// Input/Output Scalars and internal state
{}, {}, {}) {

}

FixedMatchSpansKernel::FixedMatchSpansKernel(KernelBuilder & b, unsigned length, unsigned offset, StreamSet * MatchMarks, StreamSet * MatchSpans)
: PabloKernel(b, "FixedMatchSpansKernel" + std::to_string(MatchMarks->getNumElements()) + "x1_by" + std::to_string(length) + '@' + std::to_string(offset),
{Binding{"MatchMarks", MatchMarks, FixedRate(1), LookAhead(round_up_to_blocksize(length))}}, {Binding{"MatchSpans", MatchSpans}}),
mMatchLength(length), mOffset(offset) {
}

void FixedMatchSpansKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * marks = pb.createExtract(getInputStreamVar("MatchMarks"), pb.getInteger(0));
    Var * matchSpansVar = getOutputStreamVar("MatchSpans");
    // starts of all the matches
    PabloAST * starts = pb.createLookahead(marks, mMatchLength + mOffset - 1);
    // now find all consecutive positions within mMatchLength of any start.
    unsigned consecutiveCount = 1;
    PabloAST * consecutive = starts;
    for (unsigned i = 1; i <= mMatchLength/2; i *= 2) {
        consecutiveCount += i;
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, i),
                                  "consecutive" + std::to_string(consecutiveCount));
    }
    if (consecutiveCount < mMatchLength) {
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, mMatchLength - consecutiveCount),
                                  "consecutive" + std::to_string(mMatchLength));
    }
    pb.createAssign(pb.createExtract(matchSpansVar, 0), consecutive);
}

SpansToMarksKernel::SpansToMarksKernel(KernelBuilder & b, StreamSet * Spans, StreamSet * Marks)
: PabloKernel(b, "SpansToMarksKernel",
{Binding{"Spans", Spans}}, {Binding{"Marks", Marks}}) {}

void SpansToMarksKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * spans = getInputStreamSet("Spans")[0];
    Var * matchEndsVar = getOutputStreamVar("Marks");
    PabloAST * starts = pb.createAnd(spans, pb.createNot(pb.createAdvance(spans, 1)), "starts");
    PabloAST * follows = pb.createAnd(pb.createAdvance(spans, 1), pb.createNot(spans), "follows");
    pb.createAssign(pb.createExtract(matchEndsVar, 0), starts);
    pb.createAssign(pb.createExtract(matchEndsVar, 1), follows);
}

U8Spans::U8Spans(KernelBuilder & b, StreamSet * marks, StreamSet * u8index, StreamSet * spans, pablo::BitMovementMode m)
: PabloKernel(b, "U8Spans_" + pablo::BitMovementMode_string(m), {}, {Binding{"spans", spans}}),
    mBitMovement(m) {
        if (m == pablo::BitMovementMode::LookAhead) {
            mInputStreamSets.push_back(Binding{"marks", marks, FixedRate(1), LookAhead(3)});
            mInputStreamSets.push_back(Binding{"u8index", u8index, FixedRate(1), LookAhead(3)});
        } else {
            mInputStreamSets.push_back(Binding{"marks", marks});
            mInputStreamSets.push_back(Binding{"u8index", u8index});
        }

    }

void U8Spans::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * marks = getInputStreamSet("marks")[0];
    PabloAST * u8index = getInputStreamSet("u8index")[0];
    PabloAST * spans = nullptr;
    if (mBitMovement == BitMovementMode::LookAhead) {
        PabloAST * back1 = pb.createAnd(pb.createLookahead(marks, 1), pb.createNot(u8index));
        PabloAST * ix_or_next = pb.createOr(u8index, pb.createLookahead(u8index, 1));
        PabloAST * back2 = pb.createAnd(pb.createLookahead(marks, 2), pb.createNot(ix_or_next));
        PabloAST * ix_or_next2 = pb.createOr(ix_or_next, pb.createLookahead(u8index, 2));
        PabloAST * back3 = pb.createAnd(pb.createLookahead(marks, 3), pb.createNot(ix_or_next2));
        spans = pb.createOr(marks, pb.createOr3(back1, back2, back3));
    } else {
        spans = pb.createMatchStar(marks, pb.createNot(u8index));
    }
    Var * spansVar = getOutputStreamVar("spans");
    pb.createAssign(pb.createExtract(spansVar, 0), spans);
}

void PopcountKernel::generatePabloMethod() {
    auto pb = getEntryScope();
    const auto toCount = pb->createExtract(getInputStreamVar("toCount"), pb->getInteger(0));
    const auto countResult = getOutputScalarVar("countResult");
    const auto newCount = pb->createCount(pb->createInFile(toCount));
    pb->createAssign(countResult, newCount);
}

PopcountKernel::PopcountKernel (KernelBuilder & b, StreamSet * const toCount, Scalar * countResult)
: PabloKernel(b, "Popcount",
{Binding{"toCount", toCount}},
{},
{},
{Binding{"countResult", countResult}}) {

}

PabloAST * matchDistanceCheck(PabloBuilder & b, unsigned distance, std::vector<PabloAST *> basis1, std::vector<PabloAST *> basis2) {
    PabloAST * differ = b.createZeroes();
    for (unsigned i = 0; i < basis1.size(); i++) {
        PabloAST * advanced = b.createAdvance(basis1[i], distance);
        differ = b.createOr(differ, b.createXor(advanced, basis2[i]));
    }
    return differ;
}

void FixedDistanceMatchesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    auto basis = getInputStreamSet("Basis");
    Var * mismatch = pb.createVar("mismatch", pb.createZeroes());
    if (mHasCheckStream) {
        auto ToCheck = getInputStreamSet("ToCheck")[0];
        auto it = pb.createScope();
        pb.createIf(ToCheck, it);
        PabloAST * differ = matchDistanceCheck(it, mMatchDistance, basis, basis);
        it.createAssign(mismatch, it.createAnd(ToCheck, differ));
    } else {
        pb.createAssign(mismatch, matchDistanceCheck(pb, mMatchDistance, basis, basis));
    }
    Var * const MatchVar = getOutputStreamVar("Matches");
    pb.createAssign(pb.createExtract(MatchVar, pb.getInteger(0)), pb.createNot(mismatch, "matches"));
}

FixedDistanceMatchesKernel::FixedDistanceMatchesKernel (KernelBuilder & b, unsigned distance, StreamSet * Basis, StreamSet * Matches, StreamSet * ToCheck)
: PabloKernel(b, "Distance_" + std::to_string(distance) + "_Matches_" + std::to_string(Basis->getNumElements()) + "x1" + (ToCheck == nullptr ? "" : "_withCheck"),
// inputs
{Binding{"Basis", Basis}},
// output
{Binding{"Matches", Matches}}), mMatchDistance(distance), mHasCheckStream(ToCheck != nullptr) {
    if (mHasCheckStream) {
        mInputStreamSets.push_back({"ToCheck", ToCheck});
    }
}

void CodePointMatchKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UCD::PropertyObject * propObj = UCD::getPropertyObject(mProperty);
    if (UCD::CodePointPropertyObject * p = dyn_cast<UCD::CodePointPropertyObject>(propObj)) {
        const UCD::UnicodeSet & nullSet = p->GetNullSet();
        std::vector<UCD::UnicodeSet> & xfrms = p->GetBitTransformSets();
        std::vector<re::CC *> xfrm_ccs;
        for (auto & b : xfrms) xfrm_ccs.push_back(makeCC(b, &cc::Unicode));
        UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
        Var * nullVar = nullptr;
        if (!nullSet.empty()) {
            re::CC * nullCC = makeCC(nullSet, &cc::Unicode);
            nullVar = pb.createVar("null_set", pb.createZeroes());
            unicodeCompiler.addTarget(nullVar, nullCC);
        }
        std::vector<Var *> xfrm_vars;
        for (unsigned i = 0; i < xfrm_ccs.size(); i++) {
            xfrm_vars.push_back(pb.createVar("xfrm_cc_" + std::to_string(i), pb.createZeroes()));
            unicodeCompiler.addTarget(xfrm_vars[i], xfrm_ccs[i]);
        }
        if (LLVM_UNLIKELY(AlgorithmOptionIsSet(DisableIfHierarchy))) {
            unicodeCompiler.compile(UTF::UTF_Compiler::IfHierarchy::None);
        } else {
            unicodeCompiler.compile();
        }
        std::vector<PabloAST *> basis = getInputStreamSet("Basis");
        std::vector<PabloAST *> transformed(basis.size());
        for (unsigned i = 0; i < basis.size(); i++) {
            if (i < xfrm_vars.size()) {
                transformed[i] = pb.createXor(xfrm_vars[i], basis[i]);
            } else {
                transformed[i] = basis[i];
            }
        }
        PabloAST * mismatch;
        bool involution = ((mProperty == UCD::bpb) || (mProperty == UCD::bmg));
        if (involution) {
            mismatch = matchDistanceCheck(pb, mMatchDistance, transformed, basis);
        } else {
            mismatch = matchDistanceCheck(pb, mMatchDistance, transformed, transformed);
        }
        if (!nullSet.empty()) {
            mismatch = pb.createOr(mismatch, nullVar);
        }
        PabloAST * matches = pb.createInFile(pb.createNot(mismatch));
        Var * const MatchVar = getOutputStreamVar("Matches");
        pb.createAssign(pb.createExtract(MatchVar, pb.getInteger(0)), matches);
    } else {
        llvm::report_fatal_error("Expecting codepoint property");
    }
}

CodePointMatchKernel::CodePointMatchKernel (KernelBuilder & b, UCD::property_t prop, unsigned distance, StreamSet * Basis, StreamSet * Matches)
: PabloKernel(b, getPropertyEnumName(prop) + "_dist_" + std::to_string(distance) + "_Matches_" + std::to_string(Basis->getNumElements()) + "x1",
// inputs
{Binding{"Basis", Basis}},
// output
{Binding{"Matches", Matches}}),
    mMatchDistance(distance),
    mProperty(prop) {
}

void AbortOnNull::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    Module * const m = b.getModule();
    DataLayout DL(m);
    IntegerType * const intPtrTy = DL.getIntPtrType(m->getContext());
    Type * voidPtrTy = b.getVoidPtrTy();
    Type * blockTy = b.getBitBlockType();
    const auto blocksPerStride = getStride() / b.getBitBlockWidth();
    Constant * const BLOCKS_PER_STRIDE = b.getSize(blocksPerStride);
    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const strideLoop = b.CreateBasicBlock("strideLoop");
    BasicBlock * const stridesDone = b.CreateBasicBlock("stridesDone");
    BasicBlock * const nullByteDetection = b.CreateBasicBlock("nullByteDetection");
    BasicBlock * const nullByteFound = b.CreateBasicBlock("nullByteFound");
    BasicBlock * const finalStride = b.CreateBasicBlock("finalStride");
    BasicBlock * const segmentDone = b.CreateBasicBlock("segmentDone");

    Value * const numOfBlocks = b.CreateMul(numOfStrides, BLOCKS_PER_STRIDE);
    Value * itemsToDo = b.getAccessibleItemCount("byteData");
    //
    // Fast loop to prove that there are no null bytes in a multiblock region.
    // We repeatedly combine byte packs using a SIMD unsigned min operation
    // (implemented as a Select/ICmpULT combination).
    //
    Value * byteStreamBasePtr = b.getInputStreamBlockPtr("byteData", b.getSize(0), b.getSize(0));
    Value * outputStreamBasePtr = b.getOutputStreamBlockPtr("untilNull", b.getSize(0), b.getSize(0));

    //
    // We set up a a set of eight accumulators to accumulate the minimum byte
    // values seen at each position in a block.   The initial min value at
    // each position is 0xFF (all ones).
    Value * blockMin[8];
    for (unsigned i = 0; i < 8; i++) {
        blockMin[i] = b.fwCast(8, b.allOnes());
    }
    // If we're in the final block bypass the fast loop.
    b.CreateCondBr(b.isFinal(), finalStride, strideLoop);

    b.SetInsertPoint(strideLoop);
    PHINode * const baseBlockIndex = b.CreatePHI(b.getSizeTy(), 2);
    baseBlockIndex->addIncoming(ConstantInt::get(baseBlockIndex->getType(), 0), entry);
    PHINode * const blocksRemaining = b.CreatePHI(b.getSizeTy(), 2);
    blocksRemaining->addIncoming(numOfBlocks, entry);
    FixedArray<Value *, 2> indices;
    indices[0] = baseBlockIndex;
    for (unsigned i = 0; i < 8; i++) {
        indices[1] = b.getSize(i);
        Value * next = b.CreateBlockAlignedLoad(blockTy, b.CreateGEP(blockTy, byteStreamBasePtr, indices));
        b.CreateBlockAlignedStore(next, b.CreateGEP(blockTy, outputStreamBasePtr, indices));
        next = b.fwCast(8, next);
        blockMin[i] = b.CreateSelect(b.CreateICmpULT(next, blockMin[i]), next, blockMin[i]);
    }
    Value * nextBlockIndex = b.CreateAdd(baseBlockIndex, ConstantInt::get(baseBlockIndex->getType(), 1));
    Value * nextRemaining = b.CreateSub(blocksRemaining, ConstantInt::get(blocksRemaining->getType(), 1));
    baseBlockIndex->addIncoming(nextBlockIndex, strideLoop);
    blocksRemaining->addIncoming(nextRemaining, strideLoop);
    b.CreateCondBr(b.CreateICmpUGT(nextRemaining, ConstantInt::getNullValue(blocksRemaining->getType())), strideLoop, stridesDone);

    b.SetInsertPoint(stridesDone);
    // Combine the 8 blockMin values.
    for (unsigned i = 0; i < 4; i++) {
        blockMin[i] = b.CreateSelect(b.CreateICmpULT(blockMin[i], blockMin[i+4]), blockMin[i], blockMin[i+4]);
    }
    for (unsigned i = 0; i < 2; i++) {
        blockMin[i] = b.CreateSelect(b.CreateICmpULT(blockMin[i], blockMin[i+4]), blockMin[i], blockMin[i+2]);
    }
    blockMin[0] = b.CreateSelect(b.CreateICmpULT(blockMin[0], blockMin[1]), blockMin[0], blockMin[1]);
    Value * anyNull = b.bitblock_any(b.simd_eq(8, blockMin[0], b.allZeroes()));

    b.CreateCondBr(anyNull, nullByteDetection, segmentDone);


    b.SetInsertPoint(finalStride);
    b.CreateMemCpy(b.CreatePointerCast(outputStreamBasePtr, voidPtrTy), b.CreatePointerCast(byteStreamBasePtr, voidPtrTy), itemsToDo, 1);
    b.CreateBr(nullByteDetection);

    b.SetInsertPoint(nullByteDetection);
    //  Find the exact location using memchr, which should be fast enough.
    //
    Value * ptrToNull = b.CreateMemChr(b.CreatePointerCast(byteStreamBasePtr, voidPtrTy), b.getInt32(0), itemsToDo);
    Value * ptrAddr = b.CreatePtrToInt(ptrToNull, intPtrTy);
    b.CreateCondBr(b.CreateICmpEQ(ptrAddr, ConstantInt::getNullValue(intPtrTy)), segmentDone, nullByteFound);

    // A null byte has been located; set the termination code and call the signal handler.
    b.SetInsertPoint(nullByteFound);
    Value * nullPosn = b.CreateSub(b.CreatePtrToInt(ptrToNull, intPtrTy), b.CreatePtrToInt(byteStreamBasePtr, intPtrTy));
    b.setFatalTerminationSignal();
    Function * const dispatcher = m->getFunction("signal_dispatcher"); assert (dispatcher);
    Value * handler = b.getScalarField("handler_address");
    b.CreateCall(dispatcher, {handler, ConstantInt::get(b.getInt32Ty(), static_cast<unsigned>(grep::GrepSignal::BinaryFile))});
    b.CreateBr(segmentDone);

    b.SetInsertPoint(segmentDone);
    PHINode * const produced = b.CreatePHI(b.getSizeTy(), 3);
    produced->addIncoming(nullPosn, nullByteFound);
    produced->addIncoming(itemsToDo, stridesDone);
    produced->addIncoming(itemsToDo, nullByteDetection);
    Value * producedCount = b.getProducedItemCount("untilNull");
    producedCount = b.CreateAdd(producedCount, produced);
    b.setProducedItemCount("untilNull", producedCount);
}

AbortOnNull::AbortOnNull(KernelBuilder & b, StreamSet * const InputStream, StreamSet * const OutputStream, Scalar * callbackObject)
: MultiBlockKernel(b, "AbortOnNull",
// inputs
{Binding{"byteData", InputStream, FixedRate(), Principal()}},
// outputs
{Binding{ "untilNull", OutputStream, FixedRate(), Deferred()}},
// input scalars
{Binding{"handler_address", callbackObject}},
{}, {}) {
    addAttribute(CanTerminateEarly());
    addAttribute(MayFatallyTerminate());
}

ContextSpan::ContextSpan(KernelBuilder & b, StreamSet * const markerStream, StreamSet * const contextStream, unsigned before, unsigned after)
: PabloKernel(b, "ContextSpan-" + std::to_string(before) + "+" + std::to_string(after),
              // input
{Binding{"markerStream", markerStream, FixedRate(1), LookAhead(before)}},
              // output
{Binding{"contextStream", contextStream}}),
mBeforeContext(before), mAfterContext(after) {
}

void ContextSpan::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    Var * markerStream = pb.createExtract(getInputStreamVar("markerStream"), pb.getInteger(0));
    PabloAST * contextStart = pb.createLookahead(markerStream, pb.getInteger(mBeforeContext));
    unsigned lgth = mBeforeContext + 1 + mAfterContext;
    PabloAST * consecutive = contextStart;
    unsigned consecutiveCount = 1;
    for (unsigned i = 1; i <= lgth/2; i *= 2) {
        consecutiveCount += i;
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, i),
                                  "consecutive" + std::to_string(consecutiveCount));
    }
    if (consecutiveCount < lgth) {
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, lgth - consecutiveCount),
                                  "consecutive" + std::to_string(lgth));
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("contextStream"), pb.getInteger(0)), pb.createInFile(consecutive));
}

void kernel::GraphemeClusterLogic(ProgBuilderRef P,
                                  StreamSet * Source, StreamSet * U8index, StreamSet * GCBstream) {

    re::RE * GCB = re::generateGraphemeClusterBoundaryRule();
    const auto GCB_Sets = re::collectCCs(GCB, cc::Unicode, re::NameProcessingMode::ProcessDefinition);
    auto GCB_mpx = cc::makeMultiplexedAlphabet("GCB_mpx", GCB_Sets);
    GCB = transformCCs(GCB_mpx, GCB, re::NameTransformationMode::TransformDefinition);
    auto GCB_basis = GCB_mpx->getMultiplexedCCs();
    StreamSet * const GCB_Classes = P->CreateStreamSet(GCB_basis.size());
    P->CreateKernelFamilyCall<CharClassesKernel>(GCB_basis, Source, GCB_Classes);
    std::unique_ptr<GrepKernelOptions> options = std::make_unique<GrepKernelOptions>();
    options->setIndexing(U8index);
    options->setRE(GCB);
    options->addAlphabet(GCB_mpx, GCB_Classes);
    options->setResults(GCBstream);
    options->addExternal("UTF8_index", U8index);
    P->CreateKernelFamilyCall<ICGrepKernel>(std::move(options));
}

void kernel::WordBoundaryLogic(ProgBuilderRef P,
                                  StreamSet * Source, StreamSet * U8index, StreamSet * wordBoundary_stream) {

    re::RE * wordProp = re::makePropertyExpression(PropertyExpression::Kind::Codepoint, "word");
    wordProp = UCD::linkAndResolve(wordProp);
    re::Name * word = re::makeName("word");
    word->setDefinition(wordProp);
    StreamSet * WordStream = P->CreateStreamSet(1);
    P->CreateKernelFamilyCall<UnicodePropertyKernelBuilder>(word, Source, WordStream);
    P->CreateKernelCall<BoundaryKernel>(WordStream, U8index, wordBoundary_stream);
}

LongestMatchMarks::LongestMatchMarks(KernelBuilder & b, StreamSet * start_ends, StreamSet * marks)
: PabloKernel(b, "LongestMatchMarks"  + std::to_string(marks->getNumElements()) + "x1",
              {Binding{"start_ends", start_ends, FixedRate(1), LookAhead(1)}},
              {Binding{"marks", marks}}) {}

void LongestMatchMarks::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> starts_ends = getInputStreamSet("start_ends");
    PabloAST * starts;
    PabloAST * ends;
    if (starts_ends.size() == 2) {
        starts = starts_ends[0];
        ends = starts_ends[1];
    } else {
        starts = pb.createNot(starts_ends[0]);
        ends = starts_ends[0];
    }
    PabloAST * end_follows = pb.createLookahead(ends, 1);
    PabloAST * span_starts = pb.createAnd(starts, end_follows, "span_starts");
    PabloAST * span_ends = pb.createAnd(ends, pb.createNot(end_follows), "span_ends");
    Var * marksVar = getOutputStreamVar("marks");
    pb.createAssign(pb.createExtract(marksVar, pb.getInteger(0)), span_starts);
    pb.createAssign(pb.createExtract(marksVar, pb.getInteger(1)), span_ends);
}

unsigned spanLookAhead(unsigned offset1, unsigned offset2) {
    return round_up_to_blocksize(std::max(offset1, offset2));
}

InclusiveSpans::InclusiveSpans(KernelBuilder & b,
                               unsigned prefixOffset, unsigned suffixOffset,
                               StreamSet * marks, StreamSet * spans)
: PabloKernel(b, "InclusiveSpans@" + std::to_string(prefixOffset) + ":" + std::to_string(suffixOffset),
              {Binding{"marks", marks, FixedRate(1),
                                       LookAhead(spanLookAhead(prefixOffset, suffixOffset))}},
              {Binding{"spans", spans}}),
    mPrefixOffset(prefixOffset), mSuffixOffset(suffixOffset) {
}

void InclusiveSpans::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    Var * marksVar = getInputStreamVar("marks");
    PabloAST * starts = pb.createExtract(marksVar, pb.getInteger(0));
    if (mPrefixOffset > 0) {
        starts = pb.createLookahead(starts, mPrefixOffset);
    }
    PabloAST * ends = pb.createExtract(marksVar, pb.getInteger(1));
    if (mSuffixOffset > 0) {
        ends = pb.createLookahead(ends, mSuffixOffset);
    }
    PabloAST * spans = pb.createIntrinsicCall(pablo::Intrinsic::InclusiveSpan, {starts, ends});
    pb.createAssign(pb.createExtract(getOutputStreamVar("spans"), pb.getInteger(0)), spans);
}

std::string CC_string(std::vector<const CC *> transitionCCs, StreamSet * index) {
    std::stringstream s;
    if (index != nullptr) s << "+ix";
    for (auto & cc : transitionCCs) {
        s << "_" << cc->canonicalName();
    }
    return s.str();
}

MaskCC::MaskCC(KernelBuilder & b, const CC * CC_to_mask, StreamSet * basis, StreamSet * mask, StreamSet * index)
: PabloKernel(b, "MaskCC" + basis->shapeString() + CC_string(std::vector<const CC *>{CC_to_mask}, index),
              {Binding{"basis", basis}},
              {Binding{"mask", mask}}), mCC_to_mask(CC_to_mask), mIndexStrm(nullptr) {
                  if (index != nullptr) {
                      mInputStreamSets.push_back(Binding{"index", index});
                      mIndexStrm = index;
                  }
              }

void MaskCC::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    Var * maskVar = pb.createVar("maskVar", pb.createZeroes());
    unicodeCompiler.addTarget(maskVar, mCC_to_mask);
    if (LLVM_UNLIKELY(AlgorithmOptionIsSet(DisableIfHierarchy))) {
        unicodeCompiler.compile(UTF::UTF_Compiler::IfHierarchy::None);
    } else {
        unicodeCompiler.compile();
    }
    PabloAST * mask = pb.createNot(maskVar);
    if (mIndexStrm) {
        PabloAST * idx = getInputStreamSet("index")[0];
        mask = pb.createAnd(idx, mask);
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("mask"), pb.getInteger(0)), mask);
}

MaskSelfTransitions::MaskSelfTransitions(KernelBuilder & b, const std::vector<const CC *> transitionCCs, StreamSet * basis, StreamSet * mask, StreamSet * index)
: PabloKernel(b, "MaskSelfTransitions" + basis->shapeString() + CC_string(transitionCCs, index),
              {Binding{"basis", basis}},
              {Binding{"mask", mask}}), mTransitionCCs(transitionCCs), mIndexStrm(nullptr) {
                  if (index != nullptr) {
                      mInputStreamSets.push_back(Binding{"index", index});
                      mIndexStrm = index;
                  }
              }

void MaskSelfTransitions::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::unique_ptr<cc::CC_Compiler> ccc;
    if (basis.size() == 1) {
        ccc = std::make_unique<cc::Direct_CC_Compiler>(getEntryScope(), basis[0]);
    } else {
        ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(getEntryScope(), basis);
    }
    PabloAST * transitions = pb.createZeroes();
    PabloAST * idx = nullptr;
    if (mIndexStrm) {
        idx = getInputStreamSet("index")[0];
    }
    for (unsigned i = 0; i < mTransitionCCs.size(); i++) {
        PabloAST * trCC = ccc->compileCC(mTransitionCCs[i]);
        PabloAST * transition = pb.createAnd(pb.createIndexedAdvance(trCC, idx, 1), trCC);
        transitions = pb.createOr(transitions, transition);
    }
    PabloAST * mask = pb.createNot(transitions);
    if (mIndexStrm) {
        mask = pb.createAnd(mask, idx);
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("mask"), pb.getInteger(0)), mask);
}
