/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once


#include <stdint.h>
#include <pablo/builder.hpp>
#include <re/adt/re_cc.h>
#include <re/alphabet/alphabet.h>

namespace cc {

class CC_Compiler {
public:
    virtual pablo::PabloAST * compileCC(const re::CC *cc) {
        return compileCC(cc, mBuilder);
    }
    virtual pablo::PabloAST * compileCC(const re::CC *cc, pablo::PabloBlock & block) {
        return compileCC(cc->canonicalName(), cc, block);
    }
    virtual pablo::PabloAST * compileCC(const re::CC *cc, pablo::PabloBuilder & builder) {
        return compileCC(cc->canonicalName(), cc, builder);
    }
    virtual pablo::PabloAST * compileCC(const std::string & canonicalName, const re::CC *cc, pablo::PabloBlock & block) = 0;
    virtual pablo::PabloAST * compileCC(const std::string & canonicalName, const re::CC *cc, pablo::PabloBuilder & builder) = 0;
    virtual ~CC_Compiler(){}

protected:
    CC_Compiler(pablo::PabloBlock * scope);
    pablo::PabloBuilder             mBuilder;
};


class CC_Compiler_Common {
public:
    ~CC_Compiler_Common() {}

protected:
    CC_Compiler_Common(unsigned encodingBits, std::vector<pablo::PabloAST *> basisBit, unsigned encodingMask);

protected:
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * getBasisVar(const unsigned n, PabloBlockOrBuilder & pb) const;
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * bit_pattern_expr(const unsigned pattern, unsigned selected_bits, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * char_test_expr(const re::codepoint_t ch, PabloBlockOrBuilder & pb);

protected:  
    const unsigned                  mEncodingBits;
    std::vector<pablo::PabloAST *>  mBasisBit;
    unsigned                        mEncodingMask;
};
    
    
class Parabix_CC_Compiler : public CC_Compiler, public CC_Compiler_Common {
public:
    using CC_Compiler::compileCC;

    Parabix_CC_Compiler(pablo::PabloBlock * scope, std::vector<pablo::PabloAST *> basisBitSet);
    pablo::PabloAST * compileCC(const std::string & name, const re::CC *cc, pablo::PabloBlock & block) override;
    pablo::PabloAST * compileCC(const std::string & name, const re::CC *cc, pablo::PabloBuilder & builder) override;
    ~Parabix_CC_Compiler() {}

private:
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * charset_expr(const re::CC *cc, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * make_range(const re::codepoint_t n1, const re::codepoint_t n2, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * char_or_range_expr(const re::codepoint_t lo, const re::codepoint_t hi, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * GE_Range(const unsigned N, const unsigned n, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * LE_Range(const unsigned N, const unsigned n, PabloBlockOrBuilder & pb);
};
    

class Direct_CC_Compiler : public CC_Compiler {
public:
    using CC_Compiler::compileCC;

    Direct_CC_Compiler(pablo::PabloBlock * scope, pablo::PabloAST * codeUnitStream);
    pablo::PabloAST * compileCC(const std::string & name, const re::CC *cc, pablo::PabloBlock & block) override;
    pablo::PabloAST * compileCC(const std::string & name, const re::CC *cc, pablo::PabloBuilder & builder) override;
    ~Direct_CC_Compiler() {}
    
private:
    pablo::PabloAST *               mCodeUnitStream;
};


class Parabix_Ternary_CC_Compiler : public CC_Compiler, public CC_Compiler_Common {
public:
    using CC_Compiler::compileCC;
    using ClassTypeId = pablo::PabloAST::ClassTypeId;
    using octet_pair_t = std::pair<re::codepoint_t, uint8_t>;
    using octets_intervals_union_t = std::pair<std::vector<octet_pair_t>, std::vector<re::interval_t>>;

    Parabix_Ternary_CC_Compiler(pablo::PabloBlock * scope, std::vector<pablo::PabloAST *> basisBitSet);
    pablo::PabloAST * compileCC(const std::string & name, const re::CC *cc, pablo::PabloBlock & block) override;
    pablo::PabloAST * compileCC(const std::string & name, const re::CC *cc, pablo::PabloBuilder & builder) override;
    ~Parabix_Ternary_CC_Compiler() {}
private:
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * charset_expr(const re::CC *cc, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * make_octets_expr(const std::vector<octet_pair_t> octets, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * make_intervals_expr(const std::vector<re::interval_t> intervals, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * make_range(re::codepoint_t lo, re::codepoint_t hi, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * make_octet_range(const unsigned mask, const unsigned basis_idx, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * char_or_range_expr(const re::codepoint_t lo, const re::codepoint_t hi, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * bit_pattern_expr(const unsigned pattern, const unsigned basis_idx, unsigned selected_bits, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * joinTerms(std::vector<pablo::PabloAST *> terms, ClassTypeId ty, PabloBlockOrBuilder & pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * GE_Range(const unsigned N, const unsigned n, PabloBlockOrBuilder &pb);
    template<typename PabloBlockOrBuilder>
    pablo::PabloAST * LE_Range(const unsigned N, const unsigned n, PabloBlockOrBuilder &pb);
    octets_intervals_union_t make_octets_intervals_union(const re::CC *cc);
    inline re::codepoint_t octet_base_codepoint(const octet_pair_t & i) {
        return std::get<0>(i);
    }
    inline uint8_t octet_mask(const octet_pair_t & i) {
        return std::get<1>(i);
    }
    inline std::vector<octet_pair_t> octets_in_union(const octets_intervals_union_t & i) {
        return std::get<0>(i);
    }
    inline std::vector<re::interval_t> intervals_in_union(const octets_intervals_union_t & i) {
        return std::get<1>(i);
    }
};


pablo::PabloAST * compileCCfromCodeUnitStream(const re::CC *cc, pablo::PabloAST * codeUnitStream, pablo::PabloBuilder & pb);
    
}

