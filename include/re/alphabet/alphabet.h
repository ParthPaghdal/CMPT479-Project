/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <string>
#include <vector>
#include <unicode/core/unicode_set.h>
#include <util/slab_allocator.h>

namespace cc {
    
enum class ByteNumbering {LittleEndian, BigEndian};

inline std::string numberingSuffix(ByteNumbering numbering) {
    return (numbering == ByteNumbering::LittleEndian) ? "-LE" : "-BE";
}

//
// An Alphabet is the universe of characters used to form strings in 
// a given language, together with a mapping of those characters to 
// numerical character codes.
//

class Alphabet {
public:
    using Allocator = SlabAllocator<Alphabet *>;
    const std::string & getName() const { return mAlphabetName;}
    const std::string & getCode() const { return mCode;}
    virtual const unsigned getSize() const = 0;
    enum class ClassTypeId : unsigned {UnicodeMappableAlphabet, CodeUnitAlphabet, MultiplexedAlphabet};
    inline ClassTypeId getClassTypeId() const {
        return mClassTypeId;
    }
    virtual ~Alphabet() {}
protected:
    Alphabet(const std::string name, const std::string code, ClassTypeId k) : mAlphabetName(name), mCode(code), mClassTypeId(k) {}
    void* operator new (std::size_t size) noexcept {
        return mAllocator.allocate<uint8_t>(size);
    }
    static Allocator mAllocator;
private:
    const std::string mAlphabetName;
    const std::string mCode;
    const ClassTypeId mClassTypeId;
};

class UnicodeMappableAlphabet final : public Alphabet {
public:
    //  Alphabets may be formed by some subset of Unicode characters, together
    //  with a mapping to and from Unicode.  The mapping is defined in terms of unicodeCommon:
    //  the number of character codes (if any) such that all character codes in the range
    //  0..unicodeCommon - 1 map to the same numeric value as the corresponding Unicode
    //  codepoint, together with a vector defining the Unicode codepoints for consecutive
    //  character codes (if any) above unicodeCommon - 1.
    
    UnicodeMappableAlphabet(const std::string alphabetName,
                            const std::string code,
                            unsigned unicodeCommon,
                            std::vector <UCD::codepoint_t> aboveCommon);

    static inline bool classof(const Alphabet * a) {
        return a->getClassTypeId() == ClassTypeId::UnicodeMappableAlphabet;
    }
    static inline bool classof(const void *) {return false;}
    //  The Unicode codepoint of the nth character (the character whose alphabet code is n).
    UCD::codepoint_t toUnicode(const unsigned n) const;
    
    //  The ordinal position of the character whose Unicode codepoint value is ucp.
    unsigned fromUnicode(const UCD::codepoint_t ucp) const;

    const unsigned getCommon() const {return mUnicodeCommon;}
    const unsigned getSize() const override {return mUnicodeCommon + mAboveCommon.size();}

protected:
    const UCD::codepoint_t mUnicodeCommon;
    const std::vector<UCD::codepoint_t> mAboveCommon;
};

class CodeUnitAlphabet final : public Alphabet {
public:
    CodeUnitAlphabet(const std::string name, const std::string code, uint8_t codeUnitBits);
    static inline bool classof(const Alphabet * a) {
        return a->getClassTypeId() == ClassTypeId::CodeUnitAlphabet;
    }
    static inline bool classof(const void *) {return false;}
    uint8_t getCodeUnitBitWidth() const { return mCodeUnitBits;}
    const unsigned getSize() const override {return 1<<mCodeUnitBits;}

private:
    const uint8_t mCodeUnitBits;
};

//  Some important alphabets are predefined.

const extern UnicodeMappableAlphabet Unicode; // Unicode("Unicode", "U", UCD::UNICODE_MAX, {})

const extern UnicodeMappableAlphabet ASCII;  // ASCII("ASCII", "A", 0x7F, {});

const extern UnicodeMappableAlphabet ISO_Latin1; // ISO_Latin1("ISO_Latin1", "l1", 0xFF, {});

const extern CodeUnitAlphabet Byte; // Byte("Byte", "x8", 8);
    
const extern CodeUnitAlphabet UTF8; // UTF8("UTF8", "u8", 8);

const extern CodeUnitAlphabet UTF16; // UTF16("UTF16", "u16", 16);
    
}

