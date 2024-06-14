#include <kernel/core/binding.h>

#include <kernel/core/relationship.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/DerivedTypes.h>

const auto NULL_RELATIONSHIP_ERROR = "cannot set binding relationship to null without a fixed binding type";

const auto NON_MATCHING_TYPE_ERROR = "value type did not match the given binding type";

const auto NOT_STREAM_SET = "binding relationship does not refer to a stream set type";

namespace kernel {

Binding::Binding(std::string name, Relationship * const value, ProcessingRate r)
: AttributeSet()
, mName(std::move(name))
, mRate(std::move(r))
, mType(value->getType())
, mRelationship(value)
, mDistribution(UniformDistribution()) {

}

Binding::Binding(std::string name, Relationship * const value, ProcessingRate r, Attribute && attribute)
: AttributeSet(std::move(attribute))
, mName(std::move(name))
, mRate(std::move(r))
, mType(value->getType())
, mRelationship(value)
, mDistribution(UniformDistribution()) {

}

Binding::Binding(std::string name, Relationship * const value, ProcessingRate r, std::initializer_list<Attribute> attributes)
: AttributeSet(attributes)
, mName(std::move(name))
, mRate(std::move(r))
, mType(value->getType())
, mRelationship(value)
, mDistribution(UniformDistribution()) {

}

Binding::Binding(llvm::Type * const scalarType, std::string name, ProcessingRate r)
: AttributeSet()
, mName(std::move(name))
, mRate(std::move(r))
, mType(scalarType)
, mRelationship(nullptr)
, mDistribution(UniformDistribution()) {

}

Binding::Binding(llvm::Type * const scalarType, std::string name, ProcessingRate r, Attribute && attribute)
: AttributeSet(std::move(attribute))
, mName(std::move(name))
, mRate(std::move(r))
, mType(scalarType)
, mRelationship(nullptr)
, mDistribution(UniformDistribution()) {

}

Binding::Binding(llvm::Type * const scalarType, std::string name, ProcessingRate r, std::initializer_list<Attribute> attributes)
: AttributeSet(attributes)
, mName(std::move(name))
, mRate(std::move(r))
, mType(scalarType)
, mRelationship(nullptr)
, mDistribution(UniformDistribution()) {

}

Binding::Binding(llvm::Type * const type, std::string name, Relationship * const value, ProcessingRate r)
: AttributeSet()
, mName(std::move(name))
, mRate(std::move(r))
, mType(type)
, mRelationship(value)
, mDistribution(UniformDistribution()) {
    if (LLVM_UNLIKELY(value == nullptr && type == nullptr)) {
        llvm::report_fatal_error(NULL_RELATIONSHIP_ERROR);
    }
    if (LLVM_UNLIKELY(type && value && value->getType() != type)) {
        llvm::report_fatal_error(NON_MATCHING_TYPE_ERROR);
    }
}

Binding::Binding(llvm::Type * const type, std::string name, Relationship * const value, ProcessingRate r, Attribute && attribute)
: AttributeSet(std::move(attribute))
, mName(std::move(name))
, mRate(std::move(r))
, mType(type)
, mRelationship(value)
, mDistribution(UniformDistribution()) {
    if (LLVM_UNLIKELY(value == nullptr && type == nullptr)) {
        llvm::report_fatal_error(NULL_RELATIONSHIP_ERROR);
    }
    if (LLVM_UNLIKELY(type && value && value->getType() != type)) {
        llvm::report_fatal_error(NON_MATCHING_TYPE_ERROR);
    }
}

Binding::Binding(llvm::Type * const type, std::string name, Relationship * const value, ProcessingRate r, std::initializer_list<Attribute> attributes)
: AttributeSet(attributes)
, mName(std::move(name))
, mRate(std::move(r))
, mType(type)
, mRelationship(value)
, mDistribution(UniformDistribution()) {
    if (LLVM_UNLIKELY(value == nullptr && type == nullptr)) {
        llvm::report_fatal_error(NULL_RELATIONSHIP_ERROR);
    }
    if (LLVM_UNLIKELY(type && value && value->getType() != type)) {
        llvm::report_fatal_error(NON_MATCHING_TYPE_ERROR);
    }
}

Binding::Binding(const Binding & original, ProcessingRate r)
: AttributeSet(original.getAttributes())
, mName(original.getName())
, mRate(std::move(r))
, mType(original.getType())
, mRelationship(original.getRelationship())
, mDistribution(original.getDistribution()) {

}

void Binding::setRelationship(Relationship * const value) {
    if (LLVM_UNLIKELY(value == nullptr && mType == nullptr)) {
        llvm::report_fatal_error(NULL_RELATIONSHIP_ERROR);
    }
    if (LLVM_UNLIKELY(mType && value && value->getType() != mType)) {
        llvm::report_fatal_error(NON_MATCHING_TYPE_ERROR);
    }
    mRelationship = value;
}

unsigned Binding::getNumElements() const {
    if (LLVM_UNLIKELY(mRelationship->isScalar())) {
        llvm::report_fatal_error(NOT_STREAM_SET);
    }
    return static_cast<const StreamSet *>(mRelationship)->getNumElements();
}

unsigned Binding::getFieldWidth() const {
    if (LLVM_UNLIKELY(mRelationship->isScalar())) {
        llvm::report_fatal_error(NOT_STREAM_SET);
    }
    return static_cast<const StreamSet *>(mRelationship)->getFieldWidth();
}

void Binding::print(const Kernel * kernel, llvm::raw_ostream & out) const noexcept {
    mRate.print(kernel, out);
    AttributeSet::print(out);
    assert ("binding missing type?" && getType());
    getType()->print(out);
}

}
