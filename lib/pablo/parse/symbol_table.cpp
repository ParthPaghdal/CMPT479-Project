/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <pablo/parse/symbol_table.h>

#include <pablo/builder.hpp>
#include <pablo/pabloAST.h>
#include <pablo/ps_assign.h>
#include <pablo/parse/error_text.h>
#include <pablo/parse/pablo_source_kernel.h>
#include <pablo/parse/pablo_type.h>
#include <pablo/parse/token.h>
#include <llvm/Support/ErrorHandling.h>

namespace pablo {
namespace parse {

SymbolTable::Entry::Entry()
: value(nullptr)
, token(nullptr)
, type(nullptr)
, attr(0)
, setInitStatus()
{}

SymbolTable::Entry::Entry(Entry const & other)
: value(other.value)
, token(other.token)
, type(other.type)
, attr(other.attr)
, setInitStatus(other.setInitStatus)
{}

SymbolTable::Entry::Entry(Entry && other)
: value(other.value)
, token(other.token)
, type(other.type)
, attr(std::move(other.attr))
, setInitStatus(std::move(other.setInitStatus))
{}

SymbolTable::Entry::Entry(PabloAST * value, Token * token, std::initializer_list<Attr> attr)
: value(value)
, token(token)
, type(nullptr)
, attr(0)
, setInitStatus()
{
    for (auto const & a : attr) {
        this->attr.set(a);
    }
}

SymbolTable::Entry & SymbolTable::Entry::operator = (Entry const & other) {
    this->value = other.value;
    this->token = other.token;
    this->type = other.type;
    this->attr = other.attr;
    return *this;
}

SymbolTable::Entry & SymbolTable::Entry::operator = (Entry && other) {
    this->value = other.value;
    this->token = other.token;
    this->type = other.type;
    this->attr = std::move(other.attr);
    return *this;
}


PabloAST * SymbolTable::assign(Token * token, PabloAST * value) {
    std::string name = token->getText();
    auto optEntry = find(name);
    if (optEntry) {
        Entry & e = *optEntry;
        Var * const var = llvm::cast<Var>(e.value);
        assert (var);
        Assign * const assign = mBuilder->createAssign(var, value);
        return llvm::cast<PabloAST>(assign);
    } else {
        Var * const var = mBuilder->createVar(token->getText(), value);
        Entry entry(var, token, {});
        mEntries.insert({name, std::move(entry)});
        return value;
    }
}


PabloAST * SymbolTable::indexedAssign(Token * token, Token * index, PabloAST * value) {
    std::string name = token->getText();
    auto optEntry = find(name);
    if (!optEntry) {
        mErrorManager->logError(token, errtxt_UseOfUndefinedSymbol(name));
        return nullptr;
    }
    Entry & e = optEntry.value();
    if (!e.attr[Entry::INDEXABLE]) {
        mErrorManager->logError(token, errtxt_NonIndexableSymbol(name));
        mErrorManager->logNote(e.token, errtxt_DefinitionNote(name));
        return nullptr;
    }
    uint64_t idx = 0;
    if (index->getType() != TokenType::INT_LITERAL) {
        if (!e.attr[Entry::DOT_INDEXABLE]) {
            mErrorManager->logError(token, errtxt_VarNotDotIndexable(name));
            return nullptr;
        }
        assert (e.type && llvm::isa<NamedStreamSetType>(e.type));
        auto streamset = llvm::cast<NamedStreamSetType>(e.type);
        bool found = false;
        for (auto const & n : streamset->getStreamNames()) {
            if (n == index->getText()) {
                found = true;
                break;
            }
            idx++;
        }
        if (!found) {
            mErrorManager->logError(index, errtxt_InvalidStreamName(name, index->getText()));
            return nullptr;
        }
    } else {
        idx = index->getValue();
    }
    if (isTopLevelScope()) {
        mEntries[name].setInitStatus[(size_t) idx] = true;
    }
    PabloAST * const var = e.value;
    assert (llvm::isa<Var>(var));
    Assign * const assign = mBuilder->createAssign(mBuilder->createExtract(llvm::cast<Var>(var), (int64_t) idx), value);
    return llvm::cast<PabloAST>(assign);
}


PabloAST * SymbolTable::lookup(Token * identifier) {
    assert (identifier->getType() == TokenType::IDENTIFIER);
    std::string name = identifier->getText();
    auto optEntry = find(name);
    if (!optEntry) {
        mErrorManager->logError(identifier, errtxt_UseOfUndefinedSymbol(name));
        return nullptr;
    }
    return optEntry.value().value;
}


PabloAST * SymbolTable::indexedLookup(Token * identifier, Token * index) {
    assert (identifier->getType() == TokenType::IDENTIFIER);
    std::string name = identifier->getText();
    auto optEntry = find(name);
    if (!optEntry) {
        mErrorManager->logError(identifier, errtxt_UseOfUndefinedSymbol(name));
        return nullptr;
    }
    Entry const & e = optEntry.value();
    if (!e.attr[Entry::INDEXABLE]) {
        mErrorManager->logError(identifier, errtxt_NonIndexableSymbol(name));
        mErrorManager->logNote(e.token, errtxt_DefinitionNote(name));
        return nullptr;
    }
    uint64_t idx = 0;
    if (index->getType() != TokenType::INT_LITERAL) {
        if (!e.attr[Entry::DOT_INDEXABLE]) {
            mErrorManager->logError(identifier, errtxt_VarNotDotIndexable(name));
            return nullptr;
        }
        assert (e.type && llvm::isa<NamedStreamSetType>(e.type));
        auto streamset = llvm::cast<NamedStreamSetType>(e.type);
        bool found = false;
        for (auto const & n : streamset->getStreamNames()) {
            if (n == index->getText()) {
                found = true;
                break;
            }
            idx++;
        }
        if (!found) {
            mErrorManager->logError(index, errtxt_InvalidStreamName(name, index->getText()));
            return nullptr;
        }
    } else {
        idx = index->getValue();
    }
    Var * const var = llvm::cast<Var>(e.value);
    return mBuilder->createExtract(var, idx);
}


void SymbolTable::addInputVar(Token * identifier, PabloType * type, PabloSourceKernel * kernel) {
    if (LLVM_UNLIKELY(llvm::isa<AliasType>(type))) {
        addInputVar(identifier, llvm::cast<AliasType>(type)->getAliasedType(), kernel);
        return;
    }
    std::string name = identifier->getText();
    PabloAST * var = nullptr;
    Entry e;
    if (llvm::isa<ScalarType>(type)) {
        mErrorManager->logFatalError(identifier, "input scalars are not supported in pablo kernels");
    } else if (llvm::isa<StreamType>(type)) {
        var = kernel->getInputStreamVar(name);
        e = Entry(var, identifier, {Entry::INPUT});
    } else if (llvm::isa<StreamSetType>(type)) {
        var = kernel->getInputStreamVar(name);
        e = Entry(var, identifier, {Entry::INPUT, Entry::INDEXABLE});
    } else if (llvm::isa<NamedStreamSetType>(type)) {
        var = kernel->getInputStreamVar(name);
        e = Entry(var, identifier, {Entry::INPUT, Entry::INDEXABLE, Entry::DOT_INDEXABLE});
    } else {
        llvm_unreachable(("invalid input type: " + type->asString(true)).c_str());
    }
    e.type = type;
    assert (e.value != nullptr && e.token != nullptr);
    auto rt = mEntries.insert({name, std::move(e)});
    if (!rt.second) {
        mErrorManager->logFatalError(identifier, errtxt_RedefinitionError(name));
        auto prev = mEntries[name];
        mErrorManager->logNote(prev.token, errtxt_PreviousDefinitionNote());
    }
}


void SymbolTable::addOutputVar(Token * identifier, PabloType * type, PabloSourceKernel * kernel) {
    if (LLVM_UNLIKELY(llvm::isa<AliasType>(type))) {
        addOutputVar(identifier, llvm::cast<AliasType>(type)->getAliasedType(), kernel);
        return;
    }
    std::string name = identifier->getText();
    PabloAST * var = nullptr;
    Entry e;
    if (llvm::isa<ScalarType>(type)) {
        var = kernel->getOutputScalarVar(identifier->getText());
        e = Entry(var, identifier, {Entry::OUTPUT});
    } else if (llvm::isa<StreamType>(type)) {
        var = kernel->getOutputStreamVar(identifier->getText());
        e = Entry(var, identifier, {Entry::OUTPUT});
    } else if (auto t = llvm::dyn_cast<StreamSetType>(type)) {
        var = kernel->getOutputStreamVar(identifier->getText());
        e = Entry(var, identifier, {Entry::OUTPUT, Entry::INDEXABLE});
        for (size_t i = 0; i < t->getStreamCount(); ++i) {
            e.setInitStatus.insert({i, false});
        }
    } else if (auto t = llvm::dyn_cast<NamedStreamSetType>(type)) {
        var = kernel->getOutputStreamVar(name);
        e = Entry(var, identifier, {Entry::OUTPUT, Entry::INDEXABLE, Entry::DOT_INDEXABLE});
        for (size_t i = 0; i < t->getStreamNames().size(); ++i) {
            e.setInitStatus.insert({i, false});
        }
    } else {
        llvm_unreachable(("invalid output type: " + type->asString(true)).c_str());
    }
    e.type = type;
    assert (e.value != nullptr && e.token != nullptr);
    auto rt = mEntries.insert({name, std::move(e)});
    if (!rt.second) {
        mErrorManager->logFatalError(identifier, errtxt_RedefinitionError(name));
        auto prev = mEntries[name];
        mErrorManager->logNote(prev.token, errtxt_PreviousDefinitionNote());
    }
}


boost::optional<SymbolTable::Entry> SymbolTable::localFind(std::string const & name) {
    auto it = mEntries.find(name);
    if (it == mEntries.end())
        return boost::none;
    return it->second;
}


boost::optional<SymbolTable::Entry> SymbolTable::higherFind(std::string const & name) {
    if (mParent == nullptr)
        return boost::none;
    return mParent->find(name);
}


boost::optional<SymbolTable::Entry> SymbolTable::find(std::string const & name) {
    auto e = localFind(name);
    if (e)
        return e.value();
    return higherFind(name);
}


bool SymbolTable::isTopLevelScope() const noexcept {
    return mParent == nullptr;
}


SymbolTable::SymbolTable(std::shared_ptr<ErrorManager> errorDelegate, PabloBuilder * pb)
: mErrorManager(std::move(errorDelegate))
, mBuilder(pb)
, mEntries()
, mParent(nullptr)
{
    assert (mErrorManager != nullptr);
    assert (mBuilder != nullptr);
}


SymbolTable::SymbolTable(std::shared_ptr<ErrorManager> errorDelegate, PabloBuilder * pb, SymbolTable * parent)
: mErrorManager(std::move(errorDelegate))
, mBuilder(pb)
, mEntries()
, mParent(parent)
{
    assert (mErrorManager != nullptr);
    assert (mBuilder != nullptr);
}

SymbolTable::~SymbolTable() {
    if (isTopLevelScope()) {
        // log an error if all output entries don't have top level values
        for (auto const & kv : mEntries) {
            auto e = kv.second;
            if (e.attr[Entry::OUTPUT]) {
                for (auto const & kv : e.setInitStatus) {
                    if (!kv.second) {
                        auto msg = "output variable '" 
                                 + e.token->getText() 
                                 + "[" + std::to_string(kv.first) + "]" 
                                 + "' may be uninitialized by the end of this kernel";
                        mErrorManager->logError(e.token, msg);
                    }
                }
            }
        }
    }
}

} // namespace pablo::parse
} // namespace pablo
