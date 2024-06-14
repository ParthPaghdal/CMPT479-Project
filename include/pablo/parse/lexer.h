/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <memory>
#include <vector>
#include <boost/optional.hpp>
#include <pablo/parse/error.h>
#include <pablo/parse/token.h>
#include <pablo/parse/source_file.h>

namespace pablo {
namespace parse {

/**
 * Base class for pablo lexer classes.
 */
class Lexer {
public:

    virtual ~Lexer() = default;

    /**
     * Converts an input stream of characters into a sequence of tokens. In the
     * event of a lexical error, boost::none is returned.
     * 
     * @param sourceFile A shared instance of the source file to tokenize.
     * @return A sequence of tokens or boost::none if a lexical error occurred.
     */
    virtual boost::optional<std::vector<Token *>> tokenize(std::shared_ptr<SourceFile> sourceFile) = 0;

};

} // namespace pablo::parse
} // namespace pablo
