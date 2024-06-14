/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <vector>
#include <boost/filesystem.hpp>
#include <re/parse/parser.h>
#include <re/parse/ERE_parser.h>

//  GLOB parsing for Filename expansion in accord with Posix rules
//  IEEE-1003.1 XCU section  2.13 Pattern Matching Notation

namespace re {
    enum class GLOB_kind {Posix, GIT};
    class FileGLOB_Parser : public RE_Parser  {
    public:
        FileGLOB_Parser(const std::string & glob, GLOB_kind k = GLOB_kind::Posix) : RE_Parser(glob),
            mGLOB_kind(k), mPathComponentStartContext(true) {
            mReSyntax = RE_Syntax::FileGLOB;
        }

    protected:
        RE * parse_alt() override;
        RE * parse_seq() override;
        RE * parse_next_item() override;
        RE * parse_bracket_expr();
        RE * range_extend(RE * e1);
    private:
        GLOB_kind mGLOB_kind;
        bool mPathComponentStartContext;
    };


enum class PatternKind {Include, Exclude};
using PatternType = std::pair<PatternKind, RE *>;
using PatternVector = std::vector<PatternType>;

PatternVector parseGitIgnoreFile(boost::filesystem::path dirpath,
                                                             std::string ignoreFileName);

}
