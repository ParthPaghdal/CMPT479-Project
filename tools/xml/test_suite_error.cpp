/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include "test_suite_error.h"

#include <cstddef>
#include <cassert>
#include <mutex>
#include <string>
#include <queue>
#include <vector>

template<typename T, class Compare>
class atomic_priority_queue {
public:
    atomic_priority_queue() = default;
    void push(T const & t) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(t);
    }
    T pop() {
        assert (!m_queue.empty());
        std::lock_guard<std::mutex> lock(m_mutex);
        T val = m_queue.top();
        m_queue.pop();
        return val;
    }
    bool empty() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }
private:
    std::mutex m_mutex;
    std::priority_queue<T, std::vector<T>, Compare> m_queue;
};

struct TestSuiteError {
    XmlTestSuiteError   code;
    uint64_t            line;       // mutually exclusive with position
    uint64_t            column;     // mutually exclusive with position
    uint64_t            position;   // mutually exclusive with line+column

    TestSuiteError(XmlTestSuiteError code, uint64_t line, uint64_t column)
    : code(code), line(line), column(column), position(0) {}

    TestSuiteError(XmlTestSuiteError code, uint64_t position)
    : code(code), line(UINT64_MAX), column(UINT64_MAX), position(position) {}

    struct Comparator {
        bool operator() (TestSuiteError const & lhs, TestSuiteError const & rhs) const {
            // prioritize declaration parse errors above anything else
            if (lhs.code == XmlTestSuiteError::DECLARATION_PARSE_ERROR) {
                return false;
            } else if (lhs.code == XmlTestSuiteError::DECLARATION_PARSE_ERROR) {
                return true;
            }
            if (lhs.line == rhs.line) {
                if (lhs.column == rhs.column) {
                    return lhs.position > rhs.position;
                }
                return lhs.column > rhs.column;
            }
            return lhs.line > rhs.line;
        }
    };
};


static atomic_priority_queue<TestSuiteError, TestSuiteError::Comparator> ErrorQueue{};

const char * AsMessage(XmlTestSuiteError error) {
    switch (error) {
        case XmlTestSuiteError::NAME_START:
            return "Name start error";
        case XmlTestSuiteError::NAME:
            return "Name error";
        case XmlTestSuiteError::XML_PI_NAME:
            return "[Xx][Mm][Ll] illegal as PI name";
        case XmlTestSuiteError::CDATA:
            return "CDATA error";
        case XmlTestSuiteError::UNDEFREF:
            return "Undefined reference";
        case XmlTestSuiteError::CHARREF:
            return "Illegal character reference";
        case XmlTestSuiteError::XML10CHARREF:
            return "Illegal XML 1.0 character reference";
        case XmlTestSuiteError::ATTREF:
            return "Attribute values contain '<' characters after reference expansion";
        case XmlTestSuiteError::ILLEGAL_CHAR:
            return "Error: illegal character";
        case XmlTestSuiteError::UTF8_ERROR:
            return "UTF-8 error found";
        case XmlTestSuiteError::PI_SYNTAX:
            return "Error in PI syntax";
        case XmlTestSuiteError::COMMENT:
            return "Error in comment syntax";
        case XmlTestSuiteError::PI_CD_CT_ERROR:
            return "Error in comment, CDATA or processing instruction syntax";
        case XmlTestSuiteError::TAG:
            return "Tag parsing error found";
        case XmlTestSuiteError::REF:
            return "Reference error found";
        case XmlTestSuiteError::NAME_SYNTAX:
            return "Name syntax error";
        case XmlTestSuiteError::CD_CLOSER:
            return "Error: ]]> in text";
        case XmlTestSuiteError::TAG_NAME_MISMATCH:
            return "Tag name mismatch";
        case XmlTestSuiteError::TAG_MATCH_ERROR:
            return "Tag matching error";
        case XmlTestSuiteError::CONTENT_BEFORE_ROOT:
            return "Illegal content before root element";
        case XmlTestSuiteError::CONTENT_AFTER_ROOT:
            return "Illegal content after root element";
        case XmlTestSuiteError::DUPLICATE_ATTR_NAME:
            return "Attribute name is not unique";
        case XmlTestSuiteError::DECLARATION_PARSE_ERROR:
            return "Parsing error at position %llu in XML or Text declaration.\n";
        case XmlTestSuiteError::NOT_UTF8:
            return "Sorry, this xmlwf demo only works for UTF-8.\n";
        default:
            assert ("unexpected error xml test suite error" && false);
            return "Invalid XML Error Code";
    }
}

void ReportError(XmlTestSuiteError code, const uint8_t * ptr, const uint8_t * lineBegin, const uint8_t * /*lineEnd*/, uint64_t lineNumber) {
    ptrdiff_t column = ptr - lineBegin;
    assert (column >= 0);
    column += 1; // convert from 0-indexed to 1-indexed column
    TestSuiteError err{code, lineNumber, (uint64_t) column};
    ErrorQueue.push(err);
}

void ReportError(XmlTestSuiteError code, uint64_t position) {
    ErrorQueue.push(TestSuiteError{code, position});
}

// #define SHOW_ALL_ERRORS

void ShowError() {
    if (ErrorQueue.empty()) {
        return;
    }
#ifdef SHOW_ALL_ERRORS
    while (!ErrorQueue.empty()) {
#endif
    auto err = ErrorQueue.pop();
    switch (err.code) {
        case XmlTestSuiteError::TAG_NAME_MISMATCH:
        case XmlTestSuiteError::TAG_MATCH_ERROR:
        case XmlTestSuiteError::CONTENT_BEFORE_ROOT:
        case XmlTestSuiteError::CONTENT_AFTER_ROOT:
        case XmlTestSuiteError::DUPLICATE_ATTR_NAME:
            fprintf(stderr, "%s at position = %" PRIu64 "\n", AsMessage(err.code), err.position);
            break;
        case XmlTestSuiteError::DECLARATION_PARSE_ERROR:
            fprintf(stderr, AsMessage(err.code), err.position);
            break;
        case XmlTestSuiteError::NOT_UTF8:
            fputs(AsMessage(err.code), stderr);
            break;
        default:
            fprintf(stderr, "%s at line %" PRIu64 ", column %" PRIu64 "\n", AsMessage(err.code), err.line, err.column);
            break;
#ifdef SHOW_ALL_ERRORS
    }
#endif
    }
}
