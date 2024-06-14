/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <ostream>
#include <memory>
#include <vector>
#include <boost/optional.hpp>
#include <pablo/pablo_kernel.h>
#include <pablo/builder.hpp>
#include <pablo/parse/pablo_source_kernel.h>
#include <pablo/parse/kernel_signature.h>

namespace pablo {
namespace parse {

class ErrorManager;
class SourceFile;

/**
 * Abstract interface for all pablo parser implementations.
 */
class PabloParser {
public:

    virtual ~PabloParser() = default;

    /**
     * Parses a specific kernel body from a specified source file. In the event
     * of a parse error, `false` will be returned and error information can be
     * retrivied through the parser's `ErrorManager` delegate.
     * 
     * This method only parses the definition of the requested kernel and not
     * the whole source file.
     * 
     * @param sourceFile    A pointer to the source file to parse.
     * @param kernel        An instance of the kernel to populate.
     * @param kernelName    The name of the kernel to parse in the source file.
     * @return `true` iff parse was successful, `false` otherwise.
     */
    virtual bool parseKernel(std::shared_ptr<SourceFile> sourceFile, PabloSourceKernel * kernel, std::string const & kernelName) = 0;

    /**
     * Returns a pointer to this parser's error manager delegate.
     */
    virtual std::shared_ptr<ErrorManager> getErrorManager() const = 0;

    /**
     * Convenience parse method. Takes a filename instead of a `SourceFile`.
     * 
     * @param filename      The name of the file to open and parse.
     * @param kernel        An instance of the kernel to populate.
     * @param kernelName    The name of the kernel to parse.
     * @return `true` iff parse was successful, `false` otherwise.
     */
    virtual bool parseKernel(std::string const & filename, PabloSourceKernel * kernel, std::string const & kernelName);

};

} // namespace pablo::parse
} // namespace pablo
