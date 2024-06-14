#pragma once

#include <stdint.h>
#include <boost/integer.hpp>

namespace kernel {


struct StreamSetPtr {

    // TODO: if we're returning a bitstream, we cannot actually return a single bit value.
    // should this instead return a bitblock pointer type?

    template<unsigned FieldWidth>
    using datatype_t = typename boost::uint_t<FieldWidth>::exact;

    template<unsigned FieldWidth = 8>
    datatype_t<FieldWidth> * data(const uint64_t position = 0) const {
        return static_cast<datatype_t<FieldWidth> *>(__base) + position;
    }

    uint64_t length() const {
        return __length;
    }

    StreamSetPtr() : __base(nullptr), __length(0) {}

    StreamSetPtr(void * base, uint64_t length) : __base(base), __length(length) {}

private:
    void * __base;
    uint64_t __length;
};




}

