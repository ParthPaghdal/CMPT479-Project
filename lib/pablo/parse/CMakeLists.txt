#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: pablo.parse

parabix_add_module(
NAME
    pablo.parse
SRC
    error.cpp
    kernel_signature.cpp
    pablo_parser.cpp
    pablo_source_kernel.cpp
    pablo_type.cpp
    rd_parser.cpp
    simple_lexer.cpp
    source_file.cpp
    symbol_table.cpp
    token.cpp
DEPS
    pablo
)
