/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <string>                // for string
#include <kernel/core/kernel_builder.h>
#include <pablo/pablo_toolchain.h>

namespace kernel {

class UTF8_Decoder final: public pablo::PabloKernel {
public:
UTF8_Decoder (KernelBuilder & b, StreamSet * u8_basis, StreamSet * unicode_basis,
              pablo::BitMovementMode m = pablo::BitMovementMode::Advance)
    : PabloKernel(b, "UTF8_Decoder_" + pablo::BitMovementMode_string(m), {},
                  {Binding{"unicode_bit", unicode_basis}}), mBitMovement(m) {
        
        if (m == pablo::BitMovementMode::LookAhead) {
            mInputStreamSets.push_back(Binding{"utf8_bit", u8_basis, FixedRate(1), LookAhead(3)});
        } else {
            mInputStreamSets.push_back(Binding{"utf8_bit", u8_basis});
        }
    }

    void generatePabloMethod() override;
private:
    pablo::BitMovementMode mBitMovement;
};

}
