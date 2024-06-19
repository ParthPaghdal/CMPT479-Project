#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <chrono>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include "idisa_builder.h"

class ProgramBuilder {
public:
    StreamSet* CreateStreamSet(unsigned numElements, unsigned fieldWidth = 1) {
        return new StreamSet(numElements, fieldWidth);
    }

    template <typename Kernel, typename... Args>
    void CreateKernelCall(Args... args) {

    }
};

class StreamSet {
public:
    StreamSet(size_t size, size_t fieldWidth) : data(size), size(size), fieldWidth(fieldWidth) {}

    uint8_t* getData() { return data.data(); }
    size_t getSize() const { return size; }
    size_t getFieldWidth() const { return fieldWidth; }

private:
    std::vector<uint8_t> data;
    size_t size;
    size_t fieldWidth;
};


struct S2PKernel {};
struct FieldCompressKernel {};
struct StreamCompressKernel {};
struct P2SKernel {};

// Bitwise FilterByMask implementation
void FilterByMask(const std::unique_ptr<ProgramBuilder> &P, StreamSet *mask, StreamSet *inputs, StreamSet *outputs, unsigned streamOffset, unsigned extractionFieldWidth, bool byteDeletion) {
    if (byteDeletion) {
        StreamSet * const input_streams = P->CreateStreamSet(8);
        P->CreateKernelCall<S2PKernel>(inputs, input_streams);
        StreamSet * const output_streams = P->CreateStreamSet(8);
        StreamSet * const compressed = P->CreateStreamSet(output_streams->getNumElements());
        std::vector<uint32_t> output_indices = streamutils::Range(streamOffset, streamOffset + output_streams->getNumElements());
        P->CreateKernelCall<FieldCompressKernel>(Select(mask, {0}), SelectOperationList { Select(input_streams, output_indices)}, compressed, extractionFieldWidth);
        P->CreateKernelCall<StreamCompressKernel>(mask, compressed, output_streams, extractionFieldWidth);
        P->CreateKernelCall<P2SKernel>(output_streams, outputs);
    } else {
        StreamSet * const compressed = P->CreateStreamSet(outputs->getNumElements());
        std::vector<uint32_t> output_indices = streamutils::Range(streamOffset, streamOffset + outputs->getNumElements());
        P->CreateKernelCall<FieldCompressKernel>(Select(mask, {0}), SelectOperationList { Select(inputs, output_indices)}, compressed, extractionFieldWidth);
        P->CreateKernelCall<StreamCompressKernel>(mask, compressed, outputs, extractionFieldWidth);
    }
}

// Helper function to generate test cases
void generateTestCase(size_t k, std::vector<uint8_t> &input, std::vector<uint8_t> &mask, std::vector<uint8_t> &expectedOutput) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    std::uniform_int_distribution<> bit_dis(0, 1);

    input.resize(k);
    mask.resize(k);

    for (size_t i = 0; i < k; ++i) {
        input[i] = static_cast<uint8_t>(dis(gen));
        mask[i] = static_cast<uint8_t>(bit_dis(gen));
    }

    for (size_t i = 0; i < k; ++i) {
        if (mask[i] != 0) {
            expectedOutput.push_back(input[i]);
        }
    }
}

void testFilterByMask(const std::vector<uint8_t> &input, const std::vector<uint8_t> &mask, const std::vector<uint8_t> &expectedOutput) {
    // Test byte-oriented FilterByMask
    {
        llvm::LLVMContext context;
        IDISABuilder idisaBuilder(context);
        llvm::Function *func = idisaBuilder.createByteFilterByMaskFunction();
        llvm::verifyFunction(*func, &llvm::errs());
        idisaBuilder.getModule().print(llvm::outs(), nullptr);

        // Simulate byte-oriented filter (integration with actual execution environment required)
        auto start = std::chrono::high_resolution_clock::now();
        // Call to the byte-oriented filter function (not executable in this form)
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Byte-oriented FilterByMask elapsed time: " << elapsed.count() << " seconds." << std::endl;
    }

    // Test bit-oriented FilterByMask
    {
        auto P = std::make_unique<ProgramBuilder>();
        StreamSet *maskStream = P->CreateStreamSet(mask.size(), 1);
        StreamSet *inputStream = P->CreateStreamSet(input.size(), 1);
        StreamSet *outputStream = P->CreateStreamSet(input.size(), 1);

        // Copy data to StreamSets
        std::memcpy(maskStream->getData(), mask.data(), mask.size());
        std::memcpy(inputStream->getData(), input.data(), input.size());

        // Measure execution time of bit-oriented FilterByMask
        auto start = std::chrono::high_resolution_clock::now();
        FilterByMask(P, maskStream, inputStream, outputStream, 0, 64, false);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Bit-oriented FilterByMask elapsed time: " << elapsed.count() << " seconds." << std::endl;

        // Collect output
        std::vector<uint8_t> output(outputStream->getData(), outputStream->getData() + outputStream->getSize());

        // Compare output with expected output
        bool passed = (output == expectedOutput);
        if (passed) {
            std::cout << "Bit-oriented FilterByMask test passed." << std::endl;
        } else {
            std::cout << "Bit-oriented FilterByMask test failed." << std::endl;
        }

        // Clean up
        delete maskStream;
        delete inputStream;
        delete outputStream;
    }
}

int main() {
    size_t k = 1000000; // Length of the test data for performance testing
    std::vector<uint8_t> input;
    std::vector<uint8_t> mask;
    std::vector<uint8_t> expectedOutput;

    generateTestCase(k, input, mask, expectedOutput);

    testFilterByMask(input, mask, expectedOutput);

    return 0;
}

