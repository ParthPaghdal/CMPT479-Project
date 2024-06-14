#include <kernel/pipeline/pipeline_kernel.h>
#include <toolchain/toolchain.h>
#include "compiler/pipeline_compiler.hpp"
#include <llvm/IR/Function.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/core/streamset.h>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(15, 0, 0)
#include <llvm/Analysis/ConstantFolding.h>
#endif
#ifdef ENABLE_PAPI
#include <papi.h>
#include <boost/tokenizer.hpp>
#endif


// NOTE: the pipeline kernel is primarily a proxy for the pipeline compiler. Ideally, by making some kernels
// a "family", the pipeline kernel will be compiled once for the lifetime of a program. Thus we can avoid even
// constructing any data structures for the pipeline in normal usage.

using boost::intrusive::detail::is_pow2;

using IDISA::FixedVectorType;

namespace kernel {

#define COMPILER (static_cast<PipelineCompiler *>(b.getCompiler()))

#ifdef ENABLE_PAPI
/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializePAPI
 ** ------------------------------------------------------------------------------------------------------------- */
int initializePAPI(SmallVector<int, 8> & PAPIEventList) {

    const int rvalInit = PAPI_library_init(PAPI_VER_CURRENT);
    if (rvalInit != PAPI_VER_CURRENT) {
        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "PAPI Library Init Error: ";
        out << PAPI_strerror(rvalInit);
        report_fatal_error(StringRef(out.str()));
    }

    //    if (codegen::SegmentThreads > 1 || codegen::EnableDynamicMultithreading) {
            const auto rvalThreaedInit = PAPI_thread_init(pthread_self);
            if (rvalThreaedInit != PAPI_OK) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                out << "PAPI Thread Init Error: ";
                out << PAPI_strerror(rvalThreaedInit);
                report_fatal_error(StringRef(out.str()));
            }
    //    }


    assert (!codegen::PapiCounterOptions.empty());

    tokenizer<escaped_list_separator<char>> events(codegen::PapiCounterOptions);
    for (const auto & event : events) {
        int EventCode = PAPI_NULL;
        const int rvalEventNameToCode = PAPI_event_name_to_code(const_cast<char*>(event.c_str()), &EventCode);
        if (LLVM_LIKELY(rvalEventNameToCode == PAPI_OK)) {
            PAPIEventList.push_back(EventCode);
        } else {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "PAPI Library cannot resolve event name: ";
            out << event.c_str();
            out << "\n";
            out << PAPI_strerror(rvalEventNameToCode);
            report_fatal_error(StringRef(out.str()));
        }
    }

    // sanity test whether this event set is valid
    int EventSet = PAPI_NULL;
    const auto rvalCreateEventSet = PAPI_create_eventset(&EventSet);
    if (rvalCreateEventSet != PAPI_OK) {
        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "PAPI Create Event Set Error: ";
        out << PAPI_strerror(rvalCreateEventSet);
        report_fatal_error(StringRef(out.str()));
    }

    const auto rvalAddEvents = PAPI_add_events(EventSet, PAPIEventList.data(), (int)PAPIEventList.size());

    if (rvalAddEvents != PAPI_OK) {
        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "PAPI Add Events Error: ";
        out << PAPI_strerror(rvalCreateEventSet < PAPI_OK ? rvalCreateEventSet : PAPI_EINVAL);
        out << "\n"
               "Check papi_avail for available options or enter sysctl -w kernel.perf_event_paranoid=0\n"
               "to reenable cpu event tracing at the kernel level.";
        report_fatal_error(StringRef(out.str()));
    }

    const auto rvalStart = PAPI_start(EventSet);
    if (rvalAddEvents != PAPI_OK) {
        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "PAPI Start Error: ";
        out << PAPI_strerror(rvalCreateEventSet < PAPI_OK ? rvalCreateEventSet : PAPI_EINVAL);
        report_fatal_error(StringRef(out.str()));
    }

    return EventSet;
}

void terminatePAPI(KernelBuilder & b, Value * eventSet) {
    Module * const m = b.getModule();
    FixedArray<Value *, 1> args;
    args[0] = eventSet;
    Function * const PAPICleanupEventsetFn = m->getFunction("PAPI_cleanup_eventset");
    b.CreateCall(PAPICleanupEventsetFn->getFunctionType(), PAPICleanupEventsetFn, args);
    Function * const PAPIDestroyEventsetFn = m->getFunction("PAPI_destroy_eventset");

    FunctionType * fTy = PAPIDestroyEventsetFn->getFunctionType();

//    fTy->getFunctionParamType(0)->isPointerTy()
    Value * eventSetData = b.CreateAllocaAtEntryPoint(eventSet->getType());
    b.CreateStore(eventSet, eventSetData);
    args[0] = eventSetData;

    b.CreateCall(fTy, PAPIDestroyEventsetFn, args);
    Function * const PAPIShutdownFn = m->getFunction("PAPI_shutdown");
    b.CreateCall(PAPIShutdownFn->getFunctionType(), PAPIShutdownFn, {});
}
#endif

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addInternalKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::addInternalProperties(KernelBuilder & b) {
    COMPILER->generateImplicitKernels(b);
    COMPILER->addPipelineKernelProperties(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateInitializeMethod(KernelBuilder & b) {
    COMPILER->generateInitializeMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateInitializeThreadLocalMethod(KernelBuilder & b) {
    COMPILER->generateInitializeThreadLocalMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateKernelMethod(KernelBuilder & b) {
    COMPILER->generateKernelMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateFinalizeMethod(KernelBuilder & b) {
    COMPILER->generateFinalizeMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateFinalizeThreadLocalMethod(KernelBuilder & b) {
    COMPILER->generateFinalizeThreadLocalMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addKernelDeclarations
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::addKernelDeclarations(KernelBuilder & b) {
    for (const auto & k : mKernels) {
        k.Object->addKernelDeclarations(b);
    }
    Kernel::addKernelDeclarations(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasInternalStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineKernel::allocatesInternalStreamSets() const {
    return true;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateSharedInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateAllocateSharedInternalStreamSetsMethod(KernelBuilder & b, Value * expectedNumOfStrides) {
    COMPILER->generateAllocateSharedInternalStreamSetsMethod(b, expectedNumOfStrides);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateThreadLocalInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateAllocateThreadLocalInternalStreamSetsMethod(KernelBuilder & b, Value * expectedNumOfStrides) {
    COMPILER->generateAllocateThreadLocalInternalStreamSetsMethod(b, expectedNumOfStrides);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkExternalMethods
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::linkExternalMethods(KernelBuilder & b) {
    PipelineCompiler::linkPThreadLibrary(b);
    for (const auto & k : mKernels) {
        k.Object->linkExternalMethods(b);
    }
    for (const CallBinding & call : mCallBindings) {
        call.Callee = b.LinkFunction(call.Name, call.Type, call.FunctionPointer);
    }
    #ifdef ENABLE_PAPI
    if (LLVM_UNLIKELY(codegen::PapiCounterOptions.compare(codegen::OmittedOption) != 0)) {
        PipelineCompiler::linkPAPILibrary(b);
    }
    #endif
    StreamSetBuffer::linkFunctions(b);
    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet())) {
        PipelineCompiler::linkInstrumentationFunctions(b);
        PipelineCompiler::linkHistogramFunctions(b);
        PipelineCompiler::linkDynamicThreadingReport(b);
    }
    Kernel::linkExternalMethods(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addAdditionalFunctions
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::addAdditionalFunctions(KernelBuilder & b) {
    // TODO: to ensure that we can pass the correct num of threads, we cannot statically compile the
    // main method until we add the thread count as a parameter. Investigate whether we can make a
    // better "wrapper" method for that that allows easier access to the output scalars.
#if 0
    if (hasAttribute(AttrId::InternallySynchronized) || containsKernelFamilyCalls() || generatesDynamicRepeatingStreamSets()) {
        return;
    }
    addOrDeclareMainFunction(b, Kernel::AddExternal);
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief containsKernelFamilies
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned PipelineKernel::getNumOfNestedKernelFamilyCalls() const {
    return mNumOfKernelFamilyCalls;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addFamilyInitializationArgTypes
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::addAdditionalInitializationArgTypes(KernelBuilder & b, InitArgTypes & argTypes) const {
    const auto n = getNumOfNestedKernelFamilyCalls();
    #ifndef NDEBUG
    unsigned m = 0;
    for (const auto & k : mKernels) {
        // If this is a kernel family call, the "main" will pass in the required pointers.
        // However, a non-family call could still refer to a kernel that has nested family
        // calls of its own. During initialization, we pass in the pointers that that
        m += k.isFamilyCall() ? 1U : k.Object->getNumOfNestedKernelFamilyCalls();
    }
    assert ("reported number of nested kernels does not match actual?" && (m == n));
    #endif
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    if (LLVM_LIKELY(n > 0)) {
        argTypes.append(n * 7U, voidPtrTy);
    }
    IntegerType * const sizeTy = b.getSizeTy();
    if (LLVM_UNLIKELY(hasInternallyGeneratedStreamSets())) {
        const auto m = getInternallyGeneratedStreamSets().size();
        argTypes.reserve(m * 2);
        for (unsigned i = 0; i < m; ++i) {
            argTypes.push_back(voidPtrTy);
            argTypes.push_back(sizeTy);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recursivelyConstructFamilyKernels
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::recursivelyConstructFamilyKernels(KernelBuilder & b, InitArgs & args, ParamMap & params, NestedStateObjs & toFree) const {
    for (const auto & k : mKernels) {
        const Kernel * const kernel = k.Object;
        if (LLVM_UNLIKELY(k.isFamilyCall())) {
            kernel->constructFamilyKernels(b, args, params, toFree);
        } else if (LLVM_UNLIKELY(kernel->getNumOfNestedKernelFamilyCalls() > 0)) {
            kernel->recursivelyConstructFamilyKernels(b, args, params, toFree);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief createRepeatingStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
Kernel::ParamMap::PairEntry PipelineKernel::createRepeatingStreamSet(KernelBuilder & b, const RepeatingStreamSet * ss, const size_t maxStrideLength) const {

    const auto fieldWidth = ss->getFieldWidth();
    const auto numElements = ss->getNumElements();
    const auto blockWidth = b.getBitBlockWidth();

    FixedVectorType * const vecTy = b.getBitBlockType();
    IntegerType * const intTy = cast<IntegerType>(vecTy->getScalarType());
    const auto laneWidth = intTy->getIntegerBitWidth();

    if (LLVM_UNLIKELY(!is_pow2(fieldWidth) || fieldWidth > laneWidth)) {
        report_fatal_error(StringRef("RepeatingStreamSet fieldwidth must be a power of 2 and no more than ") + std::to_string(laneWidth));
    }

    size_t patternLength = 0;
    if (numElements == 1 && ss->isUnaligned()) {
        if (fieldWidth < 8) {
            assert ((8 % fieldWidth) == 0);
            patternLength = 8U / fieldWidth;
        } else {
            patternLength = 1U;
        }
    } else {
        patternLength = blockWidth;
    }

    for (unsigned i = 0; i < numElements; ++i) {
        const auto & vec = ss->getPattern(i);
        const auto L = vec.size();
        if (LLVM_UNLIKELY(L == 0)) {
            report_fatal_error("Zero-length repeating streamset elements are not permitted");
        }

        patternLength = boost::lcm<size_t>(patternLength, L);

        #ifndef NDEBUG
        const auto maxVal = (1ULL << static_cast<size_t>(fieldWidth)) - 1ULL;
        for (auto v : vec) {
            if (LLVM_UNLIKELY(v > maxVal)) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream msg(tmp);
                msg << "Repeating streamset value " << v << " exceeds a " << fieldWidth << "-bit value";
                report_fatal_error(StringRef(msg.str()));
            }
        }
        #endif
    }

    // If this repeating streamset has a single stream element, we only need to ensure we generate a
    // byte-aligned variable since the pipeline can easily use K "memcpys" to splat the value out to
    // the desired length, where K is log2(lcm(L,blockwidth)/lcm(L,8)) and L is the pattern length.
    // However, if we have multiple stream elements, this becomes much harder because streamsets have
    // a "strip-mined" layout. I.e., for each element, BlockWidth number of values are laid out
    // sequentially in memory. Strip-mining promotes better cache utilization but means that we'd end
    // up having many tiny memcpys to reassemble the minimal set of data.

    unsigned runLength = 0;
    unsigned copyableLength = 0;
    if (numElements == 1 && ss->isUnaligned()) {
        runLength = ((patternLength + maxStrideLength + blockWidth - 1UL) / blockWidth);
    } else {
        runLength = (patternLength / blockWidth);
        copyableLength = (maxStrideLength / blockWidth);
    }

    const auto totalStrides = runLength + copyableLength;

    std::vector<Constant *> dataVectorArray(totalStrides);

    const auto numLanes = blockWidth / laneWidth;
    ArrayType * const elementTy = ArrayType::get(vecTy, fieldWidth);
    ArrayType * const streamSetTy = ArrayType::get(elementTy, numElements);


    SmallVector<Constant *, 16> laneVal(numLanes);
    SmallVector<Constant *, 16> packVal(fieldWidth);
    SmallVector<Constant *, 16> elemVal(numElements);

    SmallVector<uint64_t, 16> elementPos(numElements, 0);

    assert ((laneWidth % fieldWidth) == 0);

    for (unsigned r = 0; r < runLength; ++r) {
        for (unsigned p = 0; p < numElements; ++p) {
            const auto & vec = ss->getPattern(p);
            const auto L = vec.size();
            for (uint64_t i = 0; i < fieldWidth; ++i) {
                for (uint64_t j = 0; j < numLanes; ++j) {
                    uint64_t V = 0;
                    for (uint64_t k = 0; k != laneWidth; k += fieldWidth) {
                        assert (k < laneWidth);
                        auto & pos = elementPos[p];
                        const auto v = vec[pos];
                        V |= (v << k);
                        pos = (pos + 1U) % L;
                    }
                    laneVal[j] = ConstantInt::get(intTy, V, false);
                }
                packVal[i] = ConstantVector::get(laneVal);
            }
            elemVal[p] = ConstantArray::get(elementTy, packVal);
        }
        dataVectorArray[r] = ConstantArray::get(streamSetTy, elemVal);
    }

    for (unsigned r = 0; r < copyableLength; ++r) {
        const auto & v = dataVectorArray[r]; assert (v);
        assert (dataVectorArray[r % runLength] == v);
        assert (dataVectorArray[r + runLength] == nullptr);
        dataVectorArray[r + runLength] = v;
    }

    ArrayType * const arrTy = ArrayType::get(streamSetTy, totalStrides);

    Constant * const patternVec = ConstantArray::get(arrTy, dataVectorArray);

    Module & mod = *b.getModule();
    GlobalVariable * const patternData =
        new GlobalVariable(mod, arrTy, true, GlobalValue::ExternalLinkage, patternVec);
    const auto align = blockWidth / 8;
    patternData->setAlignment(MaybeAlign{align});
    Value * const ptr = b.CreatePointerCast(patternData, b.getVoidPtrTy());
    return ParamMap::PairEntry{ptr, b.getSize(patternLength)};
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runOptimizationPasses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::runOptimizationPasses(KernelBuilder & b) const {
    COMPILER->runOptimizationPasses(b);
}

#define JOIN3(X,Y,Z) BOOST_JOIN(X,BOOST_JOIN(Y,Z))

#define REPLACE_INTERNAL_KERNEL_BINDINGS(BindingType) \
    const auto * const from = JOIN3(m, BindingType, s)[i].getRelationship(); \
    for (const auto & P : mKernels) { \
        const auto & B = P.Object->JOIN3(get, BindingType, Bindings)(); \
        for (unsigned j = 0; j < B.size(); ++j) { \
            if (LLVM_UNLIKELY(B[j].getRelationship() == from)) { \
                P.Object->JOIN3(set, BindingType, At)(j, value); } } } \
    JOIN3(m, BindingType, s)[i].setRelationship(value);

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setInputStreamSetAt
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::setInputStreamSetAt(const unsigned i, StreamSet * const value) {
    REPLACE_INTERNAL_KERNEL_BINDINGS(InputStreamSet);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setOutputStreamSetAt
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::setOutputStreamSetAt(const unsigned i, StreamSet * const value) {
    REPLACE_INTERNAL_KERNEL_BINDINGS(OutputStreamSet);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setInputScalarAt
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::setInputScalarAt(const unsigned i, Scalar * const value) {
    REPLACE_INTERNAL_KERNEL_BINDINGS(InputScalar);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setOutputScalarAt
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::setOutputScalarAt(const unsigned i, Scalar * const value) {
    REPLACE_INTERNAL_KERNEL_BINDINGS(OutputScalar);
}

#undef JOIN3
#undef REPLACE_INTERNAL_KERNEL_BINDINGS

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief instantiateKernelCompiler
 ** ------------------------------------------------------------------------------------------------------------- */
std::unique_ptr<KernelCompiler> PipelineKernel::instantiateKernelCompiler(KernelBuilder & b) const {
    return std::make_unique<PipelineCompiler>(b, const_cast<PipelineKernel *>(this));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isCachable
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineKernel::isCachable() const {
    return codegen::EnablePipelineObjectCache && !codegen::EnableIllustrator;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeInternallyGeneratedStreamSetScaleVector
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::writeInternallyGeneratedStreamSetScaleVector(const Relationships & R, MetadataScaleVector & V, const size_t scale) const {
    assert (hasInternallyGeneratedStreamSets());

    Module * const M = getModule();
    NamedMDNode * const msl = M->getNamedMetadata("rsl");
    assert (msl);
    assert (msl->getNumOperands() > 0);
    assert (msl->getOperand(0)->getNumOperands() > 0);
    ConstantAsMetadata * const c = cast<ConstantAsMetadata>(msl->getOperand(0)->getOperand(0));
    Constant * ar = c->getValue();
    const auto m = mKernels.size();

    auto getJthOffset = [&](const unsigned j) -> size_t {
        FixedArray<unsigned, 1> off;
        off[0] = j;
        #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(15, 0, 0)
        const Constant * const v = ConstantFoldExtractValueInstruction(ar, off);
        #else
        const Constant * const v = ConstantExpr::getExtractValue(ar, off);
        #endif
        return (cast<ConstantInt>(v)->getLimitedValue() * scale);
    };

    unsigned j = 0;

    for (unsigned i = 0; i != m; ++i) {
        const Kernel * const kernel = mKernels[i].Object;
        if (LLVM_UNLIKELY(kernel->hasInternallyGeneratedStreamSets())) {
            kernel->writeInternallyGeneratedStreamSetScaleVector(R, V, getJthOffset(j++));
        }
    }

    const auto & S = getInternallyGeneratedStreamSets();
    const auto n = S.size();
    for (unsigned i = 0; i != n; ++i) {
        const auto f = std::find(R.begin(), R.end(), S[i]);
        assert (f != R.end());
        const auto k = std::distance(R.begin(), f);
        // More than one nested pipeline could require the same repeating
        // streamset yet those pipelines may have different periods for them.
        // Pick the largest one.
        V[k] = std::max<size_t>(V[k], getJthOffset(j++));
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addOrDeclareMainFunction
 ** ------------------------------------------------------------------------------------------------------------- */
Function * PipelineKernel::addOrDeclareMainFunction(KernelBuilder & b, const MainMethodGenerationType method) const {

    unsigned suppliedArgs = 0;
    if (LLVM_LIKELY(isStateful())) {
        suppliedArgs += 1;
    }
    if (LLVM_LIKELY(hasThreadLocal())) {
        suppliedArgs += 1;
    }

    Module * const m = b.getModule();
    Function * const doSegment = getDoSegmentFunction(b, false); assert (doSegment);
    assert (doSegment->arg_size() >= suppliedArgs);
   //  const auto numOfDoSegArgs = doSegment->arg_size() - suppliedArgs;
    Function * const terminate = getFinalizeFunction(b);

    const auto numOfStreamSets = mInputStreamSets.size() + mOutputStreamSets.size();

    // maintain consistency with the Kernel interface by passing first the stream sets
    // and then the scalars.
    SmallVector<Type *, 32> params;
    params.reserve(getNumOfScalarInputs() + numOfStreamSets);

    StructType * streamSetTy = nullptr;
    PointerType * streamSetPtrTy = nullptr;
    PointerType * voidPtrTy = b.getVoidPtrTy();
    IntegerType * int64Ty = b.getInt64Ty();
    if (numOfStreamSets) {
        // must match streamsetptr.h
        FixedArray<Type *, 2> fields;
        fields[0] = voidPtrTy;
        fields[1] = int64Ty;
        streamSetTy = StructType::get(b.getContext(), fields);
        streamSetPtrTy = streamSetTy->getPointerTo();
    }

    // The initial params of doSegment are its shared handle, thread-local handle and numOfStrides.
    // (assuming the kernel has both handles). The remaining are the stream set params.

    for (unsigned i = 0; i < numOfStreamSets; ++i) {
        params.push_back(streamSetPtrTy);
    }
    for (const auto & input : getInputScalarBindings()) {
        if (isa<CommandLineScalar>(input.getRelationship())) {
            continue;
        }
        params.push_back(input.getType());
    }

    Function * createIllustrator = nullptr;
    Function * displayCapturedData = nullptr;
    Function * destroyIllustrator = nullptr;
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        PointerType * voidPtrTy = b.getVoidPtrTy();
        createIllustrator = b.LinkFunction("__createStreamDataIllustrator", FunctionType::get(voidPtrTy, false), (void*)&createStreamDataIllustrator);
        BEGIN_SCOPED_REGION
        FixedArray<Type *, 2> args;
        args[0] = voidPtrTy;
        args[1] = b.getSizeTy();
        FunctionType * funTy = FunctionType::get(b.getVoidTy(), args, false);
        displayCapturedData = b.LinkFunction("__displayCapturedData", funTy, (void*)&illustratorDisplayCapturedData);
        END_SCOPED_REGION
        BEGIN_SCOPED_REGION
        FixedArray<Type *, 1> args;
        args[0] = voidPtrTy;
        FunctionType * funTy = FunctionType::get(b.getVoidTy(), args, false);
        destroyIllustrator = b.LinkFunction("__destroyStreamDataIllustrator", funTy, (void*)&destroyStreamDataIllustrator);
        END_SCOPED_REGION
    }

    const auto linkageType = (method == AddInternal) ? Function::InternalLinkage : Function::ExternalLinkage;

    SmallVector<char, 256> tmp;
    raw_svector_ostream funcNameGen(tmp);
    funcNameGen << getName() << '@' << codegen::SegmentSize << "_main";
    const auto funcName = funcNameGen.str();

    Function * main = m->getFunction(funcName);
    if (LLVM_LIKELY(main == nullptr)) {
        // get the finalize method output type and set its return type as this function's return type
        FunctionType * const mainFunctionType = FunctionType::get(terminate->getReturnType(), params, false);
        main = Function::Create(mainFunctionType, linkageType, funcName, m);
        main->setCallingConv(CallingConv::C);
    }

    // declaration only; exit
    if (method == DeclareExternal) {
        return main;
    }

    assert (main->empty());

    b.SetInsertPoint(BasicBlock::Create(b.getContext(), "entry", main));
    auto arg = main->arg_begin();
    auto nextArg = [&]() -> Value * {
        assert (arg != main->arg_end());
        Value * const v = &*arg;
        std::advance(arg, 1);
        return v;
    };
    SmallVector<Value *, 16> segmentArgs(doSegment->arg_size());

    if (LLVM_UNLIKELY(numOfStreamSets > 0)) {

        auto argCount = suppliedArgs;

        ConstantInt * const i32_ZERO = b.getInt32(0);
        ConstantInt * const i32_ONE = b.getInt32(1);

        Value * const sz_ZERO = b.getSize(0);

        FixedArray<Value *, 2> fields;
        fields[0] = i32_ZERO;

        for (auto i = mInputStreamSets.size(); i--; ) {
            Value * const streamSetArg = nextArg();
            assert (streamSetArg->getType() == streamSetPtrTy);
            // virtual base input address
            fields[1] = i32_ZERO;
            Value * const vbaPtr = b.CreateGEP(streamSetTy, streamSetArg, fields);
            segmentArgs[argCount++] = b.CreateLoad(voidPtrTy, vbaPtr);
            // processed input items
            fields[1] = i32_ONE;
            Value * const processedPtr = b.CreateAllocaAtEntryPoint(b.getSizeTy());
            b.CreateStore(sz_ZERO, processedPtr);
            segmentArgs[argCount++] = processedPtr; // updatable
            // accessible input items
            segmentArgs[argCount++] = b.CreateLoad(int64Ty, b.CreateGEP(streamSetTy, streamSetArg, fields));
        }

        for (auto i = mOutputStreamSets.size(); i--; ) {
            Value * const streamSetArg = nextArg();
            assert (streamSetArg->getType() == streamSetPtrTy);

            // shared dynamic buffer handle or virtual base output address
            fields[1] = i32_ZERO;
            segmentArgs[argCount++] = b.CreateGEP(streamSetTy, streamSetArg, fields);

            // produced output items
            fields[1] = i32_ONE;
            Value * const itemPtr = b.CreateGEP(streamSetTy, streamSetArg, fields);
            segmentArgs[argCount++] = itemPtr;
            segmentArgs[argCount++] = b.CreateLoad(int64Ty, itemPtr);
        }

        assert (argCount == doSegment->arg_size());
    }
    Value * sharedHandle = nullptr;
    NestedStateObjs toFree;
    ParamMap paramMap;

    // construct any repeating streamsets and add them to the map
    if (hasInternallyGeneratedStreamSets()) {
        const auto & I = getInternallyGeneratedStreamSets();
        MetadataScaleVector scaleVector(I.size(), 0U);
        writeInternallyGeneratedStreamSetScaleVector(I, scaleVector, 1U);
        const auto n = I.size();
        for (unsigned i = 0; i < n; ++i) {
            assert (scaleVector[i] > 0);
            const auto rs = cast<RepeatingStreamSet>(I[i]);
            paramMap.set(rs, createRepeatingStreamSet(b, rs, scaleVector[i]));
        }
    }

    #ifdef ENABLE_PAPI
    Value * eventSet = nullptr;
    Value * eventListVal = nullptr;

    if (LLVM_UNLIKELY(codegen::PapiCounterOptions.compare(codegen::OmittedOption) != 0)) {
        SmallVector<int, 8> eventList;
        Type * const intTy = TypeBuilder<int, false>::get(b.getContext());
        eventSet = ConstantInt::get(intTy, initializePAPI(eventList));
        const auto n = eventList.size();
        Constant * const initializer = ConstantDataArray::get(b.getContext(), ArrayRef<int>(eventList.data(), n));
        eventListVal = new GlobalVariable(*m, initializer->getType(), true, GlobalVariable::ExternalLinkage, initializer);
        PipelineCompiler::linkPAPILibrary(b);
    }
    #endif

    Value * illustratorObj = nullptr;

    for (const auto & input : getInputScalarBindings()) {
        const auto scalar = input.getRelationship(); assert (scalar);
        Value * value = nullptr;
        if (isa<CommandLineScalar>(scalar)) {
            using C = CommandLineScalarType;
            switch (cast<CommandLineScalar>(scalar)->getCLType()) {
                case C::MinThreadCount:
                    value = b.getSize(2);
                    break;
                case C::MaxThreadCount:
                    value = b.getSize(codegen::SegmentThreads);
                    break;
                case C::DynamicMultithreadingPeriod:
                    value = b.getSize(codegen::DynamicMultithreadingPeriod);
                    break;
                case C::BufferSegmentLength:
                    value = b.getSize(codegen::BufferSegments);
                    break;
                case C::DynamicMultithreadingAddSynchronizationThreshold:
                    value = ConstantFP::get(b.getFloatTy(), codegen::DynamicMultithreadingAddThreshold); // %
                    break;
                case C::DynamicMultithreadingRemoveSynchronizationThreshold:
                    value = ConstantFP::get(b.getFloatTy(), codegen::DynamicMultithreadingRemoveThreshold); // %
                    break;
                case C::ParabixIllustratorObject:
                    assert (createIllustrator);
                    assert (illustratorObj == nullptr);
                    illustratorObj = b.CreateCall(createIllustrator->getFunctionType(), createIllustrator);
                    value = illustratorObj;
                    break;
                #ifdef ENABLE_PAPI
                case C::PAPIEventSet:
                    value = eventSet;
                    break;
                case C::PAPIEventList:
                    value = eventListVal;
                    break;
                #endif
                default:
                    llvm_unreachable("unknown command line scalar");
            }
        } else {
            value = nextArg();
        }
        paramMap.set(scalar, value);
    }

    InitArgs args;
    sharedHandle = constructFamilyKernels(b, args, paramMap, toFree);
    assert (isStateful() || sharedHandle == nullptr);

    size_t argCount = 0;
    if (LLVM_LIKELY(isStateful())) {
        segmentArgs[argCount++] = sharedHandle;
    }
    Value * threadLocalHandle = nullptr;
    if (LLVM_LIKELY(hasThreadLocal())) {
        SmallVector<Value *, 2> args;
        if (LLVM_LIKELY(isStateful())) {
            args.push_back(sharedHandle);
        }
        args.push_back(ConstantPointerNull::get(getThreadLocalStateType()->getPointerTo()));
        threadLocalHandle = initializeThreadLocalInstance(b, args);
        segmentArgs[argCount++] = threadLocalHandle;
        toFree.push_back(threadLocalHandle);
    }

    assert (argCount == suppliedArgs);

    if (LLVM_UNLIKELY(hasAttribute(AttrId::InternallySynchronized))) {
        report_fatal_error(StringRef(doSegment->getName()) + " cannot be externally synchronized");
    }



    // allocate any internal stream sets
    if (LLVM_LIKELY(allocatesInternalStreamSets())) {
        Constant * const sz_ONE = b.getSize(1);
        Function * const allocShared = getAllocateSharedInternalStreamSetsFunction(b);
        SmallVector<Value *, 2> allocArgs;
        if (LLVM_LIKELY(isStateful())) {
            allocArgs.push_back(sharedHandle);
        }
        // pass in the desired number of segments
        allocArgs.push_back(sz_ONE);
        b.CreateCall(allocShared->getFunctionType(), allocShared, allocArgs);
        if (LLVM_LIKELY(hasThreadLocal())) {
            Function * const allocThreadLocal = getAllocateThreadLocalInternalStreamSetsFunction(b);
            SmallVector<Value *, 3> allocArgs;
            if (LLVM_LIKELY(isStateful())) {
                allocArgs.push_back(sharedHandle);
            }
            allocArgs.push_back(threadLocalHandle);
            allocArgs.push_back(sz_ONE);
            b.CreateCall(allocThreadLocal->getFunctionType(), allocThreadLocal, allocArgs);
        }
    }

    PHINode * successPhi = nullptr;
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts) ||
                      codegen::DebugOptionIsSet(codegen::EnablePipelineAsserts))) {
        BasicBlock * const handleCatch = b.CreateBasicBlock("");
        BasicBlock * const handleDeallocation = b.CreateBasicBlock("");

        IntegerType * const int32Ty = b.getInt32Ty();
        PointerType * const int8PtrTy = b.getInt8PtrTy();
        LLVMContext & C = b.getContext();
        StructType * const caughtResultType = StructType::get(C, { int8PtrTy, int32Ty });
        Function * const personalityFn = b.getDefaultPersonalityFunction();
        main->setPersonalityFn(personalityFn);

        BasicBlock * const beforeInvoke = b.GetInsertBlock();
        b.CreateInvoke(doSegment, handleDeallocation, handleCatch, segmentArgs);

        b.SetInsertPoint(handleCatch);
        LandingPadInst * const caughtResult = b.CreateLandingPad(caughtResultType, 0);
        caughtResult->addClause(ConstantPointerNull::get(int8PtrTy));
        Function * catchFn = b.getBeginCatch();
        Function * catchEndFn = b.getEndCatch();
        b.CreateCall(catchFn->getFunctionType(), catchFn, {b.CreateExtractValue(caughtResult, 0)});
        b.CreateCall(catchEndFn->getFunctionType(), catchEndFn, {});
        BasicBlock * const afterCatch = b.GetInsertBlock();
        b.CreateBr(handleDeallocation);

        b.SetInsertPoint(handleDeallocation);
        successPhi = b.CreatePHI(b.getInt1Ty(), 2);
        successPhi->addIncoming(b.getTrue(), beforeInvoke);
        successPhi->addIncoming(b.getFalse(), afterCatch);
    } else {
        b.CreateCall(doSegment->getFunctionType(), doSegment, segmentArgs);
    }
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        BEGIN_SCOPED_REGION
        FixedArray<Value *, 2> args;
        args[0] = illustratorObj;
        args[1] = b.getSize(b.getBitBlockWidth());
        b.CreateCall(displayCapturedData->getFunctionType(), displayCapturedData, args);
        END_SCOPED_REGION
        BEGIN_SCOPED_REGION
        FixedArray<Value *, 1> args;
        args[0] = illustratorObj;
        b.CreateCall(destroyIllustrator->getFunctionType(), destroyIllustrator, args);
        END_SCOPED_REGION
    }
    SmallVector<Value *, 3> finalizeArgs;
    if (LLVM_LIKELY(isStateful())) {
        finalizeArgs.push_back(sharedHandle);
    }
    if (LLVM_LIKELY(hasThreadLocal())) {
        finalizeArgs.push_back(threadLocalHandle);
        finalizeArgs.push_back(threadLocalHandle);
        finalizeThreadLocalInstance(b, finalizeArgs);
        finalizeArgs.pop_back();
    }
    Value * const result = finalizeInstance(b, finalizeArgs);
    for (Value * stateObj : toFree) {
        b.CreateFree(stateObj);
    }
    #ifdef ENABLE_PAPI
    if (LLVM_UNLIKELY(eventSet != nullptr)) {
        terminatePAPI(b, eventSet);
    }
    #endif
    b.CreateRet(result);
    return main;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief annotateKernelNameWithPipelineFlags
 ** ------------------------------------------------------------------------------------------------------------- */
/* static */ std::string PipelineKernel::annotateSignatureWithPipelineFlags(std::string && name) {
    raw_string_ostream out(name);
    switch (codegen::PipelineCompilationMode) {
        case codegen::PipelineCompilationModeOptions::DefaultFast:
            out << 'F';
            break;
        case codegen::PipelineCompilationModeOptions::Expensive:
            out << 'X';
            break;
    }


    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::EnableAnonymousMMapedDynamicLinearBuffers))) {
        out << "+AML";
    }

    if (codegen::EnableDynamicMultithreading) {
        out << "+DM";
    }

    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet())) {
        if (DebugOptionIsSet(codegen::EnableCycleCounter)) {
            out << "+CYC";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::EnableBlockingIOCounter))) {
            out << "+BIC";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceBlockedIO))) {
            out << "+TBIO";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceDynamicBuffers))) {
            out << "+TDB";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceDynamicMultithreading))) {
            out << "+TDM";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceProducedItemCounts))) {
            out << "+TPIC";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceUnconsumedItemCounts))) {
            out << "+TUIC";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceStridesPerSegment))) {
            out << "+TSS";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::GenerateTransferredItemCountHistogram))) {
            out << "+GTH";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::GenerateDeferredItemCountHistogram))) {
            out << "+GDH";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisableThreadLocalStreamSets))) {
            out << "-TL";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::EnableAnonymousMMapedDynamicLinearBuffers))) {
            out << "+AML";
        }
    }
    #ifdef ENABLE_PAPI
    const auto & S = codegen::PapiCounterOptions;
    if (LLVM_UNLIKELY(S.compare(codegen::OmittedOption) != 0)) {
        out << "+PAPI";
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisplayPAPICounterThreadTotalsOnly))) {
            out << "TT";
        }
        out << (std::count_if(S.begin(), S.end(), [](std::string::value_type c){return c == ',';}) + 1);
    }
    #endif
    out.flush();
    return std::move(name);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makePipelineHashName
 ** ------------------------------------------------------------------------------------------------------------- */
/* static */ std::string PipelineKernel::makePipelineHashName(const std::string & signature) {
    std::string tmp;
    tmp.reserve(32);
    raw_string_ostream name(tmp);
    name << 'P' << Kernel::getStringHash(signature);
    name.flush();
    return tmp;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
PipelineKernel::PipelineKernel(KernelBuilder & b,
                               std::string && signature,
                               const unsigned numOfKernelFamilyCalls,
                               Kernels && kernels, CallBindings && callBindings,
                               Bindings && stream_inputs, Bindings && stream_outputs,
                               Bindings && scalar_inputs, Bindings && scalar_outputs,
                               Relationships && internallyGenerated,
                               LengthAssertions && lengthAssertions)
: PipelineKernel(Internal{}
, b
, annotateSignatureWithPipelineFlags(std::move(signature))
, numOfKernelFamilyCalls
, std::move(kernels)
, std::move(callBindings)
, std::move(stream_inputs)
, std::move(stream_outputs)
, std::move(scalar_inputs)
, std::move(scalar_outputs)
, std::move(internallyGenerated)
, std::move(lengthAssertions)
) {


}

PipelineKernel::PipelineKernel(Internal, KernelBuilder & b,
               std::string && signature,
               const unsigned numOfKernelFamilyCalls,
               Kernels && kernels, CallBindings && callBindings,
               Bindings && stream_inputs, Bindings && stream_outputs,
               Bindings && scalar_inputs, Bindings && scalar_outputs,
               Relationships && internallyGenerated,
               LengthAssertions && lengthAssertions)
: Kernel(b, TypeId::Pipeline,
         makePipelineHashName(signature),
         std::move(stream_inputs), std::move(stream_outputs),
         std::move(scalar_inputs), std::move(scalar_outputs),
         {} /* Internal scalars are generated by the PipelineCompiler */)
, mNumOfKernelFamilyCalls(numOfKernelFamilyCalls)
, mSignature(std::move(signature))
, mInternallyGeneratedStreamSets(std::move(internallyGenerated))
, mKernels(std::move(kernels))
, mCallBindings(std::move(callBindings))
, mLengthAssertions(std::move(lengthAssertions)) {

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
PipelineKernel::PipelineKernel(KernelBuilder & b,
               Bindings && stream_inputs, Bindings && stream_outputs,
               Bindings && scalar_inputs, Bindings && scalar_outputs)
: Kernel(b, TypeId::Pipeline,
         std::move(stream_inputs), std::move(stream_outputs),
         std::move(scalar_inputs), std::move(scalar_outputs))
{

}

PipelineKernel::~PipelineKernel() {

}

}
