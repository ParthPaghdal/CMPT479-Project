#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief bindAdditionalInitializationArguments
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::bindAdditionalInitializationArguments(KernelBuilder & b, ArgIterator & arg, const ArgIterator & arg_end) {
    bindFamilyInitializationArguments(b, arg, arg_end);
    bindRepeatingStreamSetInitializationArguments(b, arg, arg_end);
    assert (arg == arg_end);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateImplicitKernels
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateImplicitKernels(KernelBuilder & b) {
    assert (b.getModule() == mTarget->getModule());
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const_cast<Kernel *>(getKernel(i))->generateOrLoadKernel(b);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addPipelineKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addPipelineKernelProperties(KernelBuilder & b) {
    // TODO: look into improving cache locality/false sharing of this struct

    // TODO: create a non-persistent / pass through input scalar type to allow the
    // pipeline to pass an input scalar to a kernel rather than recording it needlessly?
    // Non-family kernels can be contained within the shared state but family ones
    // must be allocated dynamically.

    IntegerType * const sizeTy = b.getSizeTy();

    mTarget->addInternalScalar(sizeTy, EXPECTED_NUM_OF_STRIDES_MULTIPLIER, 0);

    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        PointerType * const int8PtrTy = b.getInt8PtrTy();
        mTarget->addThreadLocalScalar(int8PtrTy, BASE_THREAD_LOCAL_STREAMSET_MEMORY, 0);
        mTarget->addThreadLocalScalar(sizeTy, BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES, 0);
    }
    // NOTE: both the shared and thread local objects are parameters to the kernel.
    // They get automatically set by reading in the appropriate params.

    if (HasZeroExtendedStream) {
        PointerType * const voidPtrTy = b.getVoidPtrTy();
        mTarget->addThreadLocalScalar(voidPtrTy, ZERO_EXTENDED_BUFFER);
        mTarget->addThreadLocalScalar(sizeTy, ZERO_EXTENDED_SPACE);
    }

    mKernelId = 0;
    mKernel = mTarget;
    auto currentPartitionId = -1U;
    addBufferHandlesToPipelineKernel(b, PipelineInput, 0);
    addConsumerKernelProperties(b, PipelineInput);
    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    unsigned nestedSynchronizationVariableCount = 0;
    #endif
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        // Is this the start of a new partition?
        const auto partitionId = KernelPartitionId[i];
        const bool isRoot = (partitionId != currentPartitionId);
        currentPartitionId = partitionId;
        addInternalKernelProperties(b, i, isRoot);
        addCycleCounterProperties(b, i, isRoot);
        #ifdef ENABLE_PAPI
        addPAPIEventCounterKernelProperties(b, i, isRoot);
        #endif
        addProducedItemCountDeltaProperties(b, i);
        addUnconsumedItemCountProperties(b, i);
        #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
        if (isRoot && PartitionJumpTargetId[partitionId] == (PartitionCount - 1)) {
            mTarget->addInternalScalar(sizeTy,
                NESTED_LOGICAL_SEGMENT_NUMBER_PREFIX + std::to_string(++nestedSynchronizationVariableCount), getCacheLineGroupId(i));
        }
        #endif
    }
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        mTarget->addThreadLocalScalar(b.getInt64Ty(), STATISTICS_CYCLE_COUNT_TOTAL,
                                      getCacheLineGroupId(PipelineOutput), ThreadLocalScalarAccumulationRule::Sum);
    }
    addRepeatingStreamSetBufferProperties(b);
    generateMetaDataForRepeatingStreamSets(b);
    #ifdef ENABLE_PAPI
    addPAPIEventCounterPipelineProperties(b);
    #endif
    if (mUseDynamicMultithreading) {
        mTarget->addInternalScalar(sizeTy, NEXT_LOGICAL_SEGMENT_NUMBER, getCacheLineGroupId(PipelineOutput));
        if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
            addDynamicThreadingReportProperties(b, getCacheLineGroupId(PipelineOutput + 1));
        }
    }
    addZeroInputStructProperties(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addInternalKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addInternalKernelProperties(KernelBuilder & b, const unsigned kernelId, const bool isRoot) {

    mKernelId = kernelId;
    mKernel = getKernel(kernelId);
    const auto isStateless = isKernelStateFree(kernelId);
    if (LLVM_UNLIKELY(isStateless)) {
        mIsStatelessKernel.set(kernelId);
    }
    assert (mIsStatelessKernel.test(kernelId) == isStateless);
    const auto isInternallySynchronized = mKernel->hasAttribute(AttrId::InternallySynchronized);
    if (LLVM_UNLIKELY(isInternallySynchronized)) {
        mIsInternallySynchronized.set(kernelId);
    }
    #if defined(DISABLE_ALL_DATA_PARALLEL_SYNCHRONIZATION)
    const auto allowDataParallelExecution = false;
    #elif defined(ALLOW_INTERNALLY_SYNCHRONIZED_KERNELS_TO_BE_DATA_PARALLEL)
    const auto allowDataParallelExecution = isStateless || isInternallySynchronized;
    #else
    const auto allowDataParallelExecution = isStateless;
    #endif

    IntegerType * const sizeTy = b.getSizeTy();

    const auto groupId = getCacheLineGroupId(kernelId);

    addTerminationProperties(b, kernelId, groupId);

    const auto name = makeKernelName(kernelId);

    const auto syncLockType = allowDataParallelExecution ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
    mTarget->addInternalScalar(sizeTy, name + LOGICAL_SEGMENT_SUFFIX[syncLockType], groupId);

    if (isRoot) {
        addSegmentLengthSlidingWindowKernelProperties(b, kernelId, groupId);
    }

    addConsumerKernelProperties(b, kernelId);

    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto prefix = makeBufferName(kernelId, br.Port);
        mTarget->addInternalScalar(sizeTy, prefix + ITEM_COUNT_SUFFIX, groupId);
        if (LLVM_UNLIKELY(isStateless)) {
            mTarget->addInternalScalar(sizeTy, prefix + STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX, groupId);
        }
        if (LLVM_UNLIKELY(br.isDeferred())) {
            mTarget->addInternalScalar(sizeTy, prefix + DEFERRED_ITEM_COUNT_SUFFIX, groupId);
        }
    }

    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto prefix = makeBufferName(kernelId, br.Port);
        mTarget->addInternalScalar(sizeTy, prefix + ITEM_COUNT_SUFFIX, groupId);
        if (LLVM_UNLIKELY(isStateless)) {
            mTarget->addInternalScalar(sizeTy, prefix + STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX, groupId);
        }
        if (LLVM_UNLIKELY(br.isDeferred())) {
            mTarget->addInternalScalar(sizeTy, prefix + DEFERRED_ITEM_COUNT_SUFFIX, groupId);
        }
    }

    addBufferHandlesToPipelineKernel(b, kernelId, groupId);

    addFamilyKernelProperties(b, kernelId, groupId);

    if (LLVM_UNLIKELY(isInternallySynchronized || mUsesIllustrator)) {
        // TODO: only needed if its possible to loop back or if we are not guaranteed that this kernel will always fire
        mTarget->addInternalScalar(sizeTy, name + INTERNALLY_SYNCHRONIZED_SUB_SEGMENT_SUFFIX, groupId);
    }

    if (LLVM_LIKELY(mKernel->isStateful())) {
        Type * sharedStateTy = nullptr;
        if (LLVM_UNLIKELY(isKernelFamilyCall(kernelId))) {
            sharedStateTy = b.getVoidPtrTy();
        } else {
            sharedStateTy = mKernel->getSharedStateType();
        }
        mTarget->addInternalScalar(sharedStateTy, name, groupId);
    }

    if (mKernel->hasThreadLocal()) {
        // we cannot statically allocate a "family" thread local object.
        Type * localStateTy = nullptr;
        if (LLVM_UNLIKELY(isKernelFamilyCall(kernelId))) {
            localStateTy = b.getVoidPtrTy();
        } else {
            localStateTy = mKernel->getThreadLocalStateType();
        }
        mTarget->addThreadLocalScalar(localStateTy, name + KERNEL_THREAD_LOCAL_SUFFIX, groupId);
    }

    if (LLVM_UNLIKELY(allowDataParallelExecution)) {
        mTarget->addInternalScalar(sizeTy, name + LOGICAL_SEGMENT_SUFFIX[SYNC_LOCK_POST_INVOCATION], groupId);
    }

    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram || mGenerateDeferredItemCountHistogram)) {
        addHistogramProperties(b, kernelId, groupId);
    }

    if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {
        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const auto bufferVertex = target(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[bufferVertex];
            if (bn.Buffer->isDynamic()) {
                const BufferPort & rd = mBufferGraph[e];
                const auto prefix = makeBufferName(kernelId, rd.Port);
                LLVMContext & C = b.getContext();
                const auto numOfConsumers = std::max(out_degree(bufferVertex, mConsumerGraph), 1UL);

                // segment num  0
                // new capacity 1
                // produced item count 2
                // consumer processed item count [3,n)
                Type * const traceStructTy = ArrayType::get(sizeTy, numOfConsumers + 3);

                FixedArray<Type *, 2> traceStruct;
                traceStruct[0] = traceStructTy->getPointerTo(); // pointer to trace log
                traceStruct[1] = sizeTy; // length of trace log
                mTarget->addInternalScalar(StructType::get(C, traceStruct),
                                                   prefix + STATISTICS_BUFFER_EXPANSION_SUFFIX, groupId);
            }
        }
    }

    if (LLVM_UNLIKELY(isRoot && DebugOptionIsSet(codegen::TraceStridesPerSegment))) {
        LLVMContext & C = b.getContext();
//        FixedArray<Type *, 2> recordStruct;
//        recordStruct[0] = sizeTy; // segment num
//        recordStruct[1] = sizeTy; // # of strides
        Type * const recordStructTy = ArrayType::get(sizeTy, 2);

        FixedArray<Type *, 4> traceStruct;
        traceStruct[0] = sizeTy; // last num of strides (to avoid unnecessary loads of the trace
                                 // log and simplify the logic for first stride)
        traceStruct[1] = recordStructTy->getPointerTo(); // pointer to trace log
        traceStruct[2] = sizeTy; // trace length
        traceStruct[3] = sizeTy; // trace capacity (for realloc)

        mTarget->addInternalScalar(StructType::get(C, traceStruct),
                                           name + STATISTICS_STRIDES_PER_SEGMENT_SUFFIX, groupId);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateInitializeMethod(KernelBuilder & b) {

    // TODO: if we detect a fatal error at init, we should not execute
    // the pipeline loop.

    initializeScalarValues(b);

    initializeKernelAssertions(b);

    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
    Constant * const aborted = getTerminationSignal(b, TerminationSignal::Aborted);

    Value * terminated = nullptr;
    auto partitionId = KernelPartitionId[PipelineInput];

    for (auto i = FirstKernel; i <= LastKernel; ++i) {

        const auto curPartitionId = KernelPartitionId[i];
        const auto isRoot = (curPartitionId != partitionId);
        partitionId = curPartitionId;
        // Family kernels must be initialized in the "main" method.
        setActiveKernel(b, i, false);
        assert (mKernelId == i);
        assert (mKernel->isGenerated());
        if (isRoot) {
            initializeStridesPerSegment(b);
        }

        if (LLVM_LIKELY(!isKernelFamilyCall(i))) {
            ArgVec args;
            if (LLVM_LIKELY(mKernel->isStateful())) {
                args.push_back(mKernelSharedHandle);
            }
            #ifndef NDEBUG
            unsigned expected = 0;
            #endif
            for (const auto e : make_iterator_range(in_edges(i, mScalarGraph))) {
                assert (mScalarGraph[e].Type == PortType::Input);
                assert (expected++ == mScalarGraph[e].Number);
                const auto scalar = source(e, mScalarGraph);
                args.push_back(getScalar(b, scalar));
            }
            addFamilyCallInitializationArguments(b, i, args);
            addRepeatingStreamSetInitializationArguments(i, args);
            #ifndef NDEBUG
            for (unsigned j = 0; j != args.size(); ++j) {
                assert (isFromCurrentFunction(b, args[j], false));
            }
            #endif
            Value * const signal = callKernelInitializeFunction(b, args);
            Value * const terminatedOnInit = b.CreateICmpNE(signal, unterminated);

            if (terminated) {
                terminated = b.CreateOr(terminated, terminatedOnInit);
            } else {
                terminated = terminatedOnInit;
            }
        }

        if (LLVM_UNLIKELY(mUsesIllustrator)) {
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & br = mBufferGraph[e];
                if (LLVM_UNLIKELY(br.isIllustrated())) {
                    registerStreamSetIllustrator(b, target(e, mBufferGraph));
                }
            }
        }

        // Is this the last kernel in a partition? If so, store the accumulated
        // termination signal.
        if (terminated && HasTerminationSignal[mKernelId]) {
            Value * const signal = b.CreateSelect(terminated, aborted, unterminated);
            writeTerminationSignal(b, mKernelId, signal);
            terminated = nullptr;
        }
    }



    if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
        initDynamicThreadingReportProperties(b);
    }
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateAllocateSharedInternalStreamSetsMethod(KernelBuilder & b, Value * const expectedNumOfStrides) {
    b.setScalarField(EXPECTED_NUM_OF_STRIDES_MULTIPLIER, expectedNumOfStrides);


    initializeInitialSlidingWindowSegmentLengths(b, expectedNumOfStrides);

    Value * allocScale = expectedNumOfStrides;
    if (LLVM_LIKELY(!mIsNestedPipeline)) {
        Value * bsl = b.getScalarField(BUFFER_SEGMENT_LENGTH);
        allocScale = b.CreateMul(allocScale, bsl);
        Value * const threadCount = b.getScalarField(MAXIMUM_NUM_OF_THREADS);
        allocScale = b.CreateMul(allocScale, threadCount);
    }

    bool hasAnyReturnedBuffer = false;
    for (const auto output : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        const auto streamSet = source(output, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isReturned())) {
            if (getReturnedBufferScaleFactor(streamSet) > 0) {
                hasAnyReturnedBuffer = true;
                break;
            }
        }
    }

    Value * expectedSourceOutputSize = nullptr;
    if (LLVM_UNLIKELY(hasAnyReturnedBuffer)) {
        for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
            if (LLVM_UNLIKELY(in_degree(kernel, mBufferGraph) == 0)) {
                setActiveKernel(b, kernel, false);
                assert (mKernel->isGenerated());
                FixedArray<Value *, 1> args;
                args[0] = mKernelSharedHandle;
                Value * eosVal = callKernelExpectedSourceOutputSizeFunction(b, args);
                expectedSourceOutputSize = b.CreateUMax(eosVal, expectedSourceOutputSize);
            }
        }
        expectedSourceOutputSize = b.CreateCeilUDiv(expectedSourceOutputSize, b.getSize(b.getBitBlockWidth()));
    }
    allocateOwnedBuffers(b, allocScale, expectedSourceOutputSize, true);
    initializeBufferExpansionHistory(b);
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateInitializeThreadLocalMethod(KernelBuilder & b) {
    assert (mTarget->hasThreadLocal());
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernel = getKernel(i);
        if (kernel->hasThreadLocal()) {
            setActiveKernel(b, i, true);
            assert (mKernel == kernel);
            callKernelInitializeThreadLocalFunction(b);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateThreadLocalInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateAllocateThreadLocalInternalStreamSetsMethod(KernelBuilder & b, Value * const expectedNumOfStrides) {
    assert (mTarget->hasThreadLocal());
    Value * allocScale = expectedNumOfStrides;
    if (LLVM_LIKELY(!mIsNestedPipeline)) {
        Value * bsl = b.getScalarField(BUFFER_SEGMENT_LENGTH);
        allocScale = b.CreateMul(allocScale, bsl);
    }
    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        auto size = RequiredThreadLocalStreamSetMemory;
        #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
        size *= THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER;
        #endif
        ConstantInt * const reqMemory = b.getSize(size);
        Value * const memorySize = b.CreateMul(reqMemory, allocScale);
        Value * const base = b.CreatePageAlignedMalloc(memorySize);
        PointerType * const int8PtrTy = b.getInt8PtrTy();
        b.setScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY, b.CreatePointerCast(base, int8PtrTy));
        b.setScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES, memorySize);
    }
    allocateOwnedBuffers(b, expectedNumOfStrides, nullptr, false);
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateKernelMethod(KernelBuilder & b) {
    initializeKernelAssertions(b);
    initializeScalarValues(b);
    if (mIsNestedPipeline) {
        generateSingleThreadKernelMethod(b);
    } else {
        generateMultiThreadKernelMethod(b);
    }
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateFinalizeMethod(KernelBuilder & b) {
    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet() || NumOfPAPIEvents > 0)) {
        // get the last segment # used by any kernel in case any reports require it.
        const auto type = isDataParallel(FirstKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        Value * const ptr = getSynchronizationLockPtrForKernel(b, FirstKernel, type);
        mSegNo = b.CreateLoad(b.getSizeTy(), ptr);

        printOptionalCycleCounter(b);
        #ifdef ENABLE_PAPI
        printPAPIReportIfRequested(b);
        #endif
        printOptionalBlockingIOStatistics(b);
        printOptionalBlockedIOPerSegment(b);
        printOptionalBufferExpansionHistory(b);
        printOptionalStridesPerSegment(b);
        printProducedItemCountDeltas(b);
        printUnconsumedItemCounts(b);
        if (mGenerateTransferredItemCountHistogram) {
            printHistogramReport(b, HistogramReportType::TransferredItems);
        }
        if (mGenerateDeferredItemCountHistogram) {
            printHistogramReport(b, HistogramReportType::DeferredItems);
        }
        if (TraceDynamicMultithreading) {
            printDynamicThreadingReport(b);
        }
    }

    initializeScalarValues(b);
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        setActiveKernel(b, i, true);
        SmallVector<Value *, 1> params;
        if (LLVM_LIKELY(mKernel->isStateful())) {
            assert (mTarget->isStateful());
            params.push_back(mKernelSharedHandle);
        }
        if (LLVM_UNLIKELY(mKernel->hasThreadLocal())) {
            assert (mTarget->hasThreadLocal());
            params.push_back(mKernelThreadLocalHandle);
        }
        mScalarValue[i] = callKernelFinalizeFunction(b, params);
    }
    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram || mGenerateDeferredItemCountHistogram)) {
        freeHistogramProperties(b);
    }
    deallocateRepeatingBuffers(b);
    releaseOwnedBuffers(b);
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateFinalizeThreadLocalMethod(KernelBuilder & b) {
    assert (mTarget->hasThreadLocal());
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernel = getKernel(i);
        assert (kernel->hasThreadLocal() || !isa<PipelineKernel>(kernel));
        if (kernel->hasThreadLocal()) {
            setActiveKernel(b, i, true, true);
            assert (mKernel == kernel);
            SmallVector<Value *, 2> args;
            if (LLVM_LIKELY(mKernelSharedHandle != nullptr)) {
                args.push_back(mKernelSharedHandle);
            }
            args.push_back(mKernelCommonThreadLocalHandle); assert (mKernelCommonThreadLocalHandle);
            args.push_back(mKernelThreadLocalHandle); assert (mKernelThreadLocalHandle);
            callKernelFinalizeThreadLocalFunction(b, args);
            if (LLVM_UNLIKELY(isKernelFamilyCall(i))) {
              //  b.CreateFree(mKernelThreadLocalHandle);
            }
        }
    }

    // Since all of the nested kernels thread local state is contained within
    // this pipeline thread's thread local state, freeing the pipeline's will
    // also free the inner kernels.
    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        b.CreateFree(b.getScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY));
    }
    if (LLVM_UNLIKELY(HasZeroExtendedStream)) {
        b.CreateFree(b.getScalarField(ZERO_EXTENDED_BUFFER));
    }
    freePendingFreeableDynamicBuffers(b);
    freeZeroedInputBuffers(b);
}

}
