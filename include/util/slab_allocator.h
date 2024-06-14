#pragma once

#include <llvm/Support/Allocator.h>

template <typename T = uint8_t, size_t SlabSize = 16 * 1024>
class SlabAllocator {
    template <typename U> friend class ProxyAllocator;
    using LLVMAllocator = llvm::BumpPtrAllocatorImpl<llvm::MallocAllocator, SlabSize>;
public:

    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U>
    struct rebind {
        typedef SlabAllocator<U> other;
    };

    template<typename Type = T>
    inline Type * allocate(const size_type n, const_pointer = nullptr) noexcept {
        static_assert(sizeof(Type) > 0, "Cannot allocate a zero-length type.");
        assert ("A memory leak will occur whenever the SlabAllocator allocates 0 items" && n > 0);
        auto ptr = static_cast<Type *>(mAllocator.Allocate(n * sizeof(Type), sizeof(void*)));
        assert ("allocator returned a null pointer. Function was likely called before Allocator creation!" && ptr);
        return ptr;
    }

    template<typename Type = T>
    inline Type * aligned_allocate(const size_type n, const size_t align, const_pointer = nullptr) noexcept {
        static_assert(sizeof(Type) > 0, "Cannot allocate a zero-length type.");
        assert ("A memory leak will occur whenever the SlabAllocator allocates 0 items" && n > 0);
        auto ptr = static_cast<Type *>(mAllocator.Allocate(n * sizeof(Type), align));
        assert ("allocator returned a null pointer. Function was likely called before Allocator creation!" && ptr);
        return ptr;
    }

    template<typename Type = T>
    inline void deallocate(Type * /*p */, size_type /* size */ = 0) noexcept {

    }

    inline size_type max_size() const {
        return std::numeric_limits<size_type>::max();
    }

    template<typename Type = T>
    inline bool operator==(SlabAllocator<Type> const & other) const noexcept {
        return this == &other;
    }

    template<typename Type = T>
    inline bool operator!=(SlabAllocator<Type> const & other) const noexcept {
        return this != &other;
    }

    inline size_type getTotalMemory() const noexcept {
        return mAllocator.getTotalMemory();
    }

    inline void Reset() {
        mAllocator.Reset();
    }

    inline void PrintStats() {
        mAllocator.PrintStats();
    }
    inline SlabAllocator() noexcept {}
    inline SlabAllocator(const SlabAllocator &) noexcept = delete;
    template <class U> inline SlabAllocator (const SlabAllocator<U> &) noexcept { }
private:
    LLVMAllocator mAllocator;
};

template <typename T = uint8_t>
class ProxyAllocator {
    using LLVMAllocator = typename SlabAllocator<T>::LLVMAllocator;
    template<typename U> friend class ProxyAllocator;
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<class U>
    struct rebind {
        typedef ProxyAllocator<U> other;
    };

    template<typename Type = T>
    inline Type * allocate(size_type n, const_pointer = nullptr) noexcept {
        static_assert(sizeof(Type) > 0, "Cannot allocate a zero-length type.");
        assert ("Cannot allocate 0 items." && n > 0);
        auto ptr = static_cast<Type *>(mAllocator->Allocate(n * sizeof(Type), sizeof(void*)));
        assert ("Allocating returned a null pointer. Function was likely called before Allocator creation!" && ptr);
        return ptr;
    }

    template<typename Type = T>
    inline Type * Allocate(size_type n) noexcept {
        return allocate<Type>(n, nullptr);
    }

    template<typename Type = T>
    inline void deallocate(Type * /*p */, size_type /* size */ = 0) noexcept {

    }

    inline size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max();
    }

    template<typename Type = T>
    inline bool operator==(ProxyAllocator<Type> const & other) const noexcept {
        return mAllocator == other.mAllocator;
    }

    template<typename Type = T>
    inline bool operator!=(ProxyAllocator<Type> const & other) const noexcept {
        return mAllocator != other.mAllocator;
    }

    inline size_type getTotalMemory() const noexcept {
        return mAllocator->getTotalMemory();
    }

    inline ProxyAllocator() noexcept = delete;
    template <class U> inline ProxyAllocator(ProxyAllocator<U> && a) noexcept : mAllocator(a.mAllocator) {}
    template <class U> inline ProxyAllocator(const ProxyAllocator<U> & a) noexcept : mAllocator(const_cast<LLVMAllocator *>(a.mAllocator)) {}
    template <class U> inline ProxyAllocator (const SlabAllocator<U> & a) noexcept : mAllocator(const_cast<LLVMAllocator *>(&a.mAllocator)) {}
private:
    LLVMAllocator * const mAllocator;
};

