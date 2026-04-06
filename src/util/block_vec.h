#pragma once
#include <array>
#include <vector>
#include <memory>
namespace zfactor {
    template<typename T, uint32_t bpa>
    struct block_vec {
        using block_type = std::array<T, bpa>;
        std::vector<std::unique_ptr<block_type>> blocks;
        uint32_t size_ = 0;

        T& operator[](uint32_t i) {
            return (*blocks[i / bpa])[i % bpa];
        }
        const T& operator[](uint32_t i) const {
            return (*blocks[i / bpa])[i % bpa];
        }

        T* operator+(uint32_t i) {
            return blocks[i / bpa]->data() + (i % bpa);
        }
        const T* operator+(uint32_t i) const {
            return blocks[i / bpa]->data() + (i % bpa);
        }

        uint32_t size() const { return size_; }
        uint32_t capacity() const { return (uint32_t)blocks.size() * bpa; }
        bool empty() const { return size_ == 0; }

        T& push_back(const T& val) {
            if (size_ == capacity())
                blocks.push_back(std::make_unique<block_type>());
            T& slot = (*this)[size_++];
            slot = val;
            return slot;
        }

        T& push_back(T&& val) {
            if (size_ == capacity())
                blocks.push_back(std::make_unique<block_type>());
            T& slot = (*this)[size_++];
            slot = std::move(val);
            return slot;
        }

        T* no_init_extend(){
            if (size_ == capacity())
                blocks.push_back(std::make_unique<block_type>());
            T& slot = (*this)[size_++];
            return slot;
        }

        template<typename... Args>
        T& emplace_back(Args&&... args) {
            if (size_ == capacity())
                blocks.push_back(std::make_unique<block_type>());
            T& slot = (*this)[size_];
            new (&slot) T(std::forward<Args>(args)...);
            size_++;
            return slot;
        }

        T& back() { return (*this)[size_ - 1]; }
        const T& back() const { return (*this)[size_ - 1]; }

        void pop_back() { --size_; }

        void clear() { size_ = 0; }

        void reset() {
            blocks.clear();
            size_ = 0;
        }

        void resize(uint32_t n) {
            while (n > capacity())
                blocks.push_back(std::make_unique<block_type>());
            size_ = n;
        }

        void resize(uint32_t n, const T& val) {
            uint32_t old = size_;
            resize(n);
            for (uint32_t i = old; i < n; i++)
                (*this)[i] = val;
        }

        // Contiguous data pointer within one block. Valid for
        // indices [i - i%bpa, i - i%bpa + bpa). Does NOT span blocks.
        T* block_ptr(uint32_t i) { return blocks[i / bpa]->data(); }
        const T* block_ptr(uint32_t i) const { return blocks[i / bpa]->data(); }

        // Number of valid elements in the block containing index i
        uint32_t block_count(uint32_t i) const {
            uint32_t block_start = (i / bpa) * bpa;
            return std::min(bpa, size_ - block_start);
        }

        template<typename F>
        void for_each_block(F&& fn) {
            uint32_t remaining = size_;
            for (auto& blk : blocks) {
                uint32_t n = std::min(remaining, bpa);
                fn(blk->data(), n);
                remaining -= n;
                if (!remaining) break;
            }
        }

        template<typename F>
        void for_each_block(F&& fn) const {
            uint32_t remaining = size_;
            for (const auto& blk : blocks) {
                uint32_t n = std::min(remaining, bpa);
                fn(blk->data(), n);
                remaining -= n;
                if (!remaining) break;
            }
        }
    };
}