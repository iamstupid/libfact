# zfactor — High-Performance Integer Factorization Library

## Design Document v0.1

---

## 1. Project Overview

**zfactor** is a C++ library implementing a comprehensive suite of integer factorization algorithms, designed for both native desktop (with optional GPU acceleration via WebGPU/Dawn) and web deployment (via Emscripten + browser WebGPU). The primary goals are:

1. Provide high-performance, optimized implementations of all major factorization algorithms.
2. Enable comparative benchmarking across algorithms with controlled crossover experiments.
3. Maintain a clean, zero-dependency core (no GMP) with a custom fixed-limb bigint library.
4. Support GPU-accelerated sieving for SIQS via a unified WGSL shader codebase.

Target integer range: up to ~150 decimal digits (~500 bits, 8 × u64 limbs). This covers the full practical range for QS-class algorithms and provides headroom for ECM on larger composites.

---

## 2. Repository Structure

```
zfactor/
├── CMakeLists.txt
├── src/
│   ├── core/                    # Zero-dependency bigint + modular arithmetic
│   │   ├── platform.h           # Cross-platform intrinsics (mulhi, addcarry, etc.)
│   │   ├── uint.h               # UInt<N> fixed-limb unsigned integer
│   │   ├── arith.h              # Free-function arithmetic (add, sub, mul, shift, gcd)
│   │   ├── montgomery.h         # MontCtx<N>, mul_full, sqr_full, mont_redc
│   │   ├── modular.h            # Mod<N>, ModWide<N>, ModLazy<N> type system
│   │   ├── ctx_stack.h          # Thread-local MontCtx stack + MontScope RAII
│   │   └── conv.h               # String ↔ UInt conversion, decimal I/O
│   │
│   ├── algo/                    # Factorization algorithms
│   │   ├── trial.h              # Trial division with wheel-210
│   │   ├── fermat.h             # Bounded Fermat method
│   │   ├── rho.h                # Pollard's rho (Brent + batch GCD)
│   │   ├── squfof.h             # SQUFOF with multiplier racing
│   │   ├── lehman.h             # Lehman's method
│   │   ├── pm1.h                # Pollard p-1 (stage 1 + stage 2)
│   │   ├── pp1.h                # Williams p+1 (Lucas sequences)
│   │   ├── ecm.h                # ECM (Montgomery ladder, Suyama curves)
│   │   ├── ecm_stage2.h         # ECM stage 2 (baby-step giant-step, later FFT)
│   │   ├── qs_common.h          # Shared QS infrastructure (factor base, sieve, relations)
│   │   ├── qs_basic.h           # Basic single-polynomial QS
│   │   ├── siqs.h               # Self-Initializing QS (Gray code, large prime)
│   │   ├── cfrac.h              # Continued Fraction Method (CFRAC)
│   │   ├── lanczos.h            # Block Lanczos over GF(2)
│   │   └── dispatcher.h         # Algorithm selection + full factorization pipeline
│   │
│   ├── gpu/                     # WebGPU compute (native via Dawn, web via Emscripten)
│   │   ├── gpu_context.h        # Device/queue initialization
│   │   ├── sieve_dispatch.h     # SIQS sieve kernel dispatch
│   │   └── shaders/
│   │       ├── sieve_small.wgsl # Wheel-210 + primes 11..210
│   │       └── sieve_large.wgsl # Bucket sieve for primes > 210
│   │
│   ├── util/
│   │   ├── primesieve.h         # Segmented sieve of Eratosthenes for factor base generation
│   │   ├── prng.h               # xoshiro256** or similar fast PRNG
│   │   └── timer.h              # High-resolution timing for benchmarks
│   │
│   └── main.cpp                 # CLI entry point
│
├── bench/
│   ├── bench_arith.cpp          # Micro-benchmarks for montmul, montsqr, gcd
│   ├── bench_algo.cpp           # Per-algorithm benchmarks on test suites
│   ├── bench_crossover.cpp      # Crossover experiments between algorithm pairs
│   ├── test_suite.h             # Curated semiprimes by bit-size (32..500 bit)
│   └── plot.py                  # Visualization of benchmark results
│
├── test/
│   ├── test_arith.cpp           # Correctness tests for bigint arithmetic
│   ├── test_mont.cpp            # Montgomery domain round-trip tests
│   ├── test_algo.cpp            # Factorization correctness against known answers
│   └── reference/               # Python reference outputs for regression
│
└── third_party/
    ├── mimalloc/                # High-performance allocator (git submodule)
    └── dawn/ or wgpu-native/    # WebGPU backend (git submodule)
```

---

## 3. Core Library: Fixed-Limb BigInt

### 3.1 Design Principles

- **Template parameter `N`** = number of 64-bit limbs. Compile-time constant ensures full loop unrolling.
- **Zero heap allocation.** All storage is stack-local or in-struct.
- **No virtual dispatch.** All arithmetic is resolved at compile time via templates.
- **Platform abstraction** confined to a single header (`platform.h`).

### 3.2 `UInt<N>` — Storage Type

```cpp
template<int N>
struct UInt {
    uint64_t d[N] = {};   // little-endian: d[0] is LSB
    static constexpr int limbs = N;
};
```

Supports `N` = 1..16. The `sizeof(UInt<N>)` is exactly `8*N` bytes with no padding or metadata. `N` up to 8 covers the primary target range (500 bits); larger `N` accommodates `UInt<2*N>` intermediates from multiplication (up to 1000 bits for N=8 operands).

### 3.3 Platform Intrinsics (`platform.h`)

| Operation        | MSVC (`<intrin.h>`)         | GCC/Clang (`__int128`)           |
|------------------|-----------------------------|----------------------------------|
| 64×64→128        | `_umul128(a, b, &hi)`      | `(u128)a * b`                    |
| 64×64→64 (hi)    | `_umul128` + discard lo     | `(u128)a * b >> 64`             |
| add with carry   | `_addcarry_u64`             | `u128` addition or `__builtin_addcll` |
| sub with borrow  | `_subborrow_u64`            | `u128` subtraction               |
| BMI2 mulx        | `_mulx_u64`                 | `_mulx_u64` (with `-mbmi2`)     |

### 3.4 Arithmetic Primitives (`arith.h`)

All implemented as `inline` free functions templated on `N`:

- `add<N>(r, a, b) → carry` — N-limb addition with carry chain
- `sub<N>(r, a, b) → borrow` — N-limb subtraction with borrow chain
- `csub<N>(r, a, b)` — branchless conditional subtract (for Montgomery reduction)
- `mul_full<N>(r, a, b)` — schoolbook N×N → 2N multiply
- `sqr_full<N>(r, a)` — optimized squaring (roughly half the partial products)
- `mul1<N>(r, a, scalar)` — N×1 → N+1 multiply
- `shl<N>(r, a, bits)`, `shr<N>(r, a, bits)` — multi-limb shifts
- `cmp<N>(a, b) → int` — three-way comparison
- `gcd(a, b)` — binary GCD (for final factor extraction; not performance-critical)
- `clz<N>(a)` — count leading zeros (for bit_length)

Compiler at `-O2` will fully unroll all loops for `N` ≤ 8. No `#pragma` needed.

---

## 4. Montgomery Arithmetic

### 4.1 `MontCtx<N>` — Context Object

```cpp
template<int N>
struct MontCtx {
    UInt<N> mod;          // the modulus n
    UInt<N> r2_mod;       // R² mod n  (for to_mont: REDC(a * R²) = aR mod n)
    uint64_t inv;         // -n⁻¹ mod 2⁶⁴ (only lowest limb needed for REDC)

    void init(const UInt<N>& n);  // compute all derived constants
};
```

### 4.2 Separated Multiply-then-Reduce

CPU uses a separated two-phase approach: full schoolbook multiply producing `UInt<2*N>`, followed by an independent REDC pass. This design is chosen over the integrated CIOS variant for several reasons:

- **Type system alignment.** The intermediate product is `UInt<2*N>`, which maps directly to `ModWide<N>`. CIOS requires an N+2 limb accumulator that fits neither `UInt<N>` nor `UInt<2*N>`, breaking the type abstraction.
- **Orthogonal optimization.** Multiply and reduce are independent functions. `mul_full<N>` can be swapped for `sqr_full<N>` (which exploits partial product symmetry, ~N²/2 mul64 instead of N²) without touching the reduction code. CIOS welds multiply and reduce together, requiring separate `montmul` and `montsqr` implementations.
- **Better ILP.** The schoolbook multiply phase has substantial instruction-level parallelism — independent partial products can be in-flight simultaneously on modern OoO cores. CIOS introduces a per-iteration serial dependency (`m = t[0] * inv` depends on the previous iteration's completed reduction).
- **Register pressure is acceptable.** For the target range N=4–8, the 2N-limb intermediate (8–16 × u64) is tight but manageable on x86-64's 16 GPRs. For N=8 the compiler will spill a few values to stack, but montmul is compute-bound so L1 spills are negligible.

The REDC phase processes the `UInt<2*N>` result:

```cpp
template<int N>
void mont_redc(UInt<N>& r, const UInt<2*N>& t, const MontCtx<N>& ctx) {
    UInt<2*N> tmp = t;
    for (int i = 0; i < N; i++) {
        uint64_t m = tmp[i] * ctx.inv;
        // tmp[i..i+N] += m * mod, with carry propagation
    }
    // Extract high N limbs + conditional subtract
    for (int i = 0; i < N; i++) r[i] = tmp[i + N];
    csub<N>(r, r, ctx.mod);
}
```

Carry never exceeds the 2N-limb bound (provable), so no overflow handling is needed.

**GPU note:** CIOS may still be preferable for GPU kernels where register pressure is a real constraint (e.g. 255 registers per thread on NVIDIA). This is a local decision within the WGSL shader code and does not affect the CPU type system.

### 4.3 Context Stack (`ctx_stack.h`)

```cpp
template<int N>
struct MontCtxStack {
    static constexpr int MAX_DEPTH = 4;
    const MontCtx<N>* stack[MAX_DEPTH];
    int depth = 0;

    void push(const MontCtx<N>* ctx);
    void pop();
    const MontCtx<N>& top() const;
};

template<int N>
inline thread_local MontCtxStack<N> _ctx_stack;

template<int N>
const MontCtx<N>& ctx() { return _ctx_stack<N>.top(); }
```

RAII guard:

```cpp
template<int N>
struct [[nodiscard]] MontScope {
    MontScope(const MontCtx<N>& c) { _ctx_stack<N>.push(&c); }
    ~MontScope()                    { _ctx_stack<N>.pop(); }
    // non-copyable, non-movable
};
```

Different `N` values use independent stacks (they are different template instantiations of the thread-local variable), so `MontScope<1>` (e.g. Tonelli-Shanks mod p) does not interfere with `MontScope<4>` (e.g. ECM mod n).

### 4.4 Modular Type System (`modular.h`)

Three types encode the reduction state at the type level:

| Type | Storage | Range | Meaning |
|------|---------|-------|---------|
| `Mod<N>` | `UInt<N>` | [0, mod) | Fully reduced Montgomery form |
| `ModLazy<N>` | `UInt<N>` | [0, 2·mod) or [0, 4·mod) | Add/sub result, deferred conditional subtraction |
| `ModWide<N>` | `UInt<2*N>` | [0, mod·R) | Unreduced product, awaiting REDC |

Operator overloads enforce valid transitions:

| Expression | Return Type | Operation |
|---|---|---|
| `Mod + Mod` | `ModLazy` | N-limb add, no conditional sub |
| `Mod - Mod` | `ModLazy` | N-limb sub, no conditional add-back |
| `ModLazy + ModLazy` | `ModLazy` | N-limb add (bound grows to 4·mod) |
| `Mod * Mod` | `ModWide` | schoolbook N×N, no REDC |
| `ModLazy * Mod` | `ModWide` | schoolbook, lazy input is fine since < 2·mod |
| `ModWide + ModWide` | `ModWide` | 2N-limb add |
| `ModWide + Mod` | `ModWide` | add Mod into low N limbs of wide |
| `ModWide - ModWide` | `ModWide` | 2N-limb sub |
| `ModWide.redc()` | `Mod` | Montgomery reduction via `ctx<N>()` |
| `ModLazy.reduce()` | `Mod` | conditional subtraction via `ctx<N>()` |
| `Mod.sqr()` | `ModWide` | optimized squaring, no REDC |

Illegal operations (e.g. `ModWide * ModWide`) are not defined and produce compile-time errors.

Implicit conversion from `ModLazy` to `Mod` (via `operator Mod<N>()`) calls `reduce()`. This allows seamless usage where a `Mod` is expected but a `ModLazy` is available.

`Mod<N>` values do not store a context pointer. All operations access the context via `ctx<N>()` (thread-local stack top). This means `sizeof(Mod<N>) == sizeof(UInt<N>) == 8*N`.

### 4.5 Lazy REDC Fusion

The primary fusion opportunity is in ECC point operations, where patterns like `a*b + c` or `a*b - c*d` can accumulate in 2N-limb space and REDC once instead of twice. Example in Montgomery curve xDBL:

```
u = X + Z        → ModLazy  (no csub)
v = X - Z        → ModLazy  (no csub)
uu = (u * u)     → ModWide  (no REDC yet)
...
```

The type system makes this automatic — the programmer writes natural expressions and the types determine when reduction actually happens.

---

## 5. Factorization Algorithms

### 5.1 Dispatch by Limb Count

```cpp
int factor(const char* n_str) {
    int bits = count_bits(n_str);
    int limbs = (bits + 63) / 64;
    switch (limbs) {
        case 1: return factor_impl<1>(n_str);
        case 2: return factor_impl<2>(n_str);
        // ... up to case 8
        default: return factor_generic(n_str);
    }
}
```

After this single runtime dispatch, all internal computation is monomorphized.

### 5.2 Algorithm Inventory

#### Phase 1 — Small Factor Algorithms

**Trial Division (wheel-210)**
- Precomputed prime table up to configurable bound (default 10⁶).
- Wheel with 2, 3, 5, 7 eliminates 77% of candidates.
- Used as mandatory preprocessing for all inputs.

**Pollard's Rho (Brent variant)**
- Iteration function: f(x) = x² + c in Montgomery form.
- Brent cycle detection (single function evaluation per step).
- Batch GCD: accumulate |x−y| products for m steps (m ≈ 128), one GCD per batch.
- Restart with new c after configurable iteration limit.
- Target range: factors up to ~40 bits.

**SQUFOF (Square Form Factorization)**
- O(n^{1/4}) worst case with tiny constant.
- Multiplier racing: try k = 1, 3, 5, 7, 11, ... in parallel (round-robin), first to find an ambiguous form wins.
- Pure 64-bit arithmetic (128-bit intermediate for the multiply). No Montgomery needed.
- Target range: n up to 62 bits (single limb).

**Fermat's Method (bounded)**
- Run for a fixed small number of iterations (e.g. 10000).
- Catches factors near √n essentially for free.
- Zero implementation complexity.

**Lehman's Method**
- O(n^{1/3}) deterministic.
- Fills the gap between trial division and rho for 21-50 digit range.

#### Phase 2 — Medium Factor Algorithms

**Pollard p−1 (stage 1 + stage 2)**
- Stage 1: compute a^{lcm(1..B1)} mod n via repeated exponentiation.
- Stage 2: standard baby-step giant-step continuation in (B1, B2].
- Use Montgomery exponentiation with binary method.
- Cheap screening pass before ECM.

**Williams p+1**
- Lucas sequence analogue of p−1.
- Catches primes where p+1 is smooth but p−1 is not.
- Same two-stage structure.

**ECM (Elliptic Curve Method)**
- Stage 1: Montgomery curve with Suyama parametrization (12-torsion).
  - Montgomery ladder (PRAC chains for near-optimal differential addition chains).
  - Accumulate prime powers into batches before ladder evaluation.
  - Detect factor via gcd(Z, n) — Z can remain in Montgomery form since gcd(R,n) = 1.
- Stage 2: convert to Weierstrass for full point addition.
  - Baby-step giant-step with prime-gap jump table.
  - Later upgrade: FFT continuation with Brent-Suyama extension (Dickson polynomials).
- Multi-curve: each curve is independent, parallelize via std::thread (one curve per thread).
- Automatic B1/B2/nCurves selection based on input size.

#### Phase 3 — Subexponential Algorithms (Congruence of Squares)

**Shared Infrastructure (`qs_common.h`)**
- Factor base generation: primes p where n is a quadratic residue mod p (Euler criterion).
- Tonelli-Shanks for computing √n mod p.
- Relation collector: full relations + partial relations (single large prime variant).
  - Partial relation combining via hash map on the large prime cofactor.
- Block Lanczos over GF(2) for null space computation.
  - Sparse matrix in coordinate format (row, col pairs).
  - 64-bit word packing for block operations.

**Basic QS**
- Single polynomial: f(x) = x² − n.
- Log-approximation sieve on interval [−M, M].
- Serves as baseline for measuring SIQS improvements.

**SIQS (Self-Initializing Quadratic Sieve)**
- Polynomial generation: choose A = product of s primes from factor base, Gray code iteration for B coefficients.
- Sieve with log-approximation: accumulate log(p) into byte array.
- Small primes (p < 50) skipped in sieve, accounted for in threshold.
- Trial division confirmation of smooth candidates.
- Large prime variant: cofactor up to B1 × K stored as partial relation.
- Double large prime variant (Phase 4 enhancement).
- Linear algebra: Block Lanczos (internal C++ implementation).

**CFRAC (Continued Fraction Method)**
- Convergent computation of √n via continued fraction expansion.
- Each convergent p_k/q_k yields p_k² ≡ (−1)^k · Q_k (mod n).
- Q_k values are small, increasing smoothness probability.
- Uses same CongruenceSolver as QS.
- Historical comparison baseline.

### 5.3 Algorithm Pipeline (`dispatcher.h`)

The full factorization pipeline for a composite n:

```
1. trial_division(n, bound=10⁶)       → strip small factors
2. if n fits 64 bits:
     a. squfof(n)                      → fast deterministic
     b. if fail: rho(n)               → probabilistic backup
3. if n fits 128 bits:
     a. rho(n)                         → try rho first
     b. if fail: ecm(n, small bounds)  → quick ECM screening
4. for larger n:
     a. fermat(n, 10000 iterations)    → cheap check
     b. pm1(n), pp1(n)                → cheap screening
     c. ecm(n, auto-parameterized)    → hunt for medium factors
     d. siqs(n)                        → general-purpose
5. Recurse on remaining cofactor.
```

For benchmarking mode, each algorithm is called independently with timing instrumentation.

---

## 6. GPU Acceleration (SIQS Sieving)

### 6.1 Approach

Use **WebGPU API** (via `webgpu.h`) for a single codebase that runs natively (Dawn or wgpu-native) and in-browser (Emscripten).

Shaders are written in **WGSL**, the only shader language needed.

### 6.2 Sieve Kernel Architecture

Each workgroup processes one polynomial. Three-layer sieve strategy:

| Layer | Primes | Method | Location |
|---|---|---|---|
| Layer 0 | 2, 3, 5, 7 | Precomputed wheel-210 pattern baked into initial sieve values | CPU init |
| Layer 1 | 11 ≤ p ≤ 210 | Dense sieve kernel, each thread handles one prime | Workgroup shared memory |
| Layer 2 | p > 210 | Bucket sieve with sorted endpoint list | Storage buffer (global memory) |

CPU prepares polynomial coefficients and sieve offsets, uploads to GPU via storage buffers. GPU runs sieve kernel, CPU reads back threshold-exceeding candidates for trial division confirmation.

### 6.3 Build Targets

| Target | GPU Backend | BigInt Backend | Build Flag |
|---|---|---|---|
| Native Linux/macOS/Windows | Dawn (Vulkan/Metal/D3D12) | Native C++ with mulx | Default |
| Web (Emscripten) | Browser WebGPU | Wasm (no `__int128`, emulated) | `-sUSE_WEBGPU=1 -sASYNCIFY` |
| CPU-only fallback | None | Same as native | `-DZFACTOR_NO_GPU` |

---

## 7. Memory Allocator: mimalloc

### 7.1 Rationale

The core bigint arithmetic uses zero heap allocation (fixed-limb stack values). However, several subsystems allocate heavily at runtime:

- **Bucket sieve:** Dynamic arrays for each bucket, appended to during sieve fill, consumed and cleared per sieve block. High-frequency small allocations with burst realloc patterns.
- **Relation collector:** Hash maps for partial relations, growing dynamically as sieving progresses.
- **Block Lanczos:** Sparse matrix construction (coordinate lists), dense block vectors.
- **Factor base / prime tables:** Large contiguous arrays allocated once at init.
- **ECM multi-curve:** Per-thread working memory for independent curve computations.

The default system allocator (`malloc`/`free`) is not optimized for these patterns — particularly the multi-threaded ECM case where per-thread allocation/deallocation causes contention on global heap locks.

### 7.2 Integration

[mimalloc](https://github.com/microsoft/mimalloc) (Microsoft, MIT license) is used as a drop-in global allocator replacement. Integration is via CMake:

```cmake
add_subdirectory(third_party/mimalloc)

# Option A: compile-time override (preferred)
target_link_libraries(zfactor PRIVATE mimalloc-static)
# mimalloc's CMake automatically sets up malloc/free override

# Option B: explicit include for mi_* prefixed API
target_link_libraries(zfactor PRIVATE mimalloc)
```

On native targets, mimalloc overrides `malloc`/`free`/`realloc` globally — no code changes needed. The bucket sieve's `realloc` pattern (double-on-full) maps directly to mimalloc's segment-based growth, avoiding page-level fragmentation that system allocators suffer from.

For Emscripten (web target), mimalloc is not used — Emscripten's `dlmalloc` is the only practical option in Wasm. The `CMakeLists.txt` conditionally excludes mimalloc:

```cmake
if (NOT EMSCRIPTEN)
    add_subdirectory(third_party/mimalloc)
    target_link_libraries(zfactor PRIVATE mimalloc-static)
endif()
```

### 7.3 Key Benefits for This Project

| Subsystem | Allocation Pattern | mimalloc Advantage |
|---|---|---|
| Bucket sieve | Burst append + realloc, then bulk consume + clear | Free-list sharding per size class; realloc in-place when possible |
| ECM multi-thread | Per-thread alloc/free of curve working memory | Thread-local heaps eliminate lock contention entirely |
| Relation collector | Hash map insert-heavy, occasional rehash | Reduced fragmentation from segment-based pages |
| Block Lanczos | Large sparse matrix build, then free all at once | `mi_heap_destroy` can release an entire heap in O(1) |

For Block Lanczos specifically, mimalloc's heap API enables arena-style allocation:

```cpp
mi_heap_t* la_heap = mi_heap_new();
// ... all Lanczos allocations via mi_heap_malloc(la_heap, size) ...
mi_heap_destroy(la_heap);  // instant bulk free, no per-object destructor
```

---

## 8. Benchmarking Framework

### 8.1 Test Suite

Curated semiprimes n = p × q organized by bit-length of n, with controlled factor ratios:

- **Balanced**: p and q have similar bit length (hardest case for all algorithms).
- **Unbalanced**: one factor is significantly smaller (favors rho/ECM).
- **Structured**: p−1 or p+1 is smooth (favors p−1/p+1 methods).

Bit-length range: 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 384, 448, 500 bits.

At least 20 random instances per (bit-length, factor-ratio) pair for statistical significance.

### 8.2 Crossover Experiments

Key crossover points to measure:

| Pair | Expected Crossover | What It Tells Us |
|---|---|---|
| Rho vs SQUFOF | ~40-50 bit n | When cycle detection loses to algebraic structure |
| Rho vs ECM | ~50-60 bit factors | When random walk loses to group structure |
| ECM vs SIQS | ~40-50 digit n | When factor-size-dependent beats input-size-dependent |
| Basic QS vs SIQS | ~60-70 digit n | Quantifies self-initialization overhead/benefit |
| CFRAC vs SIQS | ~40-60 digit n | Historical comparison |
| p−1/p+1 vs ECM | By smoothness of p±1 | When cheap screening pays off |
| CPU sieve vs GPU sieve | ~50-80 digit n | GPU launch overhead vs throughput |

### 8.3 Micro-benchmarks

- `montmul<N>` throughput and latency for N = 1..8
- `montsqr<N>` vs `montmul<N>` speedup ratio
- `gcd` performance by input size
- Sieve throughput (bytes/second) by factor base size and interval length

### 8.4 Output Format

All benchmarks emit CSV:

```
algorithm,n_bits,factor_bits,factor_ratio,trial,time_ns,success
siqs,256,128,balanced,7,1234567890,1
ecm,256,40,unbalanced,3,567890123,1
```

`plot.py` generates log-scale crossover charts (like the ones from the Python notebook).

---

## 9. Implementation Roadmap

### Phase 1 — Core + Small Algorithms (Weeks 1-3)

- [ ] `platform.h` — intrinsics with MSVC/GCC/Clang paths
- [ ] `UInt<N>` — storage + comparison
- [ ] `arith.h` — add, sub, csub, mul_full, sqr_full, mul1, shift, gcd
- [ ] `MontCtx<N>` — init, montmul, montsqr
- [ ] `Mod<N>`, `ModLazy<N>`, `ModWide<N>` — type system with operator overloads
- [ ] `MontScope` + thread-local context stack
- [ ] Correctness tests (round-trip mont, arithmetic identities)
- [ ] `trial.h` — wheel-210
- [ ] `rho.h` — Brent + batch GCD
- [ ] `squfof.h` — with multiplier racing
- [ ] `fermat.h` — bounded iterations
- [ ] Benchmark: rho vs SQUFOF vs Fermat on 32-64 bit semiprimes

### Phase 2 — Medium Factor Algorithms (Weeks 4-6)

- [ ] `pm1.h` — Pollard p−1 (stage 1 + stage 2)
- [ ] `pp1.h` — Williams p+1
- [ ] `ecm.h` — Montgomery ladder, Suyama parametrization
- [ ] `ecm_stage2.h` — baby-step giant-step
- [ ] `conv.h` — decimal string I/O
- [ ] Benchmark: p−1 vs p+1 vs ECM on 30-60 digit factors

### Phase 3 — Congruence of Squares (Weeks 7-10)

- [ ] `primesieve.h` — segmented sieve for factor base generation
- [ ] `qs_common.h` — factor base, Tonelli-Shanks, relation collector, partial combining
- [ ] `lanczos.h` — Block Lanczos over GF(2)
- [ ] `qs_basic.h` — single-polynomial QS (baseline)
- [ ] `siqs.h` — full SIQS with Gray code, large prime
- [ ] `cfrac.h` — CFRAC
- [ ] `dispatcher.h` — algorithm selection pipeline
- [ ] Benchmark: ECM vs QS vs SIQS crossover at 40-80 digits

### Phase 4 — Optimization + GPU (Weeks 11-14)

- [ ] SIQS double large prime variant
- [ ] ECM stage 2 FFT continuation (Brent-Suyama extension)
- [ ] `lehman.h` — Lehman's method
- [ ] Dixon's random squares (historical baseline)
- [ ] GPU context initialization (Dawn/wgpu-native)
- [ ] SIQS sieve GPU kernel (wheel-210 layered architecture)
- [ ] Benchmark: CPU vs GPU sieve crossover
- [ ] Comprehensive crossover chart across all algorithms

### Phase 5 — Web Deployment (Weeks 15-16)

- [ ] Emscripten build configuration
- [ ] WebGPU path via `-sUSE_WEBGPU=1`
- [ ] CPU-only fallback for browsers without WebGPU
- [ ] Simple HTML/JS frontend for interactive factorization

---

## 10. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| BigInt representation | Fixed-limb `UInt<N>` template | Full compile-time unrolling, zero heap allocation |
| Modular arithmetic | Separated multiply-then-reduce (CPU), CIOS (GPU) | Type-system-clean UInt<2N> intermediate; better ILP on OoO cores; CIOS for GPU register pressure |
| Context passing | Thread-local stack | Zero per-value overhead, safe nesting, automatic cleanup via RAII |
| Reduction tracking | Three-type system (Mod/ModLazy/ModWide) | Compile-time safety, enables lazy REDC fusion |
| ECM curve form | Montgomery XZ (stage 1) → Weierstrass (stage 2) | Montgomery ladder is simplest and fast; Weierstrass needed for stage 2 addition |
| ECM parallelism | std::thread (one curve per thread) | Embarrassingly parallel; SIMD not worth the complexity for ECM |
| SIQS sieve GPU | WebGPU/WGSL | Single codebase for native + web |
| GPU sieve architecture | Three-layer (wheel bake-in / small shared memory / large bucket) | Eliminates workload divergence within each layer |
| Linear algebra | Block Lanczos (internal C++) | Standard for QS-scale matrices; simpler than Block Wiedemann |
| External dependencies | mimalloc (allocator) only for core library | Maximizes portability; Dawn/wgpu-native optional for GPU; mimalloc excluded on Emscripten |
| GNFS | Explicitly excluded | Engineering cost too high for target integer range |

---

## 11. References

### Core Algorithms
- Montgomery, "Modular Multiplication Without Trial Division", Math. Comp. 1985
- Brent, "An Improved Monte Carlo Factorization Algorithm", BIT 1980
- Shanks, "Class Number, a Theory of Factorization", 1971 (SQUFOF)
- Lenstra, "Factoring Integers with Elliptic Curves", Annals of Math. 1987
- Pomerance, "The Quadratic Sieve Factoring Algorithm", Eurocrypt 1984
- Contini, "Factoring Integers with the Self-Initializing Quadratic Sieve", 1997
- Morrison & Brillhart, "A Method of Factoring and the Factorization of F₇", Math. Comp. 1975 (CFRAC)
- Montgomery, "A Block Lanczos Algorithm for Finding Dependencies over GF(2)", Eurocrypt 1995

### ECM Optimizations
- Zimmermann & Dodson, "20 Years of ECM", ANTS VII 2006
- Bernstein, Birkner, Lange, Peters, "ECM Using Edwards Curves", Math. Comp. 2013
- Bos & Kleinjung, "ECM at Work", ASIACRYPT 2012
- Bouvier & Imbert, "Faster Cofactorization with ECM Using Mixed Representations", EUROCRYPT 2020

### GPU
- Schmidt, Aribowo, Dang, "Iterative Sparse Matrix-Vector Multiplication for Integer Factorization on GPUs", Euro-Par 2011
- Buhrow, "CUDA Sieve of Eratosthenes" (CUDASieve)
- Wang, "Quadratic Sieve on GPUs", MIT 18.337 2011
