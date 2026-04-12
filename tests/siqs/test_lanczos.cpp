/*--------------------------------------------------------------------
Smoke test for Block Lanczos over GF(2).
Constructs a known sparse matrix, runs block_lanczos, and verifies
that the returned dependencies are valid (i.e., Ax = 0 in GF(2)).
--------------------------------------------------------------------*/

#include <zfactor/siqs/lanczos.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <set>

using namespace zfactor::siqs;

/* Build a random sparse GF(2) matrix with the given dimensions.
   Returns a malloc'd array of LaCol. Each column has avg_weight
   random row entries drawn from [0, nrows). Relation indices
   in cycle.list are trivially set to {col_index}.
   num_dense_rows is set to 0 (all sparse). */
static LaCol* build_random_matrix(uint32_t nrows, uint32_t ncols,
                                   uint32_t avg_weight, uint32_t seed) {
    LaCol* cols = (LaCol*)calloc(ncols, sizeof(LaCol));
    std::mt19937 rng(seed);

    for (uint32_t i = 0; i < ncols; i++) {
        /* pick weight in [avg_weight/2, avg_weight*3/2] */
        uint32_t w = avg_weight / 2 + rng() % (avg_weight + 1);
        if (w > nrows) w = nrows;

        /* pick w distinct random rows */
        std::set<uint32_t> row_set;
        while (row_set.size() < w)
            row_set.insert(rng() % nrows);

        std::vector<uint32_t> rows(row_set.begin(), row_set.end());

        /* no dense rows, so data is just the sparse entries */
        cols[i].data = (uint32_t*)malloc(w * sizeof(uint32_t));
        memcpy(cols[i].data, rows.data(), w * sizeof(uint32_t));
        cols[i].weight = w;

        /* cycle list: just {i} */
        cols[i].cycle.num_relations = 1;
        cols[i].cycle.list = (uint32_t*)malloc(sizeof(uint32_t));
        cols[i].cycle.list[0] = i;
    }
    return cols;
}

/* Build the original matrix as a vector of sets for easy verification.
   Returns orig_cols[col_index] = set of row indices. */
static std::vector<std::vector<uint32_t>> build_original_matrix(
        uint32_t nrows, uint32_t ncols,
        uint32_t avg_weight, uint32_t seed) {
    (void)nrows;
    LaCol* cols = build_random_matrix(nrows, ncols, avg_weight, seed);
    std::vector<std::vector<uint32_t>> result(ncols);
    for (uint32_t i = 0; i < ncols; i++) {
        result[i].assign(cols[i].data, cols[i].data + cols[i].weight);
        free(cols[i].data);
        free(cols[i].cycle.list);
    }
    free(cols);
    return result;
}

/* Verify dependencies using the cycle lists.
   For each dependency bit j, collect all original relation indices
   by iterating over the reduced columns and their cycle.lists.
   Then XOR the corresponding original matrix columns and check
   the result is the zero vector. */
static bool verify_deps_via_cycles(
        const std::vector<std::vector<uint32_t>>& orig_matrix,
        uint32_t orig_nrows,
        uint32_t reduced_ncols,
        LaCol* reduced_cols,
        uint64_t* deps,
        uint32_t num_deps) {

    if (num_deps == 0) return false;

    bool all_ok = true;
    for (uint32_t dep = 0; dep < num_deps; dep++) {
        /* Collect all original relation indices for this dep */
        std::vector<uint32_t> orig_relations;
        uint32_t cols_in_dep = 0;

        for (uint32_t i = 0; i < reduced_ncols; i++) {
            if (deps[i] & ((uint64_t)1 << dep)) {
                cols_in_dep++;
                for (uint32_t r = 0; r < reduced_cols[i].cycle.num_relations; r++) {
                    orig_relations.push_back(reduced_cols[i].cycle.list[r]);
                }
            }
        }

        /* XOR the original columns for each relation.
           Count parity of each row. */
        std::vector<uint32_t> row_count(orig_nrows, 0);
        for (uint32_t rel : orig_relations) {
            for (uint32_t row : orig_matrix[rel]) {
                row_count[row] ^= 1;
            }
        }

        /* All row counts should be 0 (mod 2) */
        bool ok = (cols_in_dep >= 2); /* non-trivial */
        for (uint32_t r = 0; r < orig_nrows; r++) {
            if (row_count[r] != 0) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            printf("  dependency %u FAILED (cols_in_dep=%u, rels=%zu)\n",
                   dep, cols_in_dep, orig_relations.size());
            all_ok = false;
        }
    }

    return all_ok;
}

static void test_small() {
    const uint32_t orig_nrows = 200;
    const uint32_t orig_ncols = 210;
    const uint32_t avg_weight = 8;
    const uint32_t seed = 42;

    printf("test_small: %u x %u matrix, avg_weight=%u\n",
           orig_nrows, orig_ncols, avg_weight);

    /* Save original matrix for verification */
    auto orig_matrix = build_original_matrix(orig_nrows, orig_ncols, avg_weight, seed);

    /* Build the matrix to be consumed by lanczos */
    LaCol* cols = build_random_matrix(orig_nrows, orig_ncols, avg_weight, seed);

    uint32_t nrows = orig_nrows;
    uint32_t ncols = orig_ncols;
    uint32_t num_deps = 0;
    uint64_t* deps = block_lanczos(&nrows, 0, &ncols, cols, &num_deps);

    printf("  reduced to %u x %u, found %u dependencies\n",
           nrows, ncols, num_deps);

    if (deps && num_deps > 0) {
        bool ok = verify_deps_via_cycles(orig_matrix, orig_nrows,
                                          ncols, cols, deps, num_deps);
        printf("  verification: %s\n", ok ? "PASS" : "FAIL");
        assert(ok);
        free(deps);
    } else {
        printf("  block_lanczos returned no deps (may be OK for tiny matrix)\n");
    }

    /* Free remaining cycle lists (data was consumed by lanczos) */
    for (uint32_t i = 0; i < ncols; i++)
        free(cols[i].cycle.list);
    free(cols);
}

static void test_medium() {
    const uint32_t orig_nrows = 2000;
    const uint32_t orig_ncols = 2040;
    const uint32_t avg_weight = 15;
    const uint32_t seed = 123;

    printf("test_medium: %u x %u matrix, avg_weight=%u\n",
           orig_nrows, orig_ncols, avg_weight);

    auto orig_matrix = build_original_matrix(orig_nrows, orig_ncols, avg_weight, seed);

    LaCol* cols = build_random_matrix(orig_nrows, orig_ncols, avg_weight, seed);

    uint32_t nrows = orig_nrows;
    uint32_t ncols = orig_ncols;
    uint32_t num_deps = 0;
    uint64_t* deps = block_lanczos(&nrows, 0, &ncols, cols, &num_deps);

    printf("  reduced to %u x %u, found %u dependencies\n",
           nrows, ncols, num_deps);

    if (deps && num_deps > 0) {
        bool ok = verify_deps_via_cycles(orig_matrix, orig_nrows,
                                          ncols, cols, deps, num_deps);
        printf("  verification: %s\n", ok ? "PASS" : "FAIL");
        assert(ok);
        free(deps);
    } else {
        printf("  block_lanczos returned no deps\n");
        assert(false && "expected dependencies for medium matrix");
    }

    for (uint32_t i = 0; i < ncols; i++)
        free(cols[i].cycle.list);
    free(cols);
}

static void test_larger() {
    const uint32_t orig_nrows = 5000;
    const uint32_t orig_ncols = 5100;
    const uint32_t avg_weight = 20;
    const uint32_t seed = 999;

    printf("test_larger: %u x %u matrix, avg_weight=%u\n",
           orig_nrows, orig_ncols, avg_weight);

    auto orig_matrix = build_original_matrix(orig_nrows, orig_ncols, avg_weight, seed);

    LaCol* cols = build_random_matrix(orig_nrows, orig_ncols, avg_weight, seed);

    uint32_t nrows = orig_nrows;
    uint32_t ncols = orig_ncols;
    uint32_t num_deps = 0;
    uint64_t* deps = block_lanczos(&nrows, 0, &ncols, cols, &num_deps);

    printf("  reduced to %u x %u, found %u dependencies\n",
           nrows, ncols, num_deps);

    if (deps && num_deps > 0) {
        bool ok = verify_deps_via_cycles(orig_matrix, orig_nrows,
                                          ncols, cols, deps, num_deps);
        printf("  verification: %s\n", ok ? "PASS" : "FAIL");
        assert(ok);
        free(deps);
    } else {
        printf("  block_lanczos returned no deps\n");
        assert(false && "expected dependencies for larger matrix");
    }

    for (uint32_t i = 0; i < ncols; i++)
        free(cols[i].cycle.list);
    free(cols);
}

int main() {
    test_small();
    test_medium();
    test_larger();
    printf("All tests passed.\n");
    return 0;
}
