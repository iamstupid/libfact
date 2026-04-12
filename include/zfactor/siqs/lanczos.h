/*--------------------------------------------------------------------
Block Lanczos over GF(2), adapted from msieve (public domain)
by Jason Papadopoulos.

This adaptation copyright (c) 2026, placed under the same
public-domain terms as the original.
--------------------------------------------------------------------*/
#pragma once
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace zfactor::siqs {

// Sparse column of a GF(2) matrix.
// data[0..weight-1] are row indices (sparse entries).
// data[weight..weight+dense_words-1] are packed dense row bits.
// cycle.list[0..cycle.num_relations-1] are relation indices forming this column.
struct LaCol {
    uint32_t* data;
    uint32_t weight;
    struct { uint32_t num_relations; uint32_t* list; } cycle;
};

// Run Block Lanczos on a nrows x ncols GF(2) matrix.
// cols: array of *ncols LaCol structures (CONSUMED -- data pointers freed).
//       After return, cols[0..*ncols-1] holds the reduced columns.
//       Each column's cycle.list is preserved so the caller can map
//       reduced column indices back to original relation indices.
// nrows: on entry, total number of rows; on exit, reduced row count.
// ncols: on entry, total number of columns; on exit, reduced column count.
// num_dense_rows: number of rows stored in packed bitfield format at end of each col.
// Returns: bitfield array of length *ncols (reduced). Bit j of bitfield[i] means
//          column i participates in null-vector dependency j.
//          Returns nullptr on failure. Caller must free() the result.
// num_deps_found: set to number of dependencies found (0-64).
uint64_t* block_lanczos(uint32_t* nrows, uint32_t num_dense_rows,
                        uint32_t* ncols, LaCol* cols,
                        uint32_t* num_deps_found);

// Free a column list (frees each col's data and cycle.list, then the array).
void free_cols(LaCol* cols, uint32_t ncols);

} // namespace zfactor::siqs
