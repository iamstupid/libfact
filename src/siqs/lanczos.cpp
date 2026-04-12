/*--------------------------------------------------------------------
Block Lanczos over GF(2).

Adapted from msieve by Jason Papadopoulos (public domain).
This file is a self-contained, single-threaded C++ adaptation
of the msieve Block Lanczos implementation. Threading, checkpoint
save/restore, platform-specific 32-bit ASM, and msieve_obj
dependencies have been removed. The generic C paths and GCC x86-64
ASM paths are retained.

Original license:
  This source distribution is placed in the public domain by its
  author, Jason Papadopoulos. You may use it for any purpose, free
  of charge, without having to notify anyone.
--------------------------------------------------------------------*/

#include <zfactor/siqs/lanczos.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdint>

/* ---- compatibility shim ---------------------------------------- */

typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef int32_t  int32;
typedef int64_t  int64;

#define INLINE inline

#ifdef __GNUC__
#define PREFETCH(addr) __builtin_prefetch(addr)
#else
#define PREFETCH(addr) ((void)0)
#endif

/* Detect x86-64 GCC inline asm availability.
   We define GCC_ASM64X when compiling 64-bit with GCC/Clang. */
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
  #define GCC_ASM64X
  #define ASM_G __asm__
  #define ALIGN_LOOP ".p2align 4,,7 \n\t"
#endif

/* ---- memory helpers --------------------------------------------- */

static INLINE void *xmalloc(size_t len) {
    void *ptr = malloc(len);
    if (!ptr) { fprintf(stderr, "xmalloc failed (%zu)\n", len); abort(); }
    return ptr;
}

static INLINE void *xcalloc(size_t num, size_t len) {
    void *ptr = calloc(num, len);
    if (!ptr) { fprintf(stderr, "xcalloc failed (%zu)\n", num*len); abort(); }
    return ptr;
}

static INLINE void *xrealloc(void *p, size_t len) {
    void *ptr = realloc(p, len);
    if (!ptr) { fprintf(stderr, "xrealloc failed (%zu)\n", len); abort(); }
    return ptr;
}

/* ---- PRNG (Marsaglia MWC) -------------------------------------- */

static INLINE uint32 get_rand(uint32 *seed, uint32 *carry) {
    #define RAND_MULT 2131995753
    uint64 temp = (uint64)(*seed) * (uint64)RAND_MULT + (uint64)(*carry);
    *seed  = (uint32)temp;
    *carry = (uint32)(temp >> 32);
    return (uint32)temp;
    #undef RAND_MULT
}

/* ---- minimal mp_t (only rshift + clear needed) ------------------ */

#define MAX_MP_WORDS 30

struct mp_t {
    uint32 nwords;
    uint32 val[MAX_MP_WORDS];
};

static INLINE void mp_clear(mp_t *a) {
    memset(a, 0, sizeof(mp_t));
}

static uint32 num_nonzero_words(uint32 *x, uint32 max_words) {
    uint32 i;
    for (i = max_words; i && !x[i-1]; i--)
        ;
    return i;
}

static void mp_rshift(mp_t *a, uint32 shift, mp_t *res) {
    int32 i;
    int32 words = (int32)a->nwords;
    int32 start_word = (int32)(shift / 32);
    uint32 word_shift = shift & 31;
    uint32 comp_word_shift = 32 - word_shift;

    if (start_word > words) {
        mp_clear(res);
        return;
    }

    if (word_shift == 0) {
        for (i = 0; i < (words - start_word); i++)
            res->val[i] = a->val[start_word + i];
    } else {
        for (i = 0; i < (words - start_word - 1); i++)
            res->val[i] = a->val[start_word + i] >> word_shift |
                           a->val[start_word + i + 1] << comp_word_shift;
        res->val[i] = a->val[start_word + i] >> word_shift;
        i++;
    }
    for (; i < MAX_MP_WORDS; i++)
        res->val[i] = 0;

    res->nwords = num_nonzero_words(res->val, (uint32)(words - start_word));
}

/* ---- la_col_t alias --------------------------------------------- */

typedef zfactor::siqs::LaCol la_col_t;

/* ---- merge_relations (symmetric difference) --------------------- */

static uint32 merge_relations(uint32 *merge_array,
                               uint32 *src1, uint32 n1,
                               uint32 *src2, uint32 n2) {
    uint32 i1 = 0, i2 = 0, num_merge = 0;
    while (i1 < n1 && i2 < n2) {
        uint32 v1 = src1[i1], v2 = src2[i2];
        if (v1 < v2)      { merge_array[num_merge++] = v1; i1++; }
        else if (v1 > v2) { merge_array[num_merge++] = v2; i2++; }
        else               { i1++; i2++; }
    }
    while (i1 < n1) merge_array[num_merge++] = src1[i1++];
    while (i2 < n2) merge_array[num_merge++] = src2[i2++];
    return num_merge;
}

/* ---- suppress logging ------------------------------------------- */

#ifdef LANCZOS_DEBUG
#define logprintf(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#else
#define logprintf(...) do {} while(0)
#endif

/* ---- constants from msieve lanczos ------------------------------ */

#define POST_LANCZOS_ROWS 48
#define MIN_POST_LANCZOS_DIM 10000
#define MIN_NCOLS_TO_PACK 30000
#define NUM_MEDIUM_ROWS 3000

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

/* ================================================================
   Internal packed-matrix types (single-threaded, no msieve_obj)
   ================================================================ */

struct entry_idx_t {
    uint16 row_off;
    uint16 col_off;
};

struct packed_block_t {
    uint32 start_row;
    uint32 start_col;
    uint32 num_rows;
    uint32 num_entries;
    uint32 num_entries_alloc;
    entry_idx_t *entries;
    uint16 *med_entries;
};

/* single-thread data */
struct thread_data_t {
    uint32 my_oid;
    la_col_t *initial_cols;
    uint32 col_min;
    uint32 col_max;
    uint32 nrows_in;
    uint32 ncols_in;
    uint32 block_size;

    uint32 ncols;
    uint32 num_dense_rows;
    uint64 **dense_blocks;
    uint32 num_blocks;
    uint64 *x;
    uint64 *b;
    packed_block_t *blocks;
};

struct packed_matrix_t {
    uint32 nrows;
    uint32 ncols;
    uint32 num_dense_rows;
    uint32 num_threads; /* always 1 */

    la_col_t *unpacked_cols;  /* used if no packing */

    thread_data_t thread_data[1];
};

/* forward declarations */
static void mul_MxN_Nx64(packed_matrix_t *A, uint64 *x, uint64 *b);
static void mul_trans_MxN_Nx64(packed_matrix_t *A, uint64 *x, uint64 *b);
static void mul_Nx64_64x64_acc(uint64 *v, uint64 *x, uint64 *y, uint32 n);
static void mul_64xN_Nx64(uint64 *x, uint64 *y, uint64 *xy, uint32 n);
static void packed_matrix_init(packed_matrix_t *p, la_col_t *A,
                               uint32 nrows, uint32 ncols,
                               uint32 num_dense_rows, uint32 cache_size2);
static void packed_matrix_free(packed_matrix_t *p);
static size_t packed_matrix_sizeof(packed_matrix_t *p);
static void mul_packed_core(thread_data_t *t);
static void mul_trans_packed_core(thread_data_t *t);

/* ================================================================
   Bitmask table
   ================================================================ */

#define BIT(x) ((uint64)(1) << (x))

static const uint64 bitmask[64] = {
    BIT( 0), BIT( 1), BIT( 2), BIT( 3), BIT( 4), BIT( 5), BIT( 6), BIT( 7),
    BIT( 8), BIT( 9), BIT(10), BIT(11), BIT(12), BIT(13), BIT(14), BIT(15),
    BIT(16), BIT(17), BIT(18), BIT(19), BIT(20), BIT(21), BIT(22), BIT(23),
    BIT(24), BIT(25), BIT(26), BIT(27), BIT(28), BIT(29), BIT(30), BIT(31),
    BIT(32), BIT(33), BIT(34), BIT(35), BIT(36), BIT(37), BIT(38), BIT(39),
    BIT(40), BIT(41), BIT(42), BIT(43), BIT(44), BIT(45), BIT(46), BIT(47),
    BIT(48), BIT(49), BIT(50), BIT(51), BIT(52), BIT(53), BIT(54), BIT(55),
    BIT(56), BIT(57), BIT(58), BIT(59), BIT(60), BIT(61), BIT(62), BIT(63),
};

/* ================================================================
   lanczos_pre.c -- matrix preprocessing
   ================================================================ */

struct row_count_t {
    uint32 index;
    uint32 count;
};

static int compare_row_count(const void *x, const void *y) {
    row_count_t *xx = (row_count_t *)x;
    row_count_t *yy = (row_count_t *)y;
    return (int)yy->count - (int)xx->count;
}

static int compare_uint32(const void *x, const void *y) {
    uint32 *xx = (uint32 *)x;
    uint32 *yy = (uint32 *)y;
    if (*xx > *yy) return 1;
    if (*xx < *yy) return -1;
    return 0;
}

static int compare_weight(const void *x, const void *y) {
    la_col_t *xx = (la_col_t *)x;
    la_col_t *yy = (la_col_t *)y;
    return (int)xx->weight - (int)yy->weight;
}

/*------------------------------------------------------------------*/
static void count_matrix_nonzero(uint32 nrows, uint32 num_dense_rows,
                                 uint32 ncols, la_col_t *cols) {
    (void)nrows;
    uint32 i, j;
    uint32 total_weight = 0;

    for (i = 0; i < ncols; i++) {
        total_weight += cols[i].weight;
    }
    if (num_dense_rows > 0) {
        for (i = 0; i < ncols; i++) {
            uint32 *dense_rows = cols[i].data + cols[i].weight;
            for (j = 0; j < num_dense_rows; j++) {
                if (dense_rows[j / 32] & (1u << (j % 32)))
                    total_weight++;
            }
        }
    }
    /* silent -- no logging */
    (void)total_weight;
}

/*------------------------------------------------------------------*/
#define MAX_COL_WEIGHT 1000

static void combine_cliques(uint32 num_dense_rows,
                             uint32 *ncols_out, la_col_t *cols,
                             row_count_t *counts) {
    uint32 i, j;
    uint32 ncols = *ncols_out;
    uint32 dense_row_words = (num_dense_rows + 31) / 32;
    uint32 num_merged;
    uint32 merge_array_buf[MAX_COL_WEIGHT];

    for (i = 0; i < ncols; i++) {
        la_col_t *c = cols + i;
        for (j = 0; j < c->weight; j++)
            counts[c->data[j]].index = i;
    }

    for (i = 0; i < ncols; i++) {
        la_col_t *c0;
        la_col_t *c1 = cols + i;
        uint32 clique_base = (uint32)(-1);

        if (c1->data == NULL) continue;

        for (j = 0; j < c1->weight; j++) {
            row_count_t *curr_clique = counts + c1->data[j];
            if (curr_clique->count == 2) {
                clique_base = curr_clique->index;
                break;
            }
        }
        if (clique_base == (uint32)(-1) || clique_base == i) continue;

        c0 = cols + clique_base;
        if (c0->data == NULL || c0->weight + c1->weight >= MAX_COL_WEIGHT)
            continue;

        for (j = 0; j < c0->weight; j++)
            counts[c0->data[j]].count--;
        for (j = 0; j < c1->weight; j++)
            counts[c1->data[j]].count--;

        num_merged = merge_relations(merge_array_buf,
                                     c0->data, c0->weight,
                                     c1->data, c1->weight);
        for (j = 0; j < dense_row_words; j++) {
            merge_array_buf[num_merged + j] = c0->data[c0->weight + j] ^
                                              c1->data[c1->weight + j];
        }
        free(c0->data);
        c0->data = (uint32 *)xmalloc((num_merged + dense_row_words) * sizeof(uint32));
        memcpy(c0->data, merge_array_buf, (num_merged + dense_row_words) * sizeof(uint32));
        c0->weight = num_merged;

        c0->cycle.list = (uint32 *)xrealloc(c0->cycle.list,
                          (c0->cycle.num_relations + c1->cycle.num_relations) * sizeof(uint32));
        memcpy(c0->cycle.list + c0->cycle.num_relations,
               c1->cycle.list, c1->cycle.num_relations * sizeof(uint32));
        c0->cycle.num_relations += c1->cycle.num_relations;

        for (j = 0; j < c0->weight; j++) {
            row_count_t *curr_row = counts + c0->data[j];
            curr_row->count++;
            curr_row->index = clique_base;
        }

        free(c1->data);  c1->data = NULL;
        free(c1->cycle.list); c1->cycle.list = NULL;
    }

    for (i = j = 0; i < ncols; i++) {
        if (cols[i].data != NULL)
            cols[j++] = cols[i];
    }
    *ncols_out = j;
}

/*------------------------------------------------------------------*/
static void reduce_matrix(uint32 *nrows,
                           uint32 num_dense_rows, uint32 *ncols,
                           la_col_t *cols, uint32 num_excess) {

    uint32 r, c, i, j, k;
    uint32 passes;
    row_count_t *counts, *old_counts;
    uint32 reduced_rows;
    uint32 reduced_cols;
    uint32 prune_cliques = (*ncols >= MIN_POST_LANCZOS_DIM);

    qsort(cols, (size_t)(*ncols), sizeof(la_col_t), compare_weight);

    reduced_rows = *nrows;
    reduced_cols = *ncols;
    passes = 0;

    old_counts = (row_count_t *)xmalloc((size_t)reduced_rows * sizeof(row_count_t));
    counts     = (row_count_t *)xmalloc((size_t)reduced_rows * sizeof(row_count_t));
    for (i = 0; i < reduced_rows; i++) {
        old_counts[i].index = i;
        old_counts[i].count = 0;
    }
    for (i = 0; i < reduced_cols; i++) {
        for (j = 0; j < cols[i].weight; j++)
            old_counts[cols[i].data[j]].count++;
    }

    memcpy(counts, old_counts, reduced_rows * sizeof(row_count_t));
    qsort(counts + num_dense_rows,
          (size_t)(reduced_rows - num_dense_rows),
          sizeof(row_count_t), compare_row_count);
    for (i = j = num_dense_rows; i < reduced_rows; i++) {
        if (counts[i].count) {
            counts[j].count = counts[i].count;
            old_counts[counts[i].index].index = j;
            j++;
        }
    }
    reduced_rows = j;

    for (i = 0; i < reduced_cols; i++) {
        la_col_t *col = cols + i;
        for (j = 0; j < col->weight; j++)
            col->data[j] = old_counts[col->data[j]].index;
        qsort(col->data, (size_t)col->weight, sizeof(uint32), compare_uint32);
    }
    free(old_counts);

    do {
        r = reduced_rows;

        do {
            c = reduced_cols;

            for (i = j = 0; i < reduced_cols; i++) {
                la_col_t *col = cols + i;
                uint32 weight = col->weight;

                for (k = 0; k < weight; k++) {
                    if (counts[col->data[k]].count < 2)
                        break;
                }

                if (weight == 0 || k < weight ||
                    (prune_cliques && col->data[weight - 1] < POST_LANCZOS_ROWS)) {
                    for (k = 0; k < weight; k++)
                        counts[col->data[k]].count--;
                    free(col->data);
                    free(col->cycle.list);
                } else {
                    cols[j++] = cols[i];
                }
            }
            reduced_cols = j;

            if (prune_cliques) {
                combine_cliques(num_dense_rows, &reduced_cols, cols, counts);
            }
        } while (c != reduced_cols);

        for (i = reduced_rows = num_dense_rows; i < *nrows; i++) {
            if (counts[i].count)
                reduced_rows++;
        }

        if (reduced_cols > reduced_rows + num_excess) {
            for (i = reduced_rows + num_excess; i < reduced_cols; i++) {
                la_col_t *col = cols + i;
                for (j = 0; j < col->weight; j++)
                    counts[col->data[j]].count--;
                free(col->data);
                free(col->cycle.list);
            }
            reduced_cols = reduced_rows + num_excess;
        }

        passes++;
    } while (r != reduced_rows);

    if (reduced_cols == 0) {
        free(counts);
        *nrows = reduced_rows;
        *ncols = reduced_cols;
        return;
    }

    count_matrix_nonzero(reduced_rows, num_dense_rows, reduced_cols, cols);

    for (i = j = num_dense_rows; i < *nrows; i++) {
        if (counts[i].count)
            counts[i].index = j++;
    }
    for (i = 0; i < reduced_cols; i++) {
        la_col_t *col = cols + i;
        for (j = 0; j < col->weight; j++)
            col->data[j] = counts[col->data[j]].index;
    }

    for (i = 1, j = reduced_cols - 2; i < j; i += 2, j -= 2) {
        la_col_t tmp = cols[i];
        cols[i] = cols[j];
        cols[j] = tmp;
    }

    free(counts);
    *nrows = reduced_rows;
    *ncols = reduced_cols;
}

/* ================================================================
   lanczos_matmul1.c -- packed multiply kernels
   ================================================================ */

static void mul_one_med_block(packed_block_t *curr_block,
                              uint64 *curr_col, uint64 *curr_b) {

    uint16 *entries = curr_block->med_entries;

    while (1) {
        uint64 accum;

#if defined(GCC_ASM64X)
        uint64 i = 0;
        uint64 row = entries[0];
        uint64 count = entries[1];
#else
        uint32 i = 0;
        uint32 row = entries[0];
        uint32 count = entries[1];
#endif

        if (count == 0) break;

#if defined(GCC_ASM64X)

    #define _txor_med(k)                        \
        "movzwq %%ax, %%rdx         \n\t"       \
        "xorq (%2,%%rdx,8), %0      \n\t"       \
        "shrq $16, %%rax             \n\t"       \
        "xorq (%2,%%rax,8), %%rsi   \n\t"       \
        "movl 2*(2+4+(" #k "))(%3,%1,2), %%eax \n\t" \
        "movzwq %%cx, %%rdx         \n\t"       \
        "xorq (%2,%%rdx,8), %0      \n\t"       \
        "shrq $16, %%rcx             \n\t"       \
        "xorq (%2,%%rcx,8), %%rsi   \n\t"       \
        "movl 2*(2+6+(" #k "))(%3,%1,2), %%ecx \n\t"

    ASM_G volatile(
        "movl 2*(2+0)(%3,%1,2), %%eax   \n\t"
        "movl 2*(2+2)(%3,%1,2), %%ecx   \n\t"
        "xorq %0, %0                     \n\t"
        "xorq %%rsi, %%rsi               \n\t"
        "cmpq $0, %4                     \n\t"
        "je 1f                            \n\t"
        ALIGN_LOOP
        "0:                               \n\t"
        _txor_med(0) _txor_med(4) _txor_med(8) _txor_med(12)
        "addq $16, %1                    \n\t"
        "cmpq %4, %1                     \n\t"
        "jne 0b                           \n\t"
        "xorq %%rsi, %0                  \n\t"
        "1:                               \n\t"
        :"=&r"(accum), "+r"(i)
        :"r"(curr_col), "r"(entries),
         "g"(count & (uint64)(~15))
        :"%rax", "%rcx", "%rdx", "%rsi", "memory", "cc");

    #undef _txor_med

#else
        accum = 0;
        for (i = 0; i < (count & (uint32)(~15)); i += 16) {
            accum ^= curr_col[entries[i+2+0]]
                   ^ curr_col[entries[i+2+1]]
                   ^ curr_col[entries[i+2+2]]
                   ^ curr_col[entries[i+2+3]]
                   ^ curr_col[entries[i+2+4]]
                   ^ curr_col[entries[i+2+5]]
                   ^ curr_col[entries[i+2+6]]
                   ^ curr_col[entries[i+2+7]]
                   ^ curr_col[entries[i+2+8]]
                   ^ curr_col[entries[i+2+9]]
                   ^ curr_col[entries[i+2+10]]
                   ^ curr_col[entries[i+2+11]]
                   ^ curr_col[entries[i+2+12]]
                   ^ curr_col[entries[i+2+13]]
                   ^ curr_col[entries[i+2+14]]
                   ^ curr_col[entries[i+2+15]];
        }
#endif
        for (; i < count; i++)
            accum ^= curr_col[entries[i+2]];
        curr_b[row] ^= accum;
        entries += count + 2;
    }
}

/*------------------------------------------------------------------*/
static void mul_one_block(packed_block_t *curr_block,
                           uint64 *curr_col, uint64 *curr_b) {

    uint32 i = 0;
    uint32 j = 0;
    uint32 k;
    uint32 num_entries = curr_block->num_entries;
    entry_idx_t *entries = curr_block->entries;

    #define _txor(x) curr_b[entries[i+x].row_off] ^= \
                     curr_col[entries[i+x].col_off]

    for (i = 0; i < (num_entries & (uint32)(~15)); i += 16) {
        #ifdef MANUAL_PREFETCH
        PREFETCH(entries + i + 48);
        #endif
        _txor( 0); _txor( 1); _txor( 2); _txor( 3);
        _txor( 4); _txor( 5); _txor( 6); _txor( 7);
        _txor( 8); _txor( 9); _txor(10); _txor(11);
        _txor(12); _txor(13); _txor(14); _txor(15);
    }
    #undef _txor

    for (; i < num_entries; i++) {
        j = entries[i].row_off;
        k = entries[i].col_off;
        curr_b[j] ^= curr_col[k];
    }
}

/*------------------------------------------------------------------*/
static void mul_packed_core(thread_data_t *t) {

    uint64 *x = t->x;
    uint64 *b = t->b;
    uint32 i;

    for (i = 0; i < t->num_blocks; i++) {
        packed_block_t *curr_block = t->blocks + i;
        if (curr_block->med_entries)
            mul_one_med_block(curr_block,
                              x + curr_block->start_col,
                              b + curr_block->start_row);
        else
            mul_one_block(curr_block,
                          x + curr_block->start_col,
                          b + curr_block->start_row);
    }

    for (i = 0; i < (t->num_dense_rows + 63) / 64; i++) {
        mul_64xN_Nx64(t->dense_blocks[i],
                      x + t->blocks[0].start_col,
                      b + 64 * i, t->ncols);
    }
}

/* ================================================================
   lanczos_matmul2.c -- packed transpose-multiply kernels
   ================================================================ */

static void mul_trans_one_med_block(packed_block_t *curr_block,
                                    uint64 *curr_row, uint64 *curr_b) {

    uint16 *entries = curr_block->med_entries;

    while (1) {
        uint64 t;
#if defined(GCC_ASM64X)
        uint64 i = 0;
        uint64 row = entries[0];
        uint64 count = entries[1];
#else
        uint32 i = 0;
        uint32 row = entries[0];
        uint32 count = entries[1];
#endif

        if (count == 0) break;

        t = curr_row[row];

#if defined(GCC_ASM64X)

    #define _txor_tmed(k)                       \
        "movzwq %%r8w, %%r9          \n\t"      \
        "xorq %4, (%1,%%r9,8)        \n\t"      \
        "shrq $16, %%r8              \n\t"      \
        "xorq %4, (%1,%%r8,8)        \n\t"      \
        "movl 2*(2+8+" #k ")(%2,%0,2), %%r8d  \n\t" \
        "movzwq %%r10w, %%r11        \n\t"      \
        "xorq %4, (%1,%%r11,8)       \n\t"      \
        "shrq $16, %%r10             \n\t"      \
        "xorq %4, (%1,%%r10,8)       \n\t"      \
        "movl 2*(2+10+" #k ")(%2,%0,2), %%r10d \n\t" \
        "movzwq %%r12w, %%r13        \n\t"      \
        "xorq %4, (%1,%%r13,8)       \n\t"      \
        "shrq $16, %%r12             \n\t"      \
        "xorq %4, (%1,%%r12,8)       \n\t"      \
        "movl 2*(2+12+" #k ")(%2,%0,2), %%r12d \n\t" \
        "movzwq %%r14w, %%r15        \n\t"      \
        "xorq %4, (%1,%%r15,8)       \n\t"      \
        "shrq $16, %%r14             \n\t"      \
        "xorq %4, (%1,%%r14,8)       \n\t"      \
        "movl 2*(2+14+" #k ")(%2,%0,2), %%r14d \n\t"

    ASM_G volatile(
        "movl 2*(2+0)(%2,%0,2), %%r8d    \n\t"
        "movl 2*(2+2)(%2,%0,2), %%r10d   \n\t"
        "movl 2*(2+4)(%2,%0,2), %%r12d   \n\t"
        "movl 2*(2+6)(%2,%0,2), %%r14d   \n\t"
        "cmpq $0, %3                      \n\t"
        "je 1f                             \n\t"
        ALIGN_LOOP
        "0:                                \n\t"
        _txor_tmed(0) _txor_tmed(8)
        "addq $16, %0                     \n\t"
        "cmpq %3, %0                      \n\t"
        "jne 0b                            \n\t"
        "1:                                \n\t"
        :"+r"(i)
        :"r"(curr_b), "r"(entries),
         "g"(count & (uint64)(~15)), "r"(t)
        :"%r8", "%r9", "%r10", "%r11",
         "%r12", "%r13", "%r14", "%r15", "memory", "cc");

    #undef _txor_tmed

#else
        for (i = 0; i < (count & (uint32)(~15)); i += 16) {
            curr_b[entries[i+2+ 0]] ^= t;
            curr_b[entries[i+2+ 1]] ^= t;
            curr_b[entries[i+2+ 2]] ^= t;
            curr_b[entries[i+2+ 3]] ^= t;
            curr_b[entries[i+2+ 4]] ^= t;
            curr_b[entries[i+2+ 5]] ^= t;
            curr_b[entries[i+2+ 6]] ^= t;
            curr_b[entries[i+2+ 7]] ^= t;
            curr_b[entries[i+2+ 8]] ^= t;
            curr_b[entries[i+2+ 9]] ^= t;
            curr_b[entries[i+2+10]] ^= t;
            curr_b[entries[i+2+11]] ^= t;
            curr_b[entries[i+2+12]] ^= t;
            curr_b[entries[i+2+13]] ^= t;
            curr_b[entries[i+2+14]] ^= t;
            curr_b[entries[i+2+15]] ^= t;
        }
#endif
        for (; i < count; i++)
            curr_b[entries[i+2]] ^= t;
        entries += count + 2;
    }
}

/*------------------------------------------------------------------*/
static void mul_trans_one_block(packed_block_t *curr_block,
                                uint64 *curr_row, uint64 *curr_b) {

    uint32 i = 0;
    uint32 j = 0;
    uint32 k;
    uint32 num_entries = curr_block->num_entries;
    entry_idx_t *entries = curr_block->entries;

    #define _txor(x) curr_b[entries[i+x].col_off] ^= \
                     curr_row[entries[i+x].row_off]

    for (i = 0; i < (num_entries & (uint32)(~15)); i += 16) {
        #ifdef MANUAL_PREFETCH
        PREFETCH(entries + i + 48);
        #endif
        _txor( 0); _txor( 1); _txor( 2); _txor( 3);
        _txor( 4); _txor( 5); _txor( 6); _txor( 7);
        _txor( 8); _txor( 9); _txor(10); _txor(11);
        _txor(12); _txor(13); _txor(14); _txor(15);
    }
    #undef _txor

    for (; i < num_entries; i++) {
        j = entries[i].row_off;
        k = entries[i].col_off;
        curr_b[k] ^= curr_row[j];
    }
}

/*------------------------------------------------------------------*/
static void mul_trans_packed_core(thread_data_t *t) {

    uint64 *x = t->x;
    uint64 *b = t->b;
    uint32 i;

    for (i = 0; i < t->num_blocks; i++) {
        packed_block_t *curr_block = t->blocks + i;
        if (curr_block->med_entries)
            mul_trans_one_med_block(curr_block,
                                   x + curr_block->start_row,
                                   b + curr_block->start_col);
        else
            mul_trans_one_block(curr_block,
                                x + curr_block->start_row,
                                b + curr_block->start_col);
    }

    for (i = 0; i < (t->num_dense_rows + 63) / 64; i++) {
        mul_Nx64_64x64_acc(t->dense_blocks[i], x + 64 * i,
                           b + t->blocks[0].start_col, t->ncols);
    }
}

/* ================================================================
   lanczos_matmul0.c -- unpacked + packed dispatch, packing init
   ================================================================ */

static void mul_unpacked(packed_matrix_t *matrix,
                          uint64 *x, uint64 *b) {

    uint32 ncols = matrix->ncols;
    uint32 num_dense_rows = matrix->num_dense_rows;
    la_col_t *A = matrix->unpacked_cols;
    uint32 i, j;

    memset(b, 0, ncols * sizeof(uint64));

    for (i = 0; i < ncols; i++) {
        la_col_t *col = A + i;
        uint32 *row_entries = col->data;
        uint64 tmp = x[i];
        for (j = 0; j < col->weight; j++)
            b[row_entries[j]] ^= tmp;
    }

    if (num_dense_rows) {
        for (i = 0; i < ncols; i++) {
            la_col_t *col = A + i;
            uint32 *row_entries = col->data + col->weight;
            uint64 tmp = x[i];
            for (j = 0; j < num_dense_rows; j++) {
                if (row_entries[j / 32] & ((uint32)1 << (j % 32)))
                    b[j] ^= tmp;
            }
        }
    }
}

/*------------------------------------------------------------------*/
static void mul_trans_unpacked(packed_matrix_t *matrix,
                                uint64 *x, uint64 *b) {

    uint32 ncols = matrix->ncols;
    uint32 num_dense_rows = matrix->num_dense_rows;
    la_col_t *A = matrix->unpacked_cols;
    uint32 i, j;

    for (i = 0; i < ncols; i++) {
        la_col_t *col = A + i;
        uint32 *row_entries = col->data;
        uint64 accum = 0;
        for (j = 0; j < col->weight; j++)
            accum ^= x[row_entries[j]];
        b[i] = accum;
    }

    if (num_dense_rows) {
        for (i = 0; i < ncols; i++) {
            la_col_t *col = A + i;
            uint32 *row_entries = col->data + col->weight;
            uint64 accum = b[i];
            for (j = 0; j < num_dense_rows; j++) {
                if (row_entries[j / 32] & ((uint32)1 << (j % 32)))
                    accum ^= x[j];
            }
            b[i] = accum;
        }
    }
}

/*------------------------------------------------------------------*/
static void mul_packed(packed_matrix_t *matrix, uint64 *x, uint64 *b) {

    /* single-threaded path */
    thread_data_t *t = matrix->thread_data;
    t->x = x;
    t->b = b;
    memset(t->b, 0, matrix->ncols * sizeof(uint64));
    mul_packed_core(t);
}

/*------------------------------------------------------------------*/
static void mul_trans_packed(packed_matrix_t *matrix, uint64 *x, uint64 *b) {

    thread_data_t *t = matrix->thread_data;
    t->x = x;
    t->b = b;
    memset(b, 0, matrix->ncols * sizeof(uint64));
    mul_trans_packed_core(t);
}

/*------------------------------------------------------------------*/
static int compare_row_off(const void *x, const void *y) {
    entry_idx_t *xx = (entry_idx_t *)x;
    entry_idx_t *yy = (entry_idx_t *)y;
    if (xx->row_off > yy->row_off) return 1;
    if (xx->row_off < yy->row_off) return -1;
    return (int)xx->col_off - (int)yy->col_off;
}

static void matrix_thread_init(thread_data_t *t) {

    uint32 i, j, k, m;
    uint32 num_row_blocks;
    uint32 num_col_blocks;
    uint32 dense_row_blocks;
    packed_block_t *curr_stripe;
    entry_idx_t *e;

    la_col_t *A = t->initial_cols;
    uint32 nrows = t->nrows_in;
    uint32 col_min = t->col_min;
    uint32 col_max = t->col_max;
    uint32 block_size = t->block_size;
    uint32 num_dense_rows = t->num_dense_rows;

    /* pack the dense rows 64 at a time */
    t->ncols = col_max - col_min + 1;
    dense_row_blocks = (num_dense_rows + 63) / 64;
    if (dense_row_blocks) {
        t->dense_blocks = (uint64 **)xmalloc(dense_row_blocks * sizeof(uint64 *));
        for (i = 0; i < dense_row_blocks; i++)
            t->dense_blocks[i] = (uint64 *)xmalloc(t->ncols * sizeof(uint64));

        for (i = 0; i < t->ncols; i++) {
            la_col_t *c = A + col_min + i;
            uint32 *src = c->data + c->weight;
            for (j = 0; j < dense_row_blocks; j++) {
                t->dense_blocks[j][i] =
                    (uint64)src[2 * j + 1] << 32 |
                    (uint64)src[2 * j];
            }
        }
    }

    num_row_blocks = (nrows - NUM_MEDIUM_ROWS + (block_size - 1)) / block_size + 1;
    num_col_blocks = ((col_max - col_min + 1) + (block_size - 1)) / block_size;
    t->num_blocks = num_row_blocks * num_col_blocks;
    t->blocks = curr_stripe = (packed_block_t *)xcalloc(
                                (size_t)t->num_blocks, sizeof(packed_block_t));

    for (i = 0; i < num_col_blocks; i++, curr_stripe++) {

        uint32 curr_cols = MIN(block_size, (col_max - col_min + 1) - i * block_size);
        packed_block_t *b;

        for (j = 0, b = curr_stripe; j < num_row_blocks; j++) {
            if (j == 0) {
                b->start_row = 0;
                b->num_rows = NUM_MEDIUM_ROWS;
            } else {
                b->start_row = NUM_MEDIUM_ROWS + (j - 1) * block_size;
                b->num_rows = block_size;
            }
            b->start_col = col_min + i * block_size;
            b += num_col_blocks;
        }

        for (j = 0; j < curr_cols; j++) {
            la_col_t *c = A + col_min + i * block_size + j;
            for (k = 0, b = curr_stripe; k < c->weight; k++) {
                uint32 index = c->data[k];
                while (index >= b->start_row + b->num_rows)
                    b += num_col_blocks;
                b->num_entries_alloc++;
            }
        }

        for (j = 0, b = curr_stripe; j < num_row_blocks; j++) {
            b->entries = (entry_idx_t *)xmalloc(
                          b->num_entries_alloc * sizeof(entry_idx_t));
            b += num_col_blocks;
        }

        for (j = 0; j < curr_cols; j++) {
            la_col_t *c = A + col_min + i * block_size + j;
            for (k = 0, b = curr_stripe; k < c->weight; k++) {
                uint32 index = c->data[k];
                while (index >= b->start_row + b->num_rows)
                    b += num_col_blocks;
                e = b->entries + b->num_entries++;
                e->row_off = (uint16)(index - b->start_row);
                e->col_off = (uint16)j;
            }
            free(c->data);
            c->data = NULL;
        }

        /* convert first block in stripe to med_entries format */
        b = curr_stripe;
        e = b->entries;
        qsort(e, (size_t)b->num_entries, sizeof(entry_idx_t), compare_row_off);
        for (j = k = 0; j < b->num_entries; j++) {
            if (j == 0 || e[j].row_off != e[j-1].row_off)
                k++;
        }

        b->med_entries = (uint16 *)xmalloc((b->num_entries + 2 * k + 8) * sizeof(uint16));
        j = k = 0;
        while (j < b->num_entries) {
            for (m = 0; j + m < b->num_entries; m++) {
                if (m > 0 && e[j+m].row_off != e[j+m-1].row_off)
                    break;
                b->med_entries[k+m+2] = e[j+m].col_off;
            }
            b->med_entries[k] = e[j].row_off;
            b->med_entries[k+1] = (uint16)m;
            j += m;
            k += m + 2;
        }
        b->med_entries[k] = b->med_entries[k+1] = 0;
        free(b->entries);
        b->entries = NULL;
    }
}

/*------------------------------------------------------------------*/
static void matrix_thread_free(thread_data_t *t) {
    uint32 i;

    for (i = 0; i < (t->num_dense_rows + 63) / 64; i++)
        free(t->dense_blocks[i]);
    free(t->dense_blocks);

    for (i = 0; i < t->num_blocks; i++) {
        free(t->blocks[i].entries);
        free(t->blocks[i].med_entries);
    }
    free(t->blocks);
}

/*------------------------------------------------------------------*/
static void packed_matrix_init(packed_matrix_t *p, la_col_t *A,
                               uint32 nrows, uint32 ncols,
                               uint32 num_dense_rows,
                               uint32 cache_size2) {

    uint32 block_size;

    memset(p, 0, sizeof(packed_matrix_t));
    p->unpacked_cols = A;
    p->nrows = nrows;
    p->ncols = ncols;
    p->num_dense_rows = num_dense_rows;

    if (ncols <= MIN_NCOLS_TO_PACK)
        return;

    p->unpacked_cols = NULL;

    /* decide on block size: split L2 cache into thirds */
    block_size = cache_size2 / (3 * sizeof(uint64));
    block_size = MIN(block_size, (uint32)(ncols / 2.5));
    block_size = MIN(block_size, 65536u);
    if (block_size == 0)
        block_size = 32768;

    /* single-thread */
    p->num_threads = 1;

    {
        thread_data_t *t = p->thread_data;
        t->my_oid = 0;
        t->initial_cols = A;
        t->col_min = 0;
        t->col_max = ncols - 1;
        t->nrows_in = nrows;
        t->ncols_in = ncols;
        t->block_size = block_size;
        t->num_dense_rows = num_dense_rows;
        matrix_thread_init(t);
    }
}

/*------------------------------------------------------------------*/
static void packed_matrix_free(packed_matrix_t *p) {
    uint32 i;

    if (p->unpacked_cols) {
        la_col_t *A = p->unpacked_cols;
        for (i = 0; i < p->ncols; i++) {
            free(A[i].data);
            A[i].data = NULL;
        }
    } else {
        matrix_thread_free(p->thread_data);
    }
}

/*------------------------------------------------------------------*/
__attribute__((unused))
static size_t packed_matrix_sizeof(packed_matrix_t *p) {
    uint32 i, j;
    size_t mem_use = 0;

    if (p->unpacked_cols) {
        la_col_t *A = p->unpacked_cols;
        mem_use = p->ncols * sizeof(la_col_t);
        for (i = 0; i < p->ncols; i++)
            mem_use += A[i].weight * sizeof(uint32);
    } else {
        for (i = 0; i < p->num_threads; i++) {
            thread_data_t *t = p->thread_data + i;
            mem_use += p->ncols * sizeof(uint64) +
                       t->num_blocks * sizeof(packed_block_t) +
                       t->ncols * sizeof(uint64) *
                           ((t->num_dense_rows + 63) / 64);
            for (j = 0; j < t->num_blocks; j++) {
                packed_block_t *b = t->blocks + j;
                if (b->entries)
                    mem_use += b->num_entries * sizeof(uint32);
                else
                    mem_use += (b->num_entries + 2 * NUM_MEDIUM_ROWS) * sizeof(uint16);
            }
        }
    }
    return mem_use;
}

/*------------------------------------------------------------------*/
static void mul_MxN_Nx64(packed_matrix_t *A, uint64 *x, uint64 *b) {
    if (A->unpacked_cols)
        mul_unpacked(A, x, b);
    else
        mul_packed(A, x, b);
}

/*------------------------------------------------------------------*/
static void mul_trans_MxN_Nx64(packed_matrix_t *A, uint64 *x, uint64 *b) {
    if (A->unpacked_cols)
        mul_trans_unpacked(A, x, b);
    else
        mul_trans_packed(A, x, b);
}

/* ================================================================
   lanczos.c -- core Block Lanczos iteration
   ================================================================ */

/*------------------------------------------------------------------*/
static void mul_64x64_64x64(uint64 *a, uint64 *b, uint64 *c) {

    uint64 ai, bj, accum;
    uint64 tmp[64];
    uint32 i, j;

    for (i = 0; i < 64; i++) {
        j = 0;
        accum = 0;
        ai = a[i];
        while (ai) {
            bj = b[j];
            if (ai & 1) accum ^= bj;
            ai >>= 1;
            j++;
        }
        tmp[i] = accum;
    }
    memcpy(c, tmp, sizeof(tmp));
}

/*------------------------------------------------------------------*/
static void transpose_64x64(uint64 *a, uint64 *b) {

    uint32 i, j;
    uint64 tmp[64] = {0};

    for (i = 0; i < 64; i++) {
        uint64 word = a[i];
        uint64 mask = bitmask[i];
        for (j = 0; j < 64; j++) {
            if (word & bitmask[j])
                tmp[j] |= mask;
        }
    }
    memcpy(b, tmp, sizeof(tmp));
}

/*------------------------------------------------------------------*/
static void mul_Nx64_64x64_acc(uint64 *v, uint64 *x,
                                uint64 *y, uint32 n) {

    uint32 i, j, k;
    uint64 c[8 * 256];

    for (i = 0; i < 8; i++) {
        uint64 *xtmp = x + 8 * i;
        uint64 *ctmp = c + 256 * i;
        for (j = 0; j < 256; j++) {
            uint64 accum = 0;
            uint32 index = j;
            for (k = 0; k < 8; k++) {
                if (index & ((uint32)1 << k))
                    accum ^= xtmp[k];
            }
            ctmp[j] = accum;
        }
    }

    for (i = 0; i < n; i++) {
        uint64 word = v[i];
        y[i] ^=  c[ 0*256 + ((uint8)(word >>  0)) ]
               ^ c[ 1*256 + ((uint8)(word >>  8)) ]
               ^ c[ 2*256 + ((uint8)(word >> 16)) ]
               ^ c[ 3*256 + ((uint8)(word >> 24)) ]
               ^ c[ 4*256 + ((uint8)(word >> 32)) ]
               ^ c[ 5*256 + ((uint8)(word >> 40)) ]
               ^ c[ 6*256 + ((uint8)(word >> 48)) ]
               ^ c[ 7*256 + ((uint8)(word >> 56)) ];
    }
}

/*------------------------------------------------------------------*/
static void mul_64xN_Nx64(uint64 *x, uint64 *y,
                           uint64 *xy, uint32 n) {

    uint32 i;
    uint64 c[8 * 256] = {0};

    memset(xy, 0, 64 * sizeof(uint64));

    for (i = 0; i < n; i++) {
        uint64 xi = x[i];
        uint64 yi = y[i];
        c[ 0*256 + ((uint8) xi       ) ] ^= yi;
        c[ 1*256 + ((uint8)(xi >>  8)) ] ^= yi;
        c[ 2*256 + ((uint8)(xi >> 16)) ] ^= yi;
        c[ 3*256 + ((uint8)(xi >> 24)) ] ^= yi;
        c[ 4*256 + ((uint8)(xi >> 32)) ] ^= yi;
        c[ 5*256 + ((uint8)(xi >> 40)) ] ^= yi;
        c[ 6*256 + ((uint8)(xi >> 48)) ] ^= yi;
        c[ 7*256 + ((uint8)(xi >> 56)) ] ^= yi;
    }

    for (i = 0; i < 8; i++) {
        uint32 j;
        uint64 a0, a1, a2, a3, a4, a5, a6, a7;
        a0 = a1 = a2 = a3 = 0;
        a4 = a5 = a6 = a7 = 0;
        for (j = 0; j < 256; j++) {
            if ((j >> i) & 1) {
                a0 ^= c[0*256 + j];
                a1 ^= c[1*256 + j];
                a2 ^= c[2*256 + j];
                a3 ^= c[3*256 + j];
                a4 ^= c[4*256 + j];
                a5 ^= c[5*256 + j];
                a6 ^= c[6*256 + j];
                a7 ^= c[7*256 + j];
            }
        }
        xy[ 0] = a0; xy[ 8] = a1; xy[16] = a2; xy[24] = a3;
        xy[32] = a4; xy[40] = a5; xy[48] = a6; xy[56] = a7;
        xy++;
    }
}

/*------------------------------------------------------------------*/
static uint32 find_nonsingular_sub(uint64 *t, uint32 *s,
                                    uint32 *last_s, uint32 last_dim,
                                    uint64 *w) {

    uint32 i, j;
    uint32 dim;
    uint32 cols[64];
    uint64 M[64][2];
    uint64 mask, *row_i, *row_j;
    uint64 m0, m1;

    for (i = 0; i < 64; i++) {
        M[i][0] = t[i];
        M[i][1] = bitmask[i];
    }

    mask = 0;
    for (i = 0; i < last_dim; i++) {
        cols[63 - i] = last_s[i];
        mask |= bitmask[last_s[i]];
    }
    for (i = j = 0; i < 64; i++) {
        if (!(mask & bitmask[i]))
            cols[j++] = i;
    }

    for (i = dim = 0; i < 64; i++) {
        mask = bitmask[cols[i]];
        row_i = M[cols[i]];

        for (j = i; j < 64; j++) {
            row_j = M[cols[j]];
            if (row_j[0] & mask) {
                m0 = row_j[0]; m1 = row_j[1];
                row_j[0] = row_i[0]; row_j[1] = row_i[1];
                row_i[0] = m0; row_i[1] = m1;
                break;
            }
        }

        if (j < 64) {
            for (j = 0; j < 64; j++) {
                row_j = M[cols[j]];
                if ((row_i != row_j) && (row_j[0] & mask)) {
                    row_j[0] ^= row_i[0];
                    row_j[1] ^= row_i[1];
                }
            }
            s[dim++] = cols[i];
            continue;
        }

        for (j = i; j < 64; j++) {
            row_j = M[cols[j]];
            if (row_j[1] & mask) {
                m0 = row_j[0]; m1 = row_j[1];
                row_j[0] = row_i[0]; row_j[1] = row_i[1];
                row_i[0] = m0; row_i[1] = m1;
                break;
            }
        }

        if (j == 64) {
            return 0;
        }

        for (j = 0; j < 64; j++) {
            row_j = M[cols[j]];
            if ((row_i != row_j) && (row_j[1] & mask)) {
                row_j[0] ^= row_i[0];
                row_j[1] ^= row_i[1];
            }
        }

        row_i[0] = row_i[1] = 0;
    }

    for (i = 0; i < 64; i++)
        w[i] = M[i][1];

    return dim;
}

/*------------------------------------------------------------------*/
static void transpose_vector(uint32 ncols, uint64 *v, uint64 **trans) {

    uint32 i, j;
    uint32 col;
    uint64 mask, word;

    for (i = 0; i < ncols; i++) {
        col = i / 64;
        mask = bitmask[i % 64];
        word = v[i];
        j = 0;
        while (word) {
            if (word & 1)
                trans[j][col] |= mask;
            word >>= 1;
            j++;
        }
    }
}

/*------------------------------------------------------------------*/
static uint32 combine_cols(uint32 ncols,
                            uint64 *x, uint64 *v,
                            uint64 *ax, uint64 *av) {

    uint32 i, j, k, bitpos, col, col_words;
    uint64 mask;
    uint64 *matrix[128], *amatrix[128], *tmp;

    col_words = (ncols + 63) / 64;

    for (i = 0; i < 128; i++) {
        matrix[i]  = (uint64 *)xcalloc((size_t)col_words, sizeof(uint64));
        amatrix[i] = (uint64 *)xcalloc((size_t)col_words, sizeof(uint64));
    }

    transpose_vector(ncols, x,  matrix);
    transpose_vector(ncols, ax, amatrix);
    transpose_vector(ncols, v,  matrix + 64);
    transpose_vector(ncols, av, amatrix + 64);

    for (i = bitpos = 0; i < 128 && bitpos < ncols; bitpos++) {
        mask = bitmask[bitpos % 64];
        col = bitpos / 64;
        for (j = i; j < 128; j++) {
            if (amatrix[j][col] & mask) {
                tmp = matrix[i];  matrix[i]  = matrix[j];  matrix[j]  = tmp;
                tmp = amatrix[i]; amatrix[i] = amatrix[j]; amatrix[j] = tmp;
                break;
            }
        }
        if (j == 128) continue;

        for (j++; j < 128; j++) {
            if (amatrix[j][col] & mask) {
                for (k = 0; k < col_words; k++) {
                    amatrix[j][k] ^= amatrix[i][k];
                    matrix[j][k]  ^= matrix[i][k];
                }
            }
        }
        i++;
    }

    for (j = 0; j < ncols; j++) {
        uint64 word = 0;
        col = j / 64;
        mask = bitmask[j % 64];
        for (k = i; k < 64; k++) {
            if (matrix[k][col] & mask)
                word |= bitmask[k - i];
        }
        x[j] = word;
    }

    for (j = 0; j < 128; j++) {
        free(matrix[j]);
        free(amatrix[j]);
    }

    if (i > 64) return 0;
    return 64 - i;
}

/*------------------------------------------------------------------*/
static uint64 *form_post_lanczos_matrix(uint32 *nrows,
                                         uint32 *dense_rows_out,
                                         uint32 ncols,
                                         la_col_t *cols) {

    uint32 i, j, k;
    uint32 num_dense_rows = *dense_rows_out;
    uint32 dense_row_words;
    uint32 new_dense_rows;
    uint32 new_dense_row_words;
    uint32 final_dense_row_words;
    uint64 mask;
    uint64 *submatrix;
    mp_t tmp;

    submatrix = NULL;
    if (ncols >= MIN_NCOLS_TO_PACK ||
        (POST_LANCZOS_ROWS > 0 && ncols >= MIN_POST_LANCZOS_DIM)) {
        if (POST_LANCZOS_ROWS > 0)
            submatrix = (uint64 *)xmalloc(ncols * sizeof(uint64));
    } else {
        return NULL;
    }

    mask = (uint64)(-1) >> (64 - POST_LANCZOS_ROWS);
    dense_row_words = (num_dense_rows + 31) / 32;
    mp_clear(&tmp);

    new_dense_rows = MAX(num_dense_rows, (uint32)POST_LANCZOS_ROWS);
    new_dense_rows += 64 - (new_dense_rows - POST_LANCZOS_ROWS) % 64;
    new_dense_row_words = (new_dense_rows + 31) / 32;
    final_dense_row_words = (new_dense_rows - POST_LANCZOS_ROWS) / 32;

    for (i = 0; i < ncols; i++) {
        uint32 curr_weight = cols[i].weight;
        uint32 *curr_row = cols[i].data;

        for (j = 0; j < dense_row_words; j++)
            tmp.val[j] = curr_row[curr_weight + j];
        for (; j < new_dense_row_words; j++)
            tmp.val[j] = 0;

        for (j = k = 0; j < curr_weight; j++) {
            uint32 curr_index = curr_row[j];
            if (curr_index < new_dense_rows)
                tmp.val[curr_index / 32] |= (uint32)bitmask[curr_index % 32];
            else
                curr_row[k++] = curr_index - POST_LANCZOS_ROWS;
        }

        tmp.nwords = new_dense_row_words;
#if POST_LANCZOS_ROWS > 0
        submatrix[i] = ((uint64)tmp.val[0] |
                        (uint64)tmp.val[1] << 32) & mask;
#endif

        cols[i].weight = k;
        if (k + final_dense_row_words > 0) {
            cols[i].data = (uint32 *)xrealloc(curr_row,
                           (k + final_dense_row_words) * sizeof(uint32));
            mp_rshift(&tmp, POST_LANCZOS_ROWS, &tmp);
            memcpy(cols[i].data + k, tmp.val,
                   final_dense_row_words * sizeof(uint32));
        } else {
            free(cols[i].data);
            cols[i].data = NULL;
        }
    }

    *nrows -= POST_LANCZOS_ROWS;
    *dense_rows_out = new_dense_rows - POST_LANCZOS_ROWS;
    count_matrix_nonzero(*nrows, *dense_rows_out, ncols, cols);
    return submatrix;
}

/*------------------------------------------------------------------*/
static void init_lanczos_state(packed_matrix_t *packed_matrix,
                               uint64 *x, uint64 *v0,
                               uint64 **vt_v0, uint64 **v,
                               uint64 **vt_a_v, uint64 **vt_a2_v,
                               uint64 **winv,
                               uint32 n, uint32 s[2][64],
                               uint32 *dim1,
                               uint32 *seed1, uint32 *seed2) {

    uint32 i;

    for (i = 0; i < n; i++) {
        x[i] = v[0][i] =
            (uint64)(get_rand(seed1, seed2)) << 32 |
            (uint64)(get_rand(seed1, seed2));
    }

    mul_MxN_Nx64(packed_matrix, v[0], v[1]);
    mul_trans_MxN_Nx64(packed_matrix, v[1], v[0]);
    memcpy(v0, v[0], n * sizeof(uint64));

    memset(v[1], 0, n * sizeof(uint64));
    memset(v[2], 0, n * sizeof(uint64));
    for (i = 0; i < 64; i++) {
        s[1][i] = i;
        vt_a_v[1][i] = 0;
        vt_a2_v[1][i] = 0;
        winv[1][i] = 0;
        winv[2][i] = 0;
        vt_v0[0][i] = 0;
        vt_v0[1][i] = 0;
        vt_v0[2][i] = 0;
    }
    *dim1 = 64;
}

/*------------------------------------------------------------------*/
static uint64 *block_lanczos_core(packed_matrix_t *packed_matrix,
                                   uint32 *num_deps_found,
                                   uint64 *post_lanczos_matrix,
                                   uint32 *seed1, uint32 *seed2) {

    uint32 n = packed_matrix->ncols;
    uint64 *vnext, *v[3], *x, *v0;
    uint64 *winv[3], *vt_v0_next;
    uint64 *vt_a_v[2], *vt_a2_v[2], *vt_v0[3];
    uint64 *scratch;
    uint64 *tmp;
    uint32 s[2][64];
    uint64 d[64], e[64], f[64], f2[64];
    uint32 i, iter;
    uint32 dim0, dim1;
    uint64 mask0, mask1;

    uint32 dim_solved = 0;

    /* allocate 64x64 variables */
    winv[0]     = (uint64 *)xmalloc(64 * sizeof(uint64));
    winv[1]     = (uint64 *)xmalloc(64 * sizeof(uint64));
    winv[2]     = (uint64 *)xmalloc(64 * sizeof(uint64));
    vt_a_v[0]   = (uint64 *)xmalloc(64 * sizeof(uint64));
    vt_a_v[1]   = (uint64 *)xmalloc(64 * sizeof(uint64));
    vt_a2_v[0]  = (uint64 *)xmalloc(64 * sizeof(uint64));
    vt_a2_v[1]  = (uint64 *)xmalloc(64 * sizeof(uint64));
    vt_v0[0]    = (uint64 *)xmalloc(64 * sizeof(uint64));
    vt_v0[1]    = (uint64 *)xmalloc(64 * sizeof(uint64));
    vt_v0[2]    = (uint64 *)xmalloc(64 * sizeof(uint64));
    vt_v0_next  = (uint64 *)xmalloc(64 * sizeof(uint64));

    /* allocate size-n variables */
    v[0]    = (uint64 *)xmalloc(n * sizeof(uint64));
    v[1]    = (uint64 *)xmalloc(n * sizeof(uint64));
    v[2]    = (uint64 *)xmalloc(n * sizeof(uint64));
    vnext   = (uint64 *)xmalloc(n * sizeof(uint64));
    x       = (uint64 *)xmalloc(n * sizeof(uint64));
    scratch = (uint64 *)xmalloc(n * sizeof(uint64));
    v0      = (uint64 *)xmalloc(n * sizeof(uint64));

    /* initialize */
    iter = 0;
    dim0 = 0;

    init_lanczos_state(packed_matrix, x, v0, vt_v0, v,
                       vt_a_v, vt_a2_v, winv, n, s, &dim1,
                       seed1, seed2);

    mask1 = 0;
    for (i = 0; i < dim1; i++)
        mask1 |= bitmask[s[1][i]];

    /* perform the iteration */
    while (1) {
        iter++;

        /* multiply v[0] by B'B */
        mul_MxN_Nx64(packed_matrix, v[0], scratch);
        mul_trans_MxN_Nx64(packed_matrix, scratch, vnext);

        /* compute v0'*A*v0 and (A*v0)'(A*v0) */
        mul_64xN_Nx64(v[0], vnext, vt_a_v[0], n);
        mul_64xN_Nx64(vnext, vnext, vt_a2_v[0], n);

        /* check for termination */
        for (i = 0; i < 64; i++) {
            if (vt_a_v[0][i] != 0) break;
        }
        if (i == 64) break;

        dim0 = find_nonsingular_sub(vt_a_v[0], s[0],
                                    s[1], dim1, winv[0]);
        if (dim0 == 0) break;

        mask0 = 0;
        for (i = 0; i < dim0; i++)
            mask0 |= bitmask[s[0][i]];

        if (dim_solved < packed_matrix->nrows - 64) {
            if ((mask0 | mask1) != (uint64)(-1)) {
                dim0 = 0;
                break;
            }
        }

        dim_solved += dim0;
        if (mask0 != (uint64)(-1)) {
            for (i = 0; i < n; i++)
                vnext[i] &= mask0;
        }

        if (iter < 4) {
            mul_64xN_Nx64(v[0], v0, vt_v0[0], n);
        } else if (iter == 4) {
            memset(v0, 0, n * sizeof(uint64));
        }

        /* compute d, fold into vnext */
        for (i = 0; i < 64; i++)
            d[i] = (vt_a2_v[0][i] & mask0) ^ vt_a_v[0][i];
        mul_64x64_64x64(winv[0], d, d);
        for (i = 0; i < 64; i++)
            d[i] ^= bitmask[i];
        mul_Nx64_64x64_acc(v[0], d, vnext, n);
        transpose_64x64(d, d);
        mul_64x64_64x64(d, vt_v0[0], vt_v0_next);

        /* compute e */
        mul_64x64_64x64(winv[1], vt_a_v[0], e);
        for (i = 0; i < 64; i++)
            e[i] &= mask0;
        mul_Nx64_64x64_acc(v[1], e, vnext, n);
        transpose_64x64(e, e);
        mul_64x64_64x64(e, vt_v0[1], e);
        for (i = 0; i < 64; i++)
            vt_v0_next[i] ^= e[i];

        /* compute f (if previous v didn't have full rank) */
        if (mask1 != (uint64)(-1)) {
            mul_64x64_64x64(vt_a_v[1], winv[1], f);
            for (i = 0; i < 64; i++)
                f[i] ^= bitmask[i];
            mul_64x64_64x64(winv[2], f, f);
            for (i = 0; i < 64; i++)
                f2[i] = ((vt_a2_v[1][i] & mask1) ^ vt_a_v[1][i]) & mask0;
            mul_64x64_64x64(f, f2, f);
            mul_Nx64_64x64_acc(v[2], f, vnext, n);
            transpose_64x64(f, f);
            mul_64x64_64x64(f, vt_v0[2], f);
            for (i = 0; i < 64; i++)
                vt_v0_next[i] ^= f[i];
        }

        /* update solution x */
        mul_64x64_64x64(winv[0], vt_v0[0], d);
        mul_Nx64_64x64_acc(v[0], d, x, n);

        /* rotate */
        tmp = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = vnext; vnext = tmp;
        tmp = winv[2]; winv[2] = winv[1]; winv[1] = winv[0]; winv[0] = tmp;
        tmp = vt_v0[2]; vt_v0[2] = vt_v0[1]; vt_v0[1] = vt_v0[0];
        vt_v0[0] = vt_v0_next; vt_v0_next = tmp;
        tmp = vt_a_v[1]; vt_a_v[1] = vt_a_v[0]; vt_a_v[0] = tmp;
        tmp = vt_a2_v[1]; vt_a2_v[1] = vt_a2_v[0]; vt_a2_v[0] = tmp;

        memcpy(s[1], s[0], 64 * sizeof(uint32));
        mask1 = mask0;
        dim1 = dim0;
    }

    /* free temporaries */
    free(vnext);
    free(scratch);
    free(v0);
    free(vt_a_v[0]);  free(vt_a_v[1]);
    free(vt_a2_v[0]); free(vt_a2_v[1]);
    free(winv[0]); free(winv[1]); free(winv[2]);
    free(vt_v0_next);
    free(vt_v0[0]); free(vt_v0[1]); free(vt_v0[2]);

    /* check for failure */
    if (dim0 == 0) {
        free(x);
        free(v[0]); free(v[1]); free(v[2]);
        return NULL;
    }

    /* convert output to nullspace vectors */
    mul_MxN_Nx64(packed_matrix, x, v[1]);
    mul_MxN_Nx64(packed_matrix, v[0], v[2]);

    if (post_lanczos_matrix) {
        for (i = 0; i < POST_LANCZOS_ROWS; i++) {
            uint64 accum0 = 0;
            uint64 accum1 = 0;
            uint32 j;
            mask0 = bitmask[i];
            for (j = 0; j < n; j++) {
                if (post_lanczos_matrix[j] & mask0) {
                    accum0 ^= x[j];
                    accum1 ^= v[0][j];
                }
            }
            v[1][i] ^= accum0;
            v[2][i] ^= accum1;
        }
    }

    *num_deps_found = combine_cols(n, x, v[0], v[1], v[2]);

    /* verify */
    mul_MxN_Nx64(packed_matrix, x, v[0]);
    for (i = 0; i < n; i++) {
        if (v[0][i] != 0) break;
    }
    if (i < n) {
        /* dependencies don't verify -- treat as failure */
        free(x);
        free(v[0]); free(v[1]); free(v[2]);
        *num_deps_found = 0;
        return NULL;
    }

    free(v[0]); free(v[1]); free(v[2]);

    return x;
}

/* ================================================================
   Public interface
   ================================================================ */

namespace zfactor::siqs {

uint64_t* block_lanczos(uint32_t* nrows_p, uint32_t num_dense_rows,
                        uint32_t* ncols_p, LaCol* cols,
                        uint32_t* num_deps_found) {

    uint32 nrows = *nrows_p;
    uint32 ncols = *ncols_p;
    uint64 *post_lanczos_matrix;
    uint64 *dependencies;
    packed_matrix_t packed_matrix;

    *num_deps_found = 0;

    if (ncols <= nrows) {
        /* matrix needs more columns than rows */
        return nullptr;
    }

    /* reduce the matrix -- DISABLED for now to keep 1:1 column mapping */
    /* uint32 num_excess = 64;
    reduce_matrix(&nrows, num_dense_rows, &ncols, cols, num_excess);
    if (ncols == 0) {
        *nrows_p = nrows;
        *ncols_p = ncols;
        return nullptr;
    } */

    /* form post-lanczos matrix */
    post_lanczos_matrix = form_post_lanczos_matrix(&nrows,
                                &num_dense_rows, ncols, cols);

    /* Use a default L2 cache size of 256 KB if we can't detect.
       Reasonable for most modern processors. */
    uint32 cache_size2 = 256 * 1024;
    packed_matrix_init(&packed_matrix, cols, nrows, ncols,
                       num_dense_rows, cache_size2);

    /* retry loop with different random seeds */
    uint32 seed1 = 11111;
    uint32 seed2 = 22222;

    for (int attempt = 0; attempt < 10; attempt++) {
        dependencies = block_lanczos_core(&packed_matrix,
                                          num_deps_found,
                                          post_lanczos_matrix,
                                          &seed1, &seed2);
        if (dependencies != NULL)
            break;
        /* perturb seeds for next attempt */
        seed1 += 31337;
        seed2 += 97531;
    }

    packed_matrix_free(&packed_matrix);
    free(post_lanczos_matrix);

    *nrows_p = nrows;
    *ncols_p = ncols;
    return dependencies;
}

void free_cols(LaCol* cols, uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; i++) {
        free(cols[i].data);
        free(cols[i].cycle.list);
    }
    free(cols);
}

} // namespace zfactor::siqs
