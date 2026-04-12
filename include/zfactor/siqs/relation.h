#pragma once

// SIQS Relation Storage and Large Prime Cycle Detection.
//
// Relations come in three flavors:
//   Full:   Q(x) factors completely over the factor base.
//   SLP:    Q(x) = (FB primes) * L where L is a single large prime.
//   DLP:    Q(x) = (FB primes) * L1 * L2 where both L_i are large primes.
//
// Full relations directly contribute to the matrix.
// SLP relations with matching large primes combine into effective full relations.
// DLP relations form a graph where large primes are vertices and relations are edges;
// cycles in this graph produce effective full relations.
//
// We track the graph with a union-find structure following msieve's approach:
// the number of effective relations = num_full + num_cycles + components - vertices,
// where components and vertices come from the DLP graph.

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace zfactor::siqs {

struct Relation {
    uint32_t sieve_offset;
    uint32_t poly_idx;      // B-poly index within current A
    uint32_t a_poly_idx;    // A-poly index (global)
    uint8_t  sign;          // 0 = positive, 1 = negative
    std::vector<uint16_t> fb_offsets;  // FB indices (with multiplicity)
    uint64_t large_prime[2];           // 1 if no large prime
};

struct RelationSet {
    // All relations (full, SLP, DLP) stored in one flat array.
    // The relation type is determined by large_prime values:
    //   both == 1: full
    //   one != 1:  SLP
    //   both != 1: DLP
    std::vector<Relation> relations;

    // Counts
    uint32_t num_full = 0;
    uint32_t num_slp = 0;
    uint32_t num_dlp = 0;

    // SLP matching: maps large_prime -> index of first SLP relation with that prime
    std::unordered_map<uint64_t, uint32_t> slp_map;
    uint32_t num_slp_matched = 0;

    // DLP graph tracking (simplified cycle counting)
    // Maps each large prime to list of relation indices
    std::unordered_map<uint64_t, std::vector<uint32_t>> dlp_graph;
    uint32_t dlp_components = 0;
    uint32_t dlp_vertices = 0;
    uint32_t dlp_cycles = 0;

    void add_relation(Relation&& rel) {
        bool is_full = (rel.large_prime[0] == 1 && rel.large_prime[1] == 1);
        bool is_slp = (!is_full && (rel.large_prime[0] == 1 || rel.large_prime[1] == 1));

        uint32_t idx = (uint32_t)relations.size();

        if (is_full) {
            num_full++;
        } else if (is_slp) {
            uint64_t lp = std::max(rel.large_prime[0], rel.large_prime[1]);
            rel.large_prime[0] = 1;
            rel.large_prime[1] = lp;
            auto it = slp_map.find(lp);
            if (it == slp_map.end()) {
                slp_map[lp] = idx;
            } else {
                num_slp_matched++;
            }
            num_slp++;
        } else {
            // DLP
            uint64_t lp1 = rel.large_prime[0];
            uint64_t lp2 = rel.large_prime[1];

            // Update graph: add edge (lp1, lp2)
            bool lp1_new = (dlp_graph.find(lp1) == dlp_graph.end());
            bool lp2_new = (dlp_graph.find(lp2) == dlp_graph.end());

            dlp_graph[lp1].push_back(idx);
            dlp_graph[lp2].push_back(idx);

            if (lp1_new && lp2_new) {
                // Two new vertices, one new component
                dlp_vertices += 2;
                dlp_components++;
            } else if (lp1_new || lp2_new) {
                // One new vertex added to existing component
                dlp_vertices++;
            } else {
                // Edge between existing vertices: might create cycle
                // Heuristic: each edge that doesn't add a vertex creates potential cycle
                dlp_cycles++;
            }

            num_dlp++;
        }

        relations.push_back(std::move(rel));
    }

    // Total effective relations (usable for linear algebra)
    uint32_t effective_count() const {
        return num_full + num_slp_matched + dlp_cycles;
    }

    // Do we have enough relations?
    bool have_enough(uint32_t fb_size) const {
        return effective_count() >= fb_size + 64;
    }
};

} // namespace zfactor::siqs
