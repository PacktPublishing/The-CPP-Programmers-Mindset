#ifndef BENCH_CONTAINERS_H
#define BENCH_CONTAINERS_H

#include <cstdint>
#include <vector>
#include <set>
#include <unordered_set>
#include <forward_list>
#include <list>
#include <deque>

#include <absl/container/btree_set.h>
#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/node_hash_set.h>
#include <absl/container/node_hash_map.h>

#include <boost/container/flat_set.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/container/vector.hpp>
#include <boost/container/stable_vector.hpp>

using StdVec = std::vector<int64_t>;
using StdFList = std::forward_list<int64_t>;
using StdList = std::list<int64_t>;
using StdSet = std::set<int64_t>;
using StdUSet = std::unordered_set<int64_t>;
using StdDeque = std::deque<int64_t>;

inline void push(StdVec &v, int64_t i) {
    v.push_back(i);
}

inline void push(StdFList &v, int64_t i) {
    v.push_front(i);
}

inline void push(StdList &v, int64_t i) {
    v.push_back(i);
}

inline void push(StdDeque &v, int64_t i) {
    v.push_back(i);
}

using AbslBTreeSet = absl::btree_set<int64_t>;
using AbslFlatHashSet = absl::flat_hash_set<int64_t>;
using AbslNodeHashSet = absl::node_hash_set<int64_t>;

using BoostVec = boost::container::vector<int64_t>;
using BoostFlatSet = boost::container::flat_set<int64_t>;
using BoostStableVec = boost::container::stable_vector<int64_t>;

inline void push(BoostStableVec &v, int64_t i) {
    v.push_back(i);
}


inline void push(BoostVec &v, int64_t i) {
    v.push_back(i);
}

template<typename Container>
inline void push(Container &v, int64_t i) {
    v.insert(i);
}


#endif //BENCH_CONTAINERS_H