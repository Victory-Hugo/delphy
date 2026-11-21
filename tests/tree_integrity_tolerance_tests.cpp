#include "gtest/gtest.h"

#include <cmath>
#include <limits>

#include "phylo_tree.h"

namespace delphy {

namespace {

// Root with two tips.  `t_root` is the root time and `t_tip` the time of both tips, whose float
// bounds are set from `bound_lo` and `bound_hi` (already rounded as the caller wishes).
auto make_cherry(double t_root, double t_tip, double t_child_b, float bound_lo, float bound_hi) -> Phylo_tree {
  auto tree = Phylo_tree{3};
  const auto r = Node_index{0};
  const auto a = Node_index{1};
  const auto b = Node_index{2};
  tree.root = r;
  tree.ref_sequence = Real_sequence{Real_seq_letter::A};
  tree.at(r).parent = k_no_node;
  tree.at(r).children = {a, b};
  tree.at(r).t_min = -std::numeric_limits<float>::max();
  tree.at(r).t_max = +std::numeric_limits<float>::max();
  tree.at(r).t = t_root;
  tree.at(a).parent = r;
  tree.at(a).t = t_tip;
  tree.at(a).t_min = bound_lo;
  tree.at(a).t_max = bound_hi;
  tree.at(b).parent = r;
  tree.at(b).t = t_child_b;
  tree.at(b).t_min = bound_lo;
  tree.at(b).t_max = bound_hi;
  return tree;
}

}  // namespace

// A tip whose float bounds are the plain float rounding of its double time (what partition
// boundaries do) is within half a float spacing of that time: 2 days at -147 ka.  That must pass.
TEST(Tree_integrity_tolerance_test, float_rounded_bounds_at_deep_times_are_accepted) {
  for (auto t : {-5.355983721e7, -6.42956037e6, -1.0e5 - 0.37}) {
    auto f = static_cast<float>(t);
    auto tree = make_cherry(t - 1000.0, t, t, f, f);
    assert_phylo_tree_integrity(tree, /*force=*/true);
  }
}

#ifndef __EMSCRIPTEN__
// The relaxed tolerance scales with the float spacing and no more: a bound violated by more than a
// couple of spacings, or a child older than its parent by more than roundoff, still fails.
TEST(Tree_integrity_tolerance_test, a_time_outside_its_bounds_by_more_than_float_roundoff_is_rejected) {
  auto t = -5.355983721e7;                                   // spacing of floats here is 4 days
  auto f = static_cast<float>(t);
  auto too_late = make_cherry(t - 1000.0, t + 40.0, t + 40.0, f, f);  // 10 spacings above its bound
  EXPECT_DEATH(assert_phylo_tree_integrity(too_late, /*force=*/true), "");
  auto too_early = make_cherry(t - 1000.0, t - 40.0, t - 40.0, f, f);
  EXPECT_DEATH(assert_phylo_tree_integrity(too_early, /*force=*/true), "");
}

TEST(Tree_integrity_tolerance_test, a_child_older_than_its_parent_is_rejected_at_deep_times) {
  auto t = -5.355983721e7;
  auto f = static_cast<float>(t);
  // Root 0.05 days younger than its child: a negative branch length, far below the float spacing
  auto tree = make_cherry(t + 0.05, t, t, f, f);
  EXPECT_DEATH(assert_phylo_tree_integrity(tree, /*force=*/true), "");
}

TEST(Tree_integrity_tolerance_test, pathogen_scale_tolerance_is_unchanged) {
  // At 100 days the float spacing is 7.6e-6, so the tolerance is the original 0.01 days
  auto f = static_cast<float>(100.0);
  auto inside = make_cherry(0.0, 100.005, 100.005, f, f);
  assert_phylo_tree_integrity(inside, /*force=*/true);
  auto outside = make_cherry(0.0, 100.05, 100.05, f, f);
  EXPECT_DEATH(assert_phylo_tree_integrity(outside, /*force=*/true), "");
}
#endif

}  // namespace delphy
