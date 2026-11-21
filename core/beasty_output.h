#ifndef DELPHY_BEASTY_OUTPUT_H_
#define DELPHY_BEASTY_OUTPUT_H_

#include <fstream>

#include "run.h"

namespace delphy {

// 1-based tip indices in node index order, as a trees file's translate table assigns them.
// k_no_node for inner nodes.
auto beasty_tip_indices(const Phylo_tree& tree) -> Node_vector<Node_index>;

// Writes one tree in the Newick form a BEAST-format trees file carries: tip labels are the 1-based
// indices of `node_to_tip`, branch lengths and mutation times are in years, printed with
// `significant_digits` significant digits.  Exposed so that the serialization precision of deep node
// ages can be measured without running an MCMC chain.
auto write_beasty_newick_tree(
    const Phylo_tree& tree,
    const Node_vector<Node_index>& node_to_tip,
    int significant_digits,
    std::ostream& os)
    -> void;

class Beasty_log_output_version_impl;  // Opaque implementer for each BEAST version

class Beasty_log_output {
 public:
  Beasty_log_output(std::ostream* os, std::string beast_version, bool own_stream = false);
  ~Beasty_log_output();

  auto os() -> std::ostream* { return os_; }
  
  auto output_headers(const Run& run) -> void;
  auto output_log(const Run& run) -> void;
  auto output_footers(const Run& run) -> void;
  auto flush() -> void;

 private:
  std::ostream* os_;
  std::string beast_version_;
  std::unique_ptr<Beasty_log_output_version_impl> impl_;
  bool own_stream_;
};

class Beasty_trees_output {
 public:
  Beasty_trees_output(std::ostream* os, std::string beast_version, bool own_stream = false);
  ~Beasty_trees_output();

  auto os() -> std::ostream* { return os_; }
  
  auto output_headers(const Run& run) -> void;
  auto output_tree(const Run& run) -> void;
  auto output_footers(const Run& run) -> void;
  auto flush() -> void;

  // Significant digits of the branch lengths and mutation times.  The default, 6, is what
  // std::ostream and "%g" print and is what every released version of Delphy has written.  Human
  // time scales need more: a 6-digit branch length of 140,000 years is quantized to 0.1 years, which
  // accumulates along a root-to-tip path (see doc/extended/clade_summary.md).
  auto significant_digits() const -> int { return significant_digits_; }
  auto set_significant_digits(int digits) -> void;

 private:
  std::ostream* os_;
  std::string beast_version_;
  bool own_stream_;
  int significant_digits_ = 6;
  Node_vector<Node_index> node_to_tip_{};

  void output_newick_tree(const Phylo_tree& tree);
};

}  // namespace delphy

#endif // DELPHY_BEASTY_OUTPUT_H_
