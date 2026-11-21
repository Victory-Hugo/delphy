#ifndef DELPHY_BEASTY_INPUT_H_
#define DELPHY_BEASTY_INPUT_H_

#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "phylo_tree.h"
#include "run.h"

namespace delphy {

// Reads a BEAST-format posterior trees file one tree at a time, so that a summary of a long run does
// not need every tree in memory at once.  The header (taxa block and translate table) is parsed by the
// constructor; next() then parses one `tree STATE_n = ...` line per call.
//
// Tip node indices are the translate-table indices minus one, as in read_beasty_trees(), so
// tip_names()[i] is the name of tip node i and that indexing is the same for every tree in the file.
class Beasty_trees_stream {
 public:
  explicit Beasty_trees_stream(std::istream& is);

  // Names of the tips, indexed by tip node index (0-based).
  auto tip_names() const -> const std::vector<std::string>& { return tip_names_; }
  auto num_tips() const -> int { return static_cast<int>(std::ssize(tip_names_)); }

  // Parses the next tree into `tree` and its state number into `state`.  Returns false at the end of
  // the trees block.  Trees with `state < min_state` or `state % every != 0` are skipped.
  auto next(std::int64_t& state, Phylo_tree& tree) -> bool;

  auto min_state() const -> std::int64_t { return min_state_; }
  auto set_min_state(std::int64_t s) -> void { min_state_ = s; }
  auto every() const -> std::int64_t { return every_; }
  auto set_every(std::int64_t e) -> void { every_ = e; }

  // When set, every tree read or skipped is reported on std::cerr, as read_beasty_trees() does.
  auto set_verbose(bool v) -> void { verbose_ = v; }

 private:
  std::istream* is_;
  std::vector<std::string> tip_names_{};
  std::int64_t min_state_ = std::numeric_limits<std::int64_t>::min();
  std::int64_t every_ = 1;
  bool verbose_ = false;
  bool done_ = false;
};

// State numbers of every tree in a BEAST-format trees file, in file order, without parsing any tree.
// Used to choose a burn-in cut before the trees are read.
auto scan_beasty_tree_states(std::istream& is) -> std::vector<std::int64_t>;

auto read_beasty_trees(
    std::istream& is,
    std::int64_t min_state,
    std::int64_t every = 1)
    -> std::map<std::int64_t, Phylo_tree>;

auto export_beast_input(
    const Run& run,
    std::string_view beast_version,
    std::ostream& os,
    int64_t chain_length = 10000000,
    int64_t log_every = 1000,
    int64_t tree_every = 1000)
    -> void;

}  // namespace delphy

#endif // DELPHY_BEASTY_INPUT_H_
