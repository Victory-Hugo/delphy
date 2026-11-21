#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "absl/log/initialize.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/strings/str_format.h"
#include "cxxopts.hpp"

#include "beasty_input.h"
#include "mcc_tree.h"
#include "newick.h"
#include "version.h"

namespace delphy {

auto output_mcc_tree(const Mcc_tree& mcc_tree, std::ostream& os) -> void {
  auto mcc_node_name = [&](Node_index mcc_node) -> const std::string& {
    CHECK(mcc_tree.at(mcc_node).is_tip());  // Otherwise, "node name" can be different in each base tree
    const auto& base_tree = *mcc_tree.base_trees().front();
    auto base_node = corresponding_node_to(mcc_tree, mcc_node, 0);
    return base_tree.at(base_node).name;
  };
  
  auto num_tips = (std::ssize(mcc_tree) + 1) / 2;
  auto next_tip_index = Node_index{1};
  auto node_to_tip_index = Node_vector<Node_index>(std::ssize(mcc_tree), k_no_node);
  for (const auto& node : index_order_traversal(mcc_tree)) {
    if (mcc_tree.at(node).is_tip()) {
      node_to_tip_index[node] = next_tip_index;
      ++next_tip_index;
    }
  }
  
  os << "#NEXUS\n"
     << "\n"
     << "Begin taxa;\n"
     << "    Dimensions ntax=" << num_tips << ";\n"
     << "        Taxlabels\n";
  
  for (const auto& node : index_order_traversal(mcc_tree)) {
    if (mcc_tree.at(node).is_tip()) {
      os << "            " << newick_label(mcc_node_name(node)) << "\n";
    }
  }

  os << "            ;\n"
     << "End;\n"
     << "Begin trees;\n"
     << "    Translate\n";

  auto first_tip = true;
  for (const auto& node : index_order_traversal(mcc_tree)) {
    if (mcc_tree.at(node).is_tip()) {
      if (first_tip) {
        first_tip = false;
      } else {
        os << ",\n";
      }
      os << absl::StrFormat("        %4d %s", node_to_tip_index[node], newick_label(mcc_node_name(node)));
    }
  }
  os << "\n";

  os << ";\n";

  os << "tree TREE1 = ";

  for (const auto& [node, children_so_far] : traversal(mcc_tree)) {
    if (children_so_far == 0) {
      // Entering `node`
      if (mcc_tree.at(node).is_inner_node()) {
        os << "(";
      }
    }

    if (0 < children_so_far && children_so_far < std::ssize(mcc_tree.at(node).children)) {
      // In between two children
      os << ",";
    }

    if (children_so_far == std::ssize(mcc_tree.at(node).children)) {
      // Exiting `node`
      if (mcc_tree.at(node).is_inner_node()) {
        os << ")";
      }
      if (mcc_tree.at(node).is_tip()) {
        os << node_to_tip_index.at(node);
      }
      
      auto posterior = mcc_tree.at(node).posterior_support();
      os << "[&posterior=" << posterior << "]";
      
      auto days = node == mcc_tree.root
          ? 0.0
          : mcc_tree.at(node).t_mrca() - mcc_tree.at_parent_of(node).t_mrca();
      os << ":" << (days / 365.0);  // Days (internal) -> Years (output)
    }
  }
  
  os << ";\n";
  os << "End;\n";
}

}  // namespace delphy

auto main(int argc, char** argv) -> int {
  using namespace delphy;

  absl::InitializeLog();

  auto scope = Local_arena_scope{};

  auto options = cxxopts::Options{
      "delphy_mcc",
      "Reads a BEAST-format posterior trees file and computes a Maximum Clade Credibility (MCC) tree"};
  options.add_options()
      ("h,help", "print usage")
      ("version", "print version string")
      ("burn-in-fraction",
       "Discard the first fraction of the state range (default 0.3, the value this tool has always used)",
       cxxopts::value<double>())
      ("burn-in-states", "Discard every sampled state <= this number", cxxopts::value<std::int64_t>())
      ("burn-in-samples", "Discard the first this many sampled trees", cxxopts::value<int>())
      ("thin", "Keep only states that are multiples of this number", cxxopts::value<std::int64_t>())
      ("seed",
       "Seed for the random clade fingerprints used to match clades across trees (default: random).  Give a "
       "seed to make the MCC tree reproducible.",
       cxxopts::value<uint32_t>())
      ("input-trees-file", "input trees file", cxxopts::value<std::string>())
      ("output-mcc-file", "output MCC tree file", cxxopts::value<std::string>())
      ;
  options.parse_positional({"input-trees-file", "output-mcc-file"});
  options.positional_help("<input-trees-file> <output-mcc-file>");

  auto args = cxxopts::ParseResult{};
  try {
    args = options.parse(argc, argv);
  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "ERROR on the command line: " << e.what() << "\n\n" << options.help() << "\n";
    return EXIT_FAILURE;
  }

  if (args.count("version")) {
    std::cerr << absl::StreamFormat("Delphy MCC Version %s (build %d, commit %s)",
                                    k_delphy_version_string,
                                    k_delphy_build_number,
                                    k_delphy_commit_string) << "\n";
    std::cerr << absl::StreamFormat("Delphy-Extended %s (extends upstream Delphy at commit %s)",
                                    k_delphy_extended_version_string,
                                    k_delphy_extended_upstream_commit) << "\n";
    return EXIT_SUCCESS;
  }

  if (args.count("help") || not args.count("input-trees-file") || not args.count("output-mcc-file")) {
    std::cerr << absl::StreamFormat("Delphy MCC Version %s (build %d, commit %s)",
                                    k_delphy_version_string,
                                    k_delphy_build_number,
                                    k_delphy_commit_string) << "\n";
    std::cerr << absl::StreamFormat("Delphy-Extended %s (extends upstream Delphy at commit %s)",
                                    k_delphy_extended_version_string,
                                    k_delphy_extended_upstream_commit) << "\n\n";
    std::cerr << options.help() << "\n";
    std::cerr << "An MCC node's `posterior` is the fraction of post-burn-in trees in which its tips form a\n"
              << "clade, and its branch lengths come from the mean time of the MRCA of those tips over *all*\n"
              << "post-burn-in trees.  Neither is a posterior interval; for tMRCA posteriors, credible\n"
              << "intervals, effective sample sizes and multi-chain diagnostics of named clades, use\n"
              << "delphy_clades.\n";
    return args.count("help") ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  auto num_burn_in_options = args.count("burn-in-fraction") + args.count("burn-in-states")
      + args.count("burn-in-samples");
  if (num_burn_in_options > 1) {
    std::cerr << "ERROR: give at most one of --burn-in-fraction, --burn-in-states, --burn-in-samples\n";
    return EXIT_FAILURE;
  }
  auto burn_in_fraction = args.count("burn-in-fraction") ? args["burn-in-fraction"].as<double>() : 0.3;
  if (not (burn_in_fraction >= 0.0 && burn_in_fraction < 1.0)) {
    std::cerr << "ERROR: --burn-in-fraction must be in [0, 1)\n";
    return EXIT_FAILURE;
  }
  auto thin = args.count("thin") ? args["thin"].as<std::int64_t>() : std::int64_t{1};
  if (thin < 1) {
    std::cerr << "ERROR: --thin must be >= 1\n";
    return EXIT_FAILURE;
  }

  auto in_path = args["input-trees-file"].as<std::string>();
  auto out_path = args["output-mcc-file"].as<std::string>();

  // Pass 1: the state numbers, so that a burn-in cut can be chosen without parsing any tree
  auto states = std::vector<std::int64_t>{};
  {
    auto tree_is = std::ifstream{in_path};
    CHECK(tree_is) << "cannot open " << in_path << " for reading";
    for (auto s : scan_beasty_tree_states(tree_is)) {
      if (thin > 1 && (s % thin) != 0) { continue; }
      states.push_back(s);
    }
  }
  if (states.empty()) {
    std::cerr << absl::StreamFormat("ERROR: '%s' contains no trees\n", in_path);
    return EXIT_FAILURE;
  }

  auto min_state_kept = std::int64_t{};
  auto burn_in_description = std::string{};
  if (args.count("burn-in-states")) {
    min_state_kept = args["burn-in-states"].as<std::int64_t>() + 1;
    burn_in_description = absl::StrFormat("every state <= %d", min_state_kept - 1);
  } else if (args.count("burn-in-samples")) {
    auto k = std::clamp(args["burn-in-samples"].as<int>(), 0, static_cast<int>(std::ssize(states)));
    min_state_kept = (k == std::ssize(states)) ? std::numeric_limits<std::int64_t>::max() : states[k];
    burn_in_description = absl::StrFormat("the first %d sampled trees", k);
  } else {
    auto min_state = states.front();
    auto max_state = states.back();
    // floor(fraction * range) reproduces the integer `(range * 3) / 10` of the historical hard-coded
    // 30% burn-in exactly, so the default output of this tool is unchanged.
    min_state_kept = min_state + static_cast<std::int64_t>(
        std::floor(burn_in_fraction * static_cast<double>(max_state - min_state))) + 1;
    burn_in_description = absl::StrFormat("fraction %.6g of the state range %d..%d",
                                          burn_in_fraction, min_state, max_state);
  }

  auto num_burn_in_trees = std::ranges::count_if(
      states, [&](std::int64_t s) { return s < min_state_kept; });
  std::cerr << absl::StreamFormat("Read %d trees", std::ssize(states)) << std::endl;
  std::cerr << absl::StreamFormat("Discarding %d trees as burn-in (%s)",
                                  num_burn_in_trees, burn_in_description) << std::endl;

  // Pass 2: parse only the trees that survive the burn-in cut
  auto trees = std::vector<Phylo_tree>{};
  {
    auto tree_is = std::ifstream{in_path};
    CHECK(tree_is) << "cannot open " << in_path << " for reading";
    auto stream = Beasty_trees_stream{tree_is};
    stream.set_min_state(min_state_kept);
    stream.set_every(thin);
    auto state = std::int64_t{};
    auto tree = Phylo_tree{};
    while (stream.next(state, tree)) { trees.push_back(std::move(tree)); }
  }

  std::cerr << absl::StreamFormat("Read %d post-burnin trees", std::ssize(trees)) << std::endl;
  if (trees.empty()) {
    std::cerr << "ERROR: no trees are left after the burn-in cut\n";
    return EXIT_FAILURE;
  }

  auto base_trees = std::vector<Phylo_tree*>{};
  for (auto& tree : trees) { base_trees.push_back(&tree); }

  std::cerr << absl::StreamFormat("Building MCC with %d trees", std::ssize(base_trees)) << std::endl;

  auto bitgen = absl::BitGen{};
  auto seeded_prng = std::mt19937{args.count("seed") ? args["seed"].as<uint32_t>() : 0u};
  auto mcc_tree = args.count("seed")
      ? derive_mcc_tree(std::move(base_trees), seeded_prng)
      : derive_mcc_tree(std::move(base_trees), bitgen);

  auto mcc_os = std::ofstream{out_path};
  CHECK(mcc_os) << "cannot open " << out_path << " for writing";
  output_mcc_tree(mcc_tree, mcc_os);

  return 0;
}
