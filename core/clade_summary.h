#ifndef DELPHY_CLADE_SUMMARY_H_
#define DELPHY_CLADE_SUMMARY_H_

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "calibration.h"
#include "phylo_tree.h"
#include "posterior_stats.h"

namespace delphy {

// Posterior summaries of the age of the most recent common ancestor of user-named sets of samples.
//
// This is a summary of trees that were already sampled; nothing here changes the posterior.  The
// mathematical definitions, the conventions and what the numbers do and do not mean are in
// doc/extended/clade_summary.md.
//
// Two quantities are kept apart throughout, because they answer different questions and coincide only
// when the taxon set is monophyletic in every sampled tree:
//
//   * the *unconditional MRCA age*: the age of the MRCA of the taxon set in every post-burn-in tree,
//     whether or not the set is monophyletic there.  This is the posterior of "the age of the common
//     ancestor of these samples" and is defined in every tree.
//   * the *clade age*: the same quantity restricted to the trees in which the set is monophyletic,
//     i.e. conditional on the clade existing.  Its sample size is the number of such trees.
//
// `Mcc_node::t_mrca` is the mean of the first and `Mcc_node::t` the mean of the second; neither is a
// posterior interval, and neither is reported as one here.

struct Clade_spec_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// One target clade, defined by a stable set of sample IDs and never by a node index.
struct Clade_spec {
  std::string id;                  // Stable identifier used in every output table
  std::vector<std::string> taxa;   // Sorted and unique; empty exactly when `all_tips` is set
  bool all_tips = false;           // "all": the root of every sampled tree
  int config_line = 0;             // 1-based line of the definition, for error messages

  auto num_taxa() const -> int { return static_cast<int>(std::ssize(taxa)); }
};

struct Clade_spec_set {
  std::vector<Clade_spec> specs;

  auto empty() const -> bool { return specs.empty(); }
  auto size() const -> int { return static_cast<int>(std::ssize(specs)); }
};

// Reads a clade definition table.  Tab-separated, a header line naming at least the columns
// `clade_id` and `taxa` is required, other columns are ignored, `#` starts a comment and blank lines
// are skipped.  `taxa` lists sample IDs separated by commas and/or whitespace, or the single word
// `all` for the root of the tree.  Throws Clade_spec_error listing every problem found.
auto read_clade_specs(std::istream& is) -> Clade_spec_set;

// The clades a calibration configuration already defines, so that a calibrated analysis can summarize
// exactly the nodes it calibrated without restating them.  Clade IDs are the calibration names.
auto clade_specs_from_calibrations(const Calibration_set& set) -> Clade_spec_set;

// Resolves clade specs against a fixed tip indexing.  Every tree passed to resolve() must use the
// same tip indices for the same sample IDs, which is what a single Delphy or BEAST .trees file
// guarantees; the caller is responsible for checking that across files.
//
// Set identity is exact: a clade is a sorted list of tip indices, and monophyly is decided by counting
// the tips below the MRCA.  No hashing and no random fingerprints are involved, so two different
// taxon sets can never be silently treated as one.
class Clade_resolver {
 public:
  // `tip_names[i]` is the name of tip node `i`.  Throws Clade_spec_error if a spec names a sample that
  // is not a tip, if a name is duplicated, or if a spec has fewer than two resolvable taxa.
  Clade_resolver(const Clade_spec_set& specs, const std::vector<std::string>& tip_names);

  auto specs() const -> const Clade_spec_set& { return specs_; }
  auto num_clades() const -> int { return specs_.size(); }
  auto num_tips() const -> int { return num_tips_; }

  // What one posterior tree says about one clade.
  struct Observation {
    double t = 0.0;              // Time of the MRCA, in the time axis of the tree as given
    int num_tips_below = 0;      // Tips below the MRCA in this tree
    bool monophyletic = false;   // num_tips_below == the number of taxa in the spec
  };

  // One Observation per clade, in the order of specs().  `tree` must have the tip indexing this
  // resolver was built with.
  auto observe(const Phylo_tree& tree, std::vector<Observation>& out) const -> void;

  // Time of the most recent tip of `tree`.  Ages are measured relative to this when the caller has no
  // absolute anchor.
  static auto youngest_tip_time(const Phylo_tree& tree) -> double;

 private:
  Clade_spec_set specs_;
  int num_tips_ = 0;
  std::vector<std::vector<Node_index>> clade_tips_;  // Sorted tip node indices per clade
  std::vector<bool> clade_is_all_tips_;
};

// Raw posterior samples of one clade, accumulated one tree at a time.
struct Clade_samples {
  std::string clade_id;
  // One entry per post-burn-in tree, in the order the trees were read.  `chain_of_sample` says which
  // chain each came from; chains are numbered from 0 in the order they were given.
  std::vector<double> age_years;
  std::vector<char> monophyletic;
  std::vector<int> num_tips_below;
  std::vector<int> chain_of_sample;

  auto num_samples() const -> int { return static_cast<int>(std::ssize(age_years)); }
};

// Statistics of one scalar quantity, with every convention named.
struct Age_statistics {
  int num_samples = 0;
  int num_chains = 0;
  double mean = 0.0;
  double median = 0.0;
  double stddev = 0.0;
  double credible_mass = 0.95;
  // Shortest contiguous interval holding `credible_mass` of the samples.  It estimates the highest
  // posterior density interval only when `hdr` has a single component; see posterior_stats.h.
  Credible_interval shortest_interval{+1.0, -1.0};
  Credible_interval eti{+1.0, -1.0};   // Equal-tailed interval; not a highest-density interval
  Ess_result ess{};
  double split_rhat = -1.0;            // Negative when undefined (too few samples)
  double rank_split_rhat = -1.0;
  Highest_density_region hdr{};  // The strict highest density region (section 3.3 of the design doc)
  auto is_defined() const -> bool { return num_samples > 0; }
  // True exactly when the highest density region is a single interval, i.e. when the shortest
  // contiguous interval above may legitimately be called a highest posterior density interval.
  auto interval_is_hpd() const -> bool { return hdr.num_components <= 1; }
};

// Everything reported for one clade.
struct Clade_summary {
  std::string clade_id;
  int num_taxa_specified = 0;
  bool all_tips = false;

  int num_samples_total = 0;                 // Post-burn-in trees that entered the summary
  std::vector<int> num_samples_per_chain;

  double posterior_support = 0.0;            // Fraction of post-burn-in trees in which the set is monophyletic
  int num_monophyletic_samples = 0;

  Age_statistics mrca;                       // Unconditional: every post-burn-in tree
  Age_statistics clade;                      // Conditional on monophyly

  // Tips below the MRCA, over all post-burn-in trees.  Equal to num_taxa_specified in every tree
  // exactly when posterior_support == 1.
  int min_tips_below = 0;
  int median_tips_below = 0;
  int max_tips_below = 0;

  std::string calibration_status = "not_calibrated";  // "calibrated:<name>", "not_calibrated", "unknown"

  std::vector<std::string> warnings;         // Diagnostic codes; see the design document
};

struct Summary_thresholds {
  double credible_mass = 0.95;
  double min_ess = 200.0;
  double max_rank_split_rhat = 1.01;
  int hdr_num_bins = 0;  // 0 = automatic
};

// Computes every statistic of one clade from its raw samples.  Chains are identified by
// `chain_of_sample`; per-chain diagnostics need at least 4 samples in a chain.
auto summarize_clade(
    const Clade_samples& samples,
    const Clade_spec& spec,
    int num_chains,
    const Summary_thresholds& thresholds)
    -> Clade_summary;

// Description of the time axis of the reported ages, so that no output table is ambiguous.
struct Age_reference {
  // Text printed in the tables, e.g. "1950-01-01 (cal BP)" or "youngest_tip_in_each_tree".
  std::string description;
  std::string units = "years";
  // True when ages are measured from a fixed calendar instant; false when they are measured from the
  // most recent tip of each posterior tree, which is not a calendar age.
  bool is_absolute = false;
};

// Writers.  All of them print a `#`-prefixed provenance block first; the column names are stable.
struct Summary_provenance {
  std::vector<std::string> input_files;
  std::string burn_in_description;
  std::string age_reference_description;
  std::string anchor_description;
  double credible_mass = 0.95;
  double min_ess = 200.0;
  double max_rank_split_rhat = 1.01;
  std::vector<std::string> notes;
};

auto write_clade_summary_table(
    const std::vector<Clade_summary>& summaries,
    const Summary_provenance& provenance,
    std::ostream& os)
    -> void;

auto write_clade_samples_table(
    const std::vector<Clade_samples>& samples,
    const std::vector<std::int64_t>& states,     // Parallel to the samples of every clade
    const Summary_provenance& provenance,
    std::ostream& os)
    -> void;

// Reads back a table written by write_clade_samples_table().  Lets a summary be recomputed, with a
// different burn-in or credible mass, without re-reading the posterior trees.
struct Clade_samples_file {
  std::vector<Clade_samples> samples;
  std::vector<std::int64_t> states;
  int num_chains = 0;
  std::vector<std::string> comment_lines;
};
auto read_clade_samples_table(std::istream& is) -> Clade_samples_file;

}  // namespace delphy

#endif  // DELPHY_CLADE_SUMMARY_H_
