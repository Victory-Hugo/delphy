#ifndef DELPHY_SITE_PARTITION_H_
#define DELPHY_SITE_PARTITION_H_

#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

#include "evo_model.h"
#include "phylo_tree.h"
#include "sequence.h"

namespace delphy {

// Site partitions: a division of the nucleotide sites into named, non-overlapping sets that share one
// timed tree, one set of HKY85 parameters (kappa and the state frequencies) and one global clock rate,
// but have their own relative substitution rate.
//
// This is NOT the Run / Subrun partitioning of the tree, which splits the MCMC work over threads and
// has nothing to do with the sites (see tree_partitioning.h).  The two are independent and can be used
// together; the tree partitioning pushes the whole evolutionary model, including the per-site-partition
// rates, to every Subrun.
//
// With relative rates r_p and the global rate mu, the substitution rate of partition p is mu * r_p.  The
// rates are normalized so that the weighted mean of r is one,
//
//     sum_p L_p r_p / sum_p L_p == 1,
//
// where the weight L_p is the number of sites of partition p that at least one sample observed.  That
// makes mu the mean substitution rate per observed site, the same quantity a single-partition run
// reports.  See doc/extended/site_partitions.md.

struct Site_partition_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// One block of the configuration: a named set of sites in the coordinates of the input reference, and,
// for variable-sites-only input, the constant sites that the filter removed from this block.
struct Site_partition_block {
  std::string name;
  std::vector<Site_interval> sites;  // merged, sorted, 0-based half-open, input reference coordinates
  Seq_vector<Site_index> constant_sites{0, 0, 0, 0};
  bool has_constant_sites = false;
  int config_line = 0;  // 1-based line of the `partition` keyword, for error messages
};

struct Site_partition_config {
  std::vector<Site_partition_block> blocks;

  auto empty() const -> bool { return blocks.empty(); }
  auto size() const -> int { return static_cast<int>(std::ssize(blocks)); }
  auto total_constant_sites() const -> Site_index;
  // Parallel to `blocks`; empty if no block declares constant sites.
  auto constant_sites_per_block() const -> std::vector<Seq_vector<Site_index>>;
};

// Parses a site-partition configuration.  Throws Site_partition_error listing every problem found.
// Format:
//
//   version 1
//
//   partition coding
//     sites 1-300,500-800
//     constant_sites A=100,C=90,G=80,T=70
//   end
//
// `#` starts a comment and blank lines are ignored.  Sites are 1-based inclusive single sites or
// ranges in the coordinates of the input reference, comma-separated; the `sites` key may be repeated.
// `constant_sites` is optional and only meaningful for variable-sites-only input: it declares, for this
// partition, how many constant sites the variability filter removed, by the state every sample shared
// there (see doc/extended/observation.md section 5.2).  Either every block declares
// it or none does.
auto read_site_partition_config(std::istream& is) -> Site_partition_config;

// Canonical text of a configuration: re-reading this gives the same Site_partition_config.  Used to
// store the configuration inside .dphy files.
auto write_site_partition_config(const Site_partition_config& config, std::ostream& os) -> void;
auto site_partition_config_to_string(const Site_partition_config& config) -> std::string;

// The configuration resolved against the analysed sequence.
struct Site_partition {
  std::vector<std::string> names;
  Site_vector<Partition_index> partition_for_site;  // one entry per analysed site
  std::vector<Site_index> num_sites;                // analysed sites per partition
  std::vector<Site_index> num_observed_sites;       // L_p: sites observed in at least one sample

  auto num_partitions() const -> int { return static_cast<int>(std::ssize(names)); }
  auto total_observed_sites() const -> Site_index;
};

// A single partition covering every analysed site, which is what every run without a configuration
// uses.  `site_is_observed` has one entry per analysed site.
auto make_single_site_partition(const std::vector<bool>& site_is_observed) -> Site_partition;

// Resolves a configuration against an analysed sequence of `num_input_sites` sites in reference
// coordinates followed by the restored constant-site blocks, in configuration order and in the layout
// human_input appends them in (for each partition, the A sites, then C, then G, then T).
// `site_is_observed` has one entry per analysed site.
//
// Throws Site_partition_error listing every problem found: sites outside the reference, overlapping
// partitions, input sites that no partition claims, a partition without a single observed site
// (its relative rate would not be identified), and a mismatch between the declared constant sites and
// the length of the analysed sequence.
auto resolve_site_partition(
    const Site_partition_config& config,
    Site_index num_input_sites,
    const std::vector<bool>& site_is_observed)
    -> Site_partition;

// Which analysed sites at least one sample observed, read from a tree: a site missing in every tip is
// missing at the root by the missation factoring invariant, and no MCMC move changes that set, so this
// is a property of the data rather than of the sampled state.
auto observed_sites_of_tree(const Phylo_tree& tree) -> std::vector<bool>;

// Scales `rates` so that sum_p L_p r_p / sum_p L_p == 1.  Throws Site_partition_error if a rate is not
// finite and positive, or if the sizes do not match.
auto normalize_site_partition_rates(const Site_partition& partition, std::vector<double> rates)
    -> std::vector<double>;

// sum_p L_p r_p / sum_p L_p, which the normalization makes 1.
auto weighted_mean_site_partition_rate(const Site_partition& partition, const std::vector<double>& rates)
    -> double;

}  // namespace delphy

#endif  // DELPHY_SITE_PARTITION_H_
