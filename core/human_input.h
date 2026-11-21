#ifndef DELPHY_HUMAN_INPUT_H_
#define DELPHY_HUMAN_INPUT_H_

#include <istream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "human_dates.h"
#include "io.h"
#include "sequence.h"

namespace delphy {

// Input path for human uniparental (mtDNA) data.  It is entered only on request; the pathogen input path
// (read_fasta / read_maple / fasta_to_maple) is untouched.  Differences from that path:
//
//   * Sampling times come from a separate metadata table matched by sample ID, never from the sequence ID.
//   * Nothing is dropped silently: any invalid sample, date or reference inconsistency rejects the whole input
//     with a list of every problem found.
//   * The reference sequence is explicit and its length is checked against the declared standard, so that
//     a trimmed or SNP-only alignment cannot be mistaken for a full-length one.

// Thrown for every rejected input.  what() lists the problems, one per line.
struct Human_input_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class Reference_standard { rcrs, rsrs, custom };

inline constexpr Site_index k_rcrs_length = 16569;  // rCRS and RSRS share rCRS numbering

auto to_string(Reference_standard standard) -> std::string_view;
auto parse_reference_standard(std::string_view name) -> Reference_standard;  // throws Human_input_error

struct Human_input_config {
  Reference_standard reference_standard = Reference_standard::custom;

  // Required for `custom`; for rCRS / RSRS it must be empty or equal to k_rcrs_length.
  std::optional<Site_index> expected_length;

  // FASTA input only: ID of the record in the input FASTA that serves as the reference (it is not a sample).
  // Alternatively pass a separate reference stream to load_human_fasta().
  std::optional<std::string> reference_id;

  // ISO date (YYYY-MM-DD, YYYY-MM or YYYY) given to every sample without its own date in the metadata.
  std::optional<std::string> uniform_sample_date;

  // A sample whose fraction of missing sites (N, gaps, ambiguity codes, masked sites) exceeds this is rejected.
  double max_missing_fraction = 0.10;

  // SNP-only completion (observation mode C1 of doc/extended/observation.md).  The input
  // alignment kept only the sites that passed a variability filter; the counts given here are the sites that
  // were removed because every sample shared one state, listed by that state.  The removed sites are restored
  // as a block of constant sites appended after the input reference, which makes the likelihood of the
  // completed data equal to the likelihood of the full-length data (sites are independent and exchangeable
  // within a partition).  The appended sites are treated as callable in every sample, which is the assumption
  // the declaration encodes; if it is false, the full-length or variant-plus-callable input must be used
  // instead.  Empty means no completion, which leaves SNP-only input rejected.
  //
  // One entry per site partition, in the order of the site-partition configuration (one entry without
  // partitions): partition p's block is appended after partition p-1's, and inside a block the A sites come
  // first, then C, G and T.  `resolve_site_partition` (core/site_partition.h) assigns the appended sites to
  // the partitions in exactly that layout.
  std::vector<Seq_vector<Site_index>> snp_only_constant_sites;
};

// Parses "A=1000,C=900,G=850,T=1100" (all four states required, counts >= 0) for
// Human_input_config::snp_only_constant_sites.  Throws Human_input_error.
auto parse_constant_site_counts(std::string_view spec) -> Seq_vector<Site_index>;

// One row of the sample metadata table (tab-separated, header required):
//
//   sample_id         required; must match the sequence ID exactly (first whitespace-delimited token of a FASTA
//                     header, or of a MAPLE '>' line)
//   date_convention   iso | ce | bce | bp | cal_bp | c14_uncal_bp ; empty => use uniform_sample_date
//   date_from         older bound (or only value) in the units of date_convention
//   date_to           younger bound; empty => point (year conventions) or whole period (iso)
//   bp_reference_year baseline year CE: required for bp, 1950 or empty for cal_bp, empty otherwise
//
// Other columns are ignored but listed in the audit report.
struct Sample_metadata_row {
  int line = 0;  // 1-based line number in the metadata file
  std::string sample_id;
  std::optional<Sample_date_input> date;  // empty if date_convention is empty
};
struct Sample_metadata {
  std::vector<Sample_metadata_row> rows;
  std::vector<std::string> ignored_columns;
};
auto read_sample_metadata(std::istream& is) -> Sample_metadata;

// Site mask: one entry per line, 1-based inclusive: "303-315" or "310".  '#' starts a comment.  Returns merged,
// sorted, 0-based half-open intervals.  Masked sites are treated as missing in every sample.
auto read_site_mask(std::istream& is, Site_index num_sites) -> std::vector<Site_interval>;

// Callable sites: same file format as the site mask, with the opposite meaning.  The listed sites are the ones
// that were examined; every site outside the list is missing in every sample, and no call may contradict the
// declaration.  Returns merged, sorted, 0-based half-open intervals of callable sites.
auto read_callable_sites(std::istream& is, Site_index num_sites) -> std::vector<Site_interval>;

// Per-sample callable sites: one line per sample, "<sample_id>\t<intervals>", where <intervals> is a
// comma-separated list of 1-based inclusive sites or ranges ("310", "303-315"), or "none".  '#' starts a
// comment.  Every sample of the input must appear exactly once.  Sites outside a sample's list are missing in
// that sample.
struct Sample_callable_row {
  int line = 0;
  std::string sample_id;
  std::vector<Site_interval> callable;  // merged, sorted, 0-based half-open
};
struct Sample_callable_sites {
  std::vector<Sample_callable_row> rows;
};
auto read_sample_callable_sites(std::istream& is, Site_index num_sites) -> Sample_callable_sites;

// Variant table: tab-separated with a header line, required columns sample_id, site, base; 'site' is 1-based in
// reference coordinates and 'base' is a single unambiguous A, C, G or T.  '#' lines and blank lines are ignored.
// Other columns are ignored and listed in the audit report.  A variant list says nothing about the sites it does
// not mention, so this input requires an explicit callable declaration; sites that are neither listed as a
// variant nor declared callable are missing, never reference states.
struct Variant_record {
  int line = 0;
  std::string sample_id;
  Site_index site = 0;  // 0-based
  Real_seq_letter base = Real_seq_letter::A;
};
struct Variant_table {
  std::vector<Variant_record> records;
  std::vector<std::string> ignored_columns;
};
auto read_variant_table(std::istream& is, Site_index num_sites) -> Variant_table;

// The observation inputs that declare which sites were examined.  All are optional, except that variant input
// requires at least one callable declaration.  `mask` excludes sites deliberately (a call inside the mask is
// dropped and counted); `callable` and `sample_callable` declare what was examined (a non-reference call outside
// the declaration is a contradiction and an error).
struct Human_observation_inputs {
  std::istream* mask = nullptr;
  std::istream* callable = nullptr;
  std::istream* sample_callable = nullptr;
};

struct Human_sample_audit {
  std::string sample_id;
  Site_index num_callable = 0;          // sites observed in this sample (num_sites minus its missing sites)
  std::string date_source;  // "metadata" or "uniform"
  std::string date_convention;
  std::string date_text;    // The date as given, e.g. "cal_bp 3200..3100" or "iso 2015-06-01"
  double t_min = 0.0;       // days since 2020-01-01
  double t_max = 0.0;
  Site_index num_differences = 0;       // unambiguous differences from the reference
  Site_index num_missing = 0;           // sites treated as missing, after masking
  Site_index num_masked_overrides = 0;  // sites made missing by the mask that had an observed base
  // Breakdown of the characters in the input; -1 when the format (MAPLE) does not carry them
  Site_index num_iupac_ambiguous = -1;  // R Y S W K M B D H V (treated as missing, heteroplasmy is not modelled)
  Site_index num_unknown = -1;          // N or ?
  Site_index num_gap = -1;              // - or .  (treated as missing, not as a deletion)
  double missing_fraction = 0.0;
  int duplicate_group = -1;             // index into Human_audit::duplicate_groups, or -1
};

struct Human_audit {
  int report_version = 3;
  std::string input_format;  // "fasta", "maple" or "variants"
  // How the input determines the observed state of every site in every sample:
  //   "full_alignment"            every site of every sample is given explicitly (FASTA)
  //   "reference_plus_deltas"     reference plus per-sample differences and missing regions (MAPLE)
  //   "variants_plus_callable"    variant list plus declared callable sites
  // A "+completed_constant_sites" suffix marks input whose removed constant sites were restored from a
  // declared count (SNP-only completion).
  std::string observation_mode;
  std::string reference_standard;
  std::string reference_id;
  Site_index num_sites = 0;        // sites analysed, including any appended constant-site block
  Site_index num_input_sites = 0;  // sites of the input reference, in reference coordinates
  std::string reference_checksum_fnv1a64;  // over the input reference as upper-case ACGT text
  // Constant sites appended by the SNP-only completion, by state, summed over the site partitions; all zero
  // without completion.  These sites carry no reference coordinates: they occupy
  // [num_input_sites, num_sites) at the end of the analysed sequence, partition by partition and, inside a
  // partition, in the order A, C, G, T.
  Seq_vector<Site_index> num_constant_sites_appended{0, 0, 0, 0};
  std::vector<Seq_vector<Site_index>> num_constant_sites_appended_per_partition;

  int num_samples = 0;
  Site_index num_masked_sites = 0;
  // Sites excluded because a callable declaration did not list them (missing in every sample), and calls that
  // the declaration overrode (reference-matching calls at sites declared not callable).
  Site_index num_non_callable_sites = 0;
  bool callable_declared = false;
  bool sample_callable_declared = false;
  Site_index num_variable_sites = 0;            // some sample differs from the reference
  Site_index num_invariant_observed_sites = 0;  // observed in >= 1 sample, never different from the reference
  // Sites where every observed sample shares the same state and that state is not the reference base.  They
  // are invariant data (no polymorphism) and are also counted in num_variable_sites, which keeps its meaning
  // "differs from the reference somewhere".
  Site_index num_invariant_nonreference_sites = 0;
  Site_index num_never_observed_sites = 0;      // missing or masked in every sample

  int num_dates_from_metadata = 0;
  int num_dates_from_uniform = 0;
  double earliest_t_min = 0.0;
  double latest_t_max = 0.0;
  double midpoint_span_days = 0.0;  // latest minus earliest sample-time midpoint
  // True if every sample has the same sampling bounds: the data then carry no information about the
  // absolute time scale.
  bool no_time_signal = false;

  std::vector<std::vector<std::string>> duplicate_groups;  // sample IDs with identical processed sequences
  std::vector<std::string> ignored_metadata_columns;
  std::vector<std::string> warnings;
  std::vector<Human_sample_audit> samples;
};

struct Human_input_result {
  Maple_file maple;  // Tips are named by sample ID, in input order; dates are filled in
  Human_audit audit;
};

// FASTA: all records must be aligned to the reference (same length).  Supply the reference either as
// `reference_is` (exactly one record) or via config.reference_id.  `metadata` may be null only if
// config.uniform_sample_date is set.  `mask` may be null.
auto load_human_fasta(
    std::istream& fasta_is,
    std::istream* reference_is,
    const Sample_metadata* metadata,
    std::istream* mask_is,
    const Human_input_config& config)
    -> Human_input_result;
auto load_human_fasta(
    std::istream& fasta_is,
    std::istream* reference_is,
    const Sample_metadata* metadata,
    const Human_observation_inputs& observation,
    const Human_input_config& config)
    -> Human_input_result;

// MAPLE: the first record is the full-length reference.  config.reference_id must be empty.
auto load_human_maple(
    std::istream& maple_is,
    const Sample_metadata* metadata,
    std::istream* mask_is,
    const Human_input_config& config)
    -> Human_input_result;
auto load_human_maple(
    std::istream& maple_is,
    const Sample_metadata* metadata,
    const Human_observation_inputs& observation,
    const Human_input_config& config)
    -> Human_input_result;

// Variant list input.  The sample set is the metadata table (so that samples without variants are present and
// every sample has a declared date), the reference comes from a separate FASTA file, and `observation` must
// declare the callable sites.
auto load_human_variants(
    std::istream& variants_is,
    std::istream& reference_is,
    const Sample_metadata& metadata,
    const Human_observation_inputs& observation,
    const Human_input_config& config)
    -> Human_input_result;

auto write_human_audit_json(const Human_audit& audit, std::ostream& os) -> void;

// Human-readable summary, including warnings
auto write_human_audit_summary(const Human_audit& audit, std::ostream& os) -> void;

}  // namespace delphy

#endif // DELPHY_HUMAN_INPUT_H_
