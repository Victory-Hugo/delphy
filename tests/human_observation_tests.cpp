#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "ctpl_stl.h"

#include "api.h"
#include "cmdline.h"
#include "evo_hky.h"
#include "external_tree.h"
#include "human_input.h"
#include "phylo_tree_calc.h"
#include "independent_seq_density.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

//
// Test data: a small alignment whose site classes are known by construction
//

struct Dataset {
  std::string reference;
  std::vector<std::string> ids;
  std::vector<std::string> sequences;  // aligned to the reference, 'N' for missing

  auto num_sites() const -> Site_index { return static_cast<Site_index>(std::ssize(reference)); }
  auto num_samples() const -> int { return static_cast<int>(std::ssize(ids)); }
};

// Four samples over 20 sites with every site class that matters: invariant sites equal to the reference,
// variable sites, a site that is invariant but differs from the reference in every sample, a site missing in
// one sample only, and a site missing in every sample.
auto make_dataset() -> Dataset {
  auto d = Dataset{};
  d.reference = "ACGTACGTACGTACGTACGT";
  d.ids = {"S0", "S1", "S2", "S3"};
  d.sequences = {
      "ACGTACGTACGTACGTACGT",
      "ACGTACGTACGTACGTACGT",
      "ACGTACGTACGTACGTACGT",
      "ACGTACGTACGTACGTACGT",
  };
  auto set = [&](int sample, int site_1based, char c) { d.sequences[sample][site_1based - 1] = c; };
  set(1, 3, 'T');  set(2, 3, 'T');                 // variable site, two samples share the derived base
  set(3, 7, 'A');                                  // variable site, one sample
  set(0, 11, 'C'); set(1, 11, 'C');
  set(2, 11, 'C'); set(3, 11, 'C');                // invariant, but different from the reference G
  set(0, 15, 'N');                                 // missing in one sample only
  for (auto i = 0; i != 4; ++i) { set(i, 20, 'N'); }  // never observed
  return d;
}

auto to_fasta(const Dataset& d) -> std::string {
  auto text = std::string{};
  for (auto i = 0; i != d.num_samples(); ++i) { absl::StrAppend(&text, ">", d.ids[i], "\n", d.sequences[i], "\n"); }
  return text;
}

auto reference_fasta(const Dataset& d) -> std::string { return absl::StrCat(">REF\n", d.reference, "\n"); }

auto metadata_text(const Dataset& d) -> std::string {
  auto text = std::string{"sample_id\tdate_convention\tdate_from\n"};
  for (auto i = 0; i != d.num_samples(); ++i) {
    absl::StrAppend(&text, d.ids[i], "\tiso\t", absl::StrFormat("20%02d-01-01", 10 + i), "\n");
  }
  return text;
}

auto config_for(const Dataset& d) -> Human_input_config {
  auto config = Human_input_config{};
  config.reference_standard = Reference_standard::custom;
  config.expected_length = d.num_sites();
  config.max_missing_fraction = 1.0;
  return config;
}

auto parse_metadata(const std::string& text) -> Sample_metadata {
  auto is = std::istringstream{text};
  return read_sample_metadata(is);
}

struct Observation_files {
  std::optional<std::string> mask;
  std::optional<std::string> callable;
  std::optional<std::string> sample_callable;
};

auto with_mask(std::string text) -> Observation_files {
  auto files = Observation_files{};
  files.mask = std::move(text);
  return files;
}
auto with_callable(std::string text) -> Observation_files {
  auto files = Observation_files{};
  files.callable = std::move(text);
  return files;
}
auto with_sample_callable(std::string text) -> Observation_files {
  auto files = Observation_files{};
  files.sample_callable = std::move(text);
  return files;
}

template<typename F>
auto with_observation(const Observation_files& files, F&& f) {
  auto mask_is = std::istringstream{files.mask.value_or("")};
  auto callable_is = std::istringstream{files.callable.value_or("")};
  auto sample_callable_is = std::istringstream{files.sample_callable.value_or("")};
  auto observation = Human_observation_inputs{};
  if (files.mask.has_value()) { observation.mask = &mask_is; }
  if (files.callable.has_value()) { observation.callable = &callable_is; }
  if (files.sample_callable.has_value()) { observation.sample_callable = &sample_callable_is; }
  return f(observation);
}

auto load_fasta(const Dataset& d, const Human_input_config& config, const Observation_files& files = {})
    -> Human_input_result {
  auto fasta_is = std::istringstream{to_fasta(d)};
  auto ref_is = std::istringstream{reference_fasta(d)};
  auto metadata = parse_metadata(metadata_text(d));
  return with_observation(files, [&](const Human_observation_inputs& observation) {
    return load_human_fasta(fasta_is, &ref_is, &metadata, observation, config);
  });
}

auto load_variants(const Dataset& d, const std::string& variants, const Human_input_config& config,
                   const Observation_files& files) -> Human_input_result {
  auto variants_is = std::istringstream{variants};
  auto ref_is = std::istringstream{reference_fasta(d)};
  auto metadata = parse_metadata(metadata_text(d));
  return with_observation(files, [&](const Human_observation_inputs& observation) {
    return load_human_variants(variants_is, ref_is, metadata, observation, config);
  });
}

template<typename F>
auto expect_error(F&& f, const std::string& fragment) -> void {
  try {
    f();
    ADD_FAILURE() << "expected Human_input_error containing: " << fragment;
  } catch (const Human_input_error& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr(fragment));
  }
}

//
// Observed site patterns: what the input determines, independently of its format
//

// One entry per analysed site: the reference base followed by the state of every sample ('-' for missing).
auto site_patterns(const Human_input_result& result) -> std::vector<std::string> {
  const auto& maple = result.maple;
  auto L = static_cast<Site_index>(std::ssize(maple.ref_sequence));
  auto n = static_cast<int>(std::ssize(maple.tip_descs));
  auto patterns = std::vector<std::string>(L);
  for (auto l = Site_index{0}; l != L; ++l) {
    patterns[l] = std::string(n + 1, to_char(maple.ref_sequence[l]));
  }
  for (auto i = 0; i != n; ++i) {
    for (const auto& [start, end] : maple.tip_descs[i].missations.intervals) {
      for (auto l = start; l != end; ++l) { patterns[l][i + 1] = '-'; }
    }
    for (const auto& d : maple.tip_descs[i].seq_deltas) { patterns[d.site][i + 1] = to_char(d.to); }
  }
  return patterns;
}

auto make_evo(Site_index num_sites) -> Global_evo_model {
  auto evo = make_single_partition_global_evo_model(num_sites);
  evo.partition_evo_model[0] = Hky_model{.mu = 3e-4, .kappa = 2.5, .pi_a = {0.31, 0.19, 0.17, 0.33}}
                                   .derive_site_evo_model();
  evo.nu_l.assign(num_sites, 1.0);
  return evo;
}

auto delphy_log_G(const Phylo_tree& tree, const Global_evo_model& evo) -> double {
  return calc_log_root_prior(tree, evo) + calc_log_G_below_root(tree, evo);
}

// The four-tip topology the equivalence tests share; branch lengths are days, so both datasets get identical
// node times and only the mutation history can differ.
constexpr auto k_newick = "(((S0:400,S1:400):600,S2:1000):900,S3:1900);";

auto build_tree(Maple_file maple) -> Phylo_tree {
  auto is = std::istringstream{k_newick};
  auto ntree = read_newick_tree(is);
  auto prng = std::mt19937{7};
  auto options = External_tree_options{};
  options.units = Branch_length_units::days;
  return build_phylo_tree_from_newick(ntree, std::move(maple.ref_sequence), std::move(maple.tip_descs), prng,
                                      options);
}

}  // namespace

//
// 1. Lossless formats: a variant list with a callable declaration is the same observation as the alignment
//

TEST(Human_observation_test, variant_input_reproduces_the_equivalent_full_alignment) {
  auto d = make_dataset();
  auto from_fasta = load_fasta(d, config_for(d));

  // The same observation as a variant list: every difference from the reference, plus the sites that were
  // examined in each sample (S0 misses site 15, every sample misses site 20).
  auto variants = std::string{"sample_id\tsite\tbase\n"
                              "S1\t3\tT\n"
                              "S2\t3\tT\n"
                              "S3\t7\tA\n"
                              "S0\t11\tC\n"
                              "S1\t11\tC\n"
                              "S2\t11\tC\n"
                              "S3\t11\tC\n"};
  auto files = with_sample_callable("S0\t1-14,16-19\n"
                                   "S1\t1-19\n"
                                   "S2\t1-19\n"
                                   "S3\t1-19\n");
  auto from_variants = load_variants(d, variants, config_for(d), files);

  EXPECT_EQ(from_variants.audit.observation_mode, "variants_plus_callable");
  EXPECT_EQ(from_fasta.audit.observation_mode, "full_alignment");
  EXPECT_THAT(site_patterns(from_variants), testing::Eq(site_patterns(from_fasta)));
  ASSERT_EQ(std::ssize(from_variants.maple.tip_descs), std::ssize(from_fasta.maple.tip_descs));
  for (auto i = 0; i != std::ssize(from_fasta.maple.tip_descs); ++i) {
    const auto& a = from_fasta.maple.tip_descs[i];
    const auto& b = from_variants.maple.tip_descs[i];
    EXPECT_EQ(a.name, b.name);
    EXPECT_THAT(b.seq_deltas, testing::ElementsAreArray(a.seq_deltas)) << a.name;
    EXPECT_EQ(b.missations.intervals, a.missations.intervals) << a.name;
  }
  EXPECT_EQ(from_variants.audit.num_variable_sites, from_fasta.audit.num_variable_sites);
  EXPECT_EQ(from_variants.audit.num_invariant_observed_sites, from_fasta.audit.num_invariant_observed_sites);
  EXPECT_EQ(from_variants.audit.num_never_observed_sites, from_fasta.audit.num_never_observed_sites);
}

TEST(Human_observation_test, the_same_observation_gives_the_same_likelihood_from_either_format) {
  auto d = make_dataset();
  auto variants = std::string{"sample_id\tsite\tbase\nS1\t3\tT\nS2\t3\tT\nS3\t7\tA\n"
                              "S0\t11\tC\nS1\t11\tC\nS2\t11\tC\nS3\t11\tC\n"};
  auto files = with_sample_callable("S0\t1-14,16-19\nS1\t1-19\nS2\t1-19\nS3\t1-19\n");
  auto tree_a = build_tree(load_fasta(d, config_for(d)).maple);
  auto tree_b = build_tree(load_variants(d, variants, config_for(d), files).maple);
  auto evo = make_evo(d.num_sites());

  EXPECT_NEAR(delphy_log_G(tree_a, evo), delphy_log_G(tree_b, evo), 1e-12);
  EXPECT_NEAR(delphy_log_G(tree_a, evo), independent_log_G(tree_a, evo), 1e-9);
  EXPECT_NEAR(delphy_log_G(tree_b, evo), independent_log_G(tree_b, evo), 1e-9);
}

TEST(Human_observation_test, never_observed_sites_do_not_enter_the_likelihood) {
  auto d = make_dataset();
  auto wide = d;
  wide.reference += "GGGG";
  for (auto& s : wide.sequences) { s += "NNNN"; }
  auto config = config_for(d);
  auto wide_config = config;
  wide_config.expected_length = wide.num_sites();

  auto narrow_tree = build_tree(load_fasta(d, config).maple);
  auto wide_result = load_fasta(wide, wide_config);
  auto wide_tree = build_tree(wide_result.maple);

  EXPECT_EQ(wide_result.audit.num_never_observed_sites, 5);  // site 20 of the dataset plus the four new ones
  EXPECT_NEAR(delphy_log_G(narrow_tree, make_evo(d.num_sites())),
              delphy_log_G(wide_tree, make_evo(wide.num_sites())), 1e-12);
}

//
// 2. Callable declarations
//

TEST(Human_observation_test, callable_declaration_makes_unlisted_sites_missing) {
  auto d = make_dataset();
  // Only sites 1-12 were examined; the reference bases of sites 13-20 are not observations.
  auto files = with_callable("1-12\n");
  auto result = load_fasta(d, config_for(d), files);

  EXPECT_EQ(result.audit.num_non_callable_sites, 8);
  EXPECT_TRUE(result.audit.callable_declared);
  for (const auto& tip : result.maple.tip_descs) {
    for (auto l = Site_index{12}; l != d.num_sites(); ++l) { EXPECT_TRUE(tip.missations.contains(l)); }
  }
  auto patterns = site_patterns(result);
  EXPECT_EQ(patterns[12], "A----");
  EXPECT_EQ(result.audit.num_never_observed_sites, 8);
  for (const auto& sample : result.audit.samples) {
    EXPECT_EQ(sample.num_callable + sample.num_missing, d.num_sites());
  }
}

TEST(Human_observation_test, callable_declaration_rejects_calls_it_does_not_cover) {
  auto d = make_dataset();
  auto files = with_callable("1-5\n");  // leaves out the call at site 7
  expect_error([&] { load_fasta(d, config_for(d), files); },
               "call a base that differs from the reference at a site that the callable declaration does not list");
}

TEST(Human_observation_test, the_mask_excludes_sites_without_being_a_contradiction) {
  auto d = make_dataset();
  auto masked = with_mask("7\n");
  auto result = load_fasta(d, config_for(d), masked);
  EXPECT_EQ(result.audit.num_masked_sites, 1);
  EXPECT_EQ(result.audit.samples[3].num_masked_overrides, 1);
  EXPECT_EQ(site_patterns(result)[6], "G----");
}

TEST(Human_observation_test, per_sample_callable_declaration_must_cover_every_sample) {
  auto d = make_dataset();
  auto files = with_sample_callable("S0\t1-19\nS1\t1-19\nS2\t1-19\n");
  expect_error([&] { load_fasta(d, config_for(d), files); },
               "no row in the per-sample callable-sites file: S3");

  auto extra = with_sample_callable("S0\t1-19\nS1\t1-19\nS2\t1-19\nS3\t1-19\nGHOST\t1-19\n");
  expect_error([&] { load_fasta(d, config_for(d), extra); }, "no sample in the input: GHOST");

  auto twice = with_sample_callable("S0\t1-19\nS0\t1-5\nS1\t1-19\nS2\t1-19\nS3\t1-19\n");
  expect_error([&] { load_fasta(d, config_for(d), twice); }, "already has a row on line");

  auto bad_range = with_sample_callable("S0\t1-99\nS1\t1-19\nS2\t1-19\nS3\t1-19\n");
  expect_error([&] { load_fasta(d, config_for(d), bad_range); }, "outside the valid sites 1-20");
}

TEST(Human_observation_test, per_sample_callable_declaration_reports_calls_that_failed_inside_it) {
  auto d = make_dataset();
  // S0 declares site 15 callable although the alignment has no call there: the warning says so.
  auto files = with_sample_callable("S0\t1-19\nS1\t1-19\nS2\t1-19\nS3\t1-19\n");
  auto result = load_fasta(d, config_for(d), files);
  EXPECT_THAT(absl::StrJoin(result.audit.warnings, " | "),
              testing::HasSubstr("data missing at 1 site declared callable"));
}

//
// 3. Variant table validation
//

TEST(Human_observation_test, variant_input_requires_a_callable_declaration) {
  auto d = make_dataset();
  expect_error([&] { load_variants(d, "sample_id\tsite\tbase\nS1\t3\tT\n", config_for(d), {}); },
               "Variant input requires a callable declaration");
}

TEST(Human_observation_test, variant_table_errors) {
  auto d = make_dataset();
  auto files = with_callable("1-20\n");
  auto load = [&](const std::string& body) {
    return [&, body] { load_variants(d, absl::StrCat("sample_id\tsite\tbase\n", body), config_for(d), files); };
  };

  expect_error(load("S1\t0\tT\n"), "is outside the reference sites 1-20");
  expect_error(load("S1\t21\tT\n"), "is outside the reference sites 1-20");
  expect_error(load("S1\tx\tT\n"), "is not an integer");
  expect_error(load("S1\t3\tT\nS1\t3\tT\n"), "already recorded on line");
  expect_error(load("S1\t3\tT\nS1\t3\tA\n"), "a site has one state per sample");
  expect_error(load("S1\t3\tG\n"), "the reference base");  // site 3 of the reference is G
  expect_error(load("S1\t3\tR\n"), "IUPAC ambiguity code");
  expect_error(load("S1\t3\tN\n"), "means 'not called'");
  expect_error(load("S1\t3\tACG\n"), "is not a single nucleotide");
  expect_error(load("S1\t3\t-\n"), "means 'not called'");
  expect_error(load("GHOST\t3\tT\n"), "no metadata row");
  expect_error([&] { load_variants(d, "sample\tsite\tbase\nS1\t3\tT\n", config_for(d), files); },
               "has no 'sample_id' column");
  expect_error([&] { load_variants(d, "", config_for(d), files); }, "Variant table is empty");
}

TEST(Human_observation_test, variants_outside_the_callable_declaration_are_rejected) {
  auto d = make_dataset();
  auto files = with_callable("1-5\n");
  expect_error([&] { load_variants(d, "sample_id\tsite\tbase\nS1\t7\tA\n", config_for(d), files); },
               "callable declaration does not list");
}

//
// 4. SNP-only completion
//

namespace {

// The dataset reduced to the sites that a variability filter keeps, and the counts of the sites it removes.
struct Snp_only {
  Dataset dataset;
  Seq_vector<Site_index> removed{0, 0, 0, 0};
};

auto to_snp_only(const Dataset& d) -> Snp_only {
  auto out = Snp_only{};
  out.dataset.ids = d.ids;
  out.dataset.sequences.assign(d.num_samples(), std::string{});
  for (auto l = Site_index{0}; l != d.num_sites(); ++l) {
    auto states = std::vector<char>{};
    for (auto i = 0; i != d.num_samples(); ++i) { states.push_back(d.sequences[i][l]); }
    auto constant_everywhere =
        std::ranges::none_of(states, [&](char c) { return c == 'N' || c != states.front(); });
    if (constant_everywhere) {
      // The filter removes it and reports only its state.  Sites with missing data are kept: the declaration
      // says nothing about per-sample missingness, so a filter that drops them loses information (see the
      // test below).
      ++out.removed[char_to_real_seq_letter(states.front())];
    } else {
      out.dataset.reference += d.reference[l];
      for (auto i = 0; i != d.num_samples(); ++i) { out.dataset.sequences[i] += states[i]; }
    }
  }
  return out;
}

}  // namespace

// The guard against variable-sites-only input is a heuristic: it fires when no site at all is observed as
// invariant.  An input that keeps a few non-polymorphic sites passes it, so the protection against
// ascertainment bias is the audit report and the explicit declarations, not this check.
TEST(Human_observation_test, input_without_any_invariant_site_is_rejected) {
  auto d = Dataset{};
  d.reference = "ACG";
  d.ids = {"S0", "S1", "S2", "S3"};
  d.sequences = {"ACG", "CCG", "AGG", "ACA"};
  expect_error([&] { load_fasta(d, config_for(d)); }, "No site is observed as invariant");

  auto with_one_invariant = d;
  with_one_invariant.reference = "ACGA";
  with_one_invariant.sequences = {"ACGA", "CCGA", "AGGA", "ACAA"};
  auto result = load_fasta(with_one_invariant, config_for(with_one_invariant));
  EXPECT_EQ(result.audit.num_invariant_observed_sites, 1);  // one invariant site is enough to pass the guard
}

TEST(Human_observation_test, snp_only_completion_restores_the_full_site_patterns) {
  auto d = make_dataset();
  auto snp = to_snp_only(d);
  auto config = config_for(snp.dataset);
  config.snp_only_constant_sites = {snp.removed};
  auto completed = load_fasta(snp.dataset, config);
  auto full = load_fasta(d, config_for(d));

  EXPECT_EQ(completed.audit.num_sites, full.audit.num_sites);
  EXPECT_EQ(completed.audit.num_input_sites, snp.dataset.num_sites());
  EXPECT_EQ(completed.audit.observation_mode, "full_alignment+completed_constant_sites");
  EXPECT_EQ(completed.audit.num_variable_sites,
            full.audit.num_variable_sites - full.audit.num_invariant_nonreference_sites);
  EXPECT_EQ(completed.audit.num_invariant_observed_sites,
            full.audit.num_invariant_observed_sites + full.audit.num_invariant_nonreference_sites);
  EXPECT_EQ(completed.audit.num_never_observed_sites, full.audit.num_never_observed_sites);
  EXPECT_THAT(absl::StrJoin(completed.audit.warnings, " | "),
              testing::HasSubstr("the input itself still contains 1 site observed as invariant"));

  // The same multiset of observed site patterns: the completed input differs from the full-length one only in
  // the order of the sites and in the reference base of the restored constant sites, neither of which any
  // site-exchangeable model can distinguish (see the reference-base test above).
  auto tip_states = [](const Human_input_result& r) {
    auto out = std::vector<std::string>{};
    for (auto& p : site_patterns(r)) { out.push_back(p.substr(1)); }
    std::ranges::sort(out);
    return out;
  };
  EXPECT_THAT(tip_states(completed), testing::Eq(tip_states(full)));
}

TEST(Human_observation_test, snp_only_completion_gives_the_full_length_likelihood) {
  auto d = make_dataset();
  auto snp = to_snp_only(d);
  auto config = config_for(snp.dataset);
  config.snp_only_constant_sites = {snp.removed};

  auto full_tree = build_tree(load_fasta(d, config_for(d)).maple);
  auto completed_tree = build_tree(load_fasta(snp.dataset, config).maple);
  ASSERT_EQ(completed_tree.num_sites(), full_tree.num_sites());
  auto evo = make_evo(full_tree.num_sites());

  EXPECT_NEAR(delphy_log_G(full_tree, evo), delphy_log_G(completed_tree, evo), 1e-12);
  EXPECT_NEAR(delphy_log_G(completed_tree, evo), independent_log_G(completed_tree, evo), 1e-9);
}

TEST(Human_observation_test, snp_only_completion_errors) {
  auto d = make_dataset();
  auto snp = to_snp_only(d);

  auto zero = config_for(snp.dataset);
  zero.snp_only_constant_sites = {Seq_vector<Site_index>{0, 0, 0, 0}};
  expect_error([&] { load_fasta(snp.dataset, zero); }, "declared constant-site counts are all zero");

  auto negative = config_for(snp.dataset);
  negative.snp_only_constant_sites = {Seq_vector<Site_index>{-1, 0, 0, 0}};
  expect_error([&] { load_fasta(snp.dataset, negative); }, "is negative");

  // Completing an input that still has invariant sites of its own is allowed (a real filter may keep some),
  // but the audit says so, because the declaration must not count those sites again.
  auto twice = config_for(d);
  twice.snp_only_constant_sites = {snp.removed};
  auto result = load_fasta(d, twice);
  EXPECT_THAT(absl::StrJoin(result.audit.warnings, " | "),
              testing::HasSubstr("the input itself still contains"));
}

TEST(Human_observation_test, constant_site_count_spec_parsing) {
  auto counts = parse_constant_site_counts("A=4000,C=3000,G=3100,T=2400");
  EXPECT_EQ(counts[Real_seq_letter::A], 4000);
  EXPECT_EQ(counts[Real_seq_letter::C], 3000);
  EXPECT_EQ(counts[Real_seq_letter::G], 3100);
  EXPECT_EQ(counts[Real_seq_letter::T], 2400);

  expect_error([] { parse_constant_site_counts("A=1,C=2,G=3"); }, "state T is missing");
  expect_error([] { parse_constant_site_counts("A=1,C=2,G=3,T=4,A=5"); }, "state A is given twice");
  expect_error([] { parse_constant_site_counts("A=1,C=2,G=3,T=-4"); }, "negative count");
  expect_error([] { parse_constant_site_counts("A=1,C=2,G=3,U=4"); }, "is not a nucleotide");
  expect_error([] { parse_constant_site_counts("A1,C=2,G=3,T=4"); }, "is not of the form");
}


// The reference base is a representation choice, not an observation: Delphy rebases the tree's reference to
// the root sequence, so a site where every sample shares a non-reference state has the same likelihood as the
// same observation written against that state.  This is what lets the SNP-only completion append constant
// sites whose reference base is the shared state.
TEST(Human_observation_test, the_reference_base_at_an_invariant_site_does_not_change_the_likelihood) {
  auto a = Dataset{};
  a.reference = "GA";
  a.ids = {"S0", "S1", "S2", "S3"};
  a.sequences = {"CA", "CA", "CA", "CA"};
  auto b = a;
  b.reference = "CA";
  auto result_a = load_fasta(a, config_for(a));
  EXPECT_EQ(result_a.audit.num_invariant_nonreference_sites, 1);
  EXPECT_EQ(result_a.audit.num_variable_sites, 1);
  EXPECT_EQ(load_fasta(b, config_for(b)).audit.num_invariant_nonreference_sites, 0);

  auto tree_a = build_tree(result_a.maple);
  auto tree_b = build_tree(load_fasta(b, config_for(b)).maple);
  auto evo = make_evo(2);
  EXPECT_EQ(calc_num_muts(tree_a), 0);
  EXPECT_NEAR(delphy_log_G(tree_a, evo), delphy_log_G(tree_b, evo), 1e-12);
  EXPECT_NEAR(delphy_log_G(tree_a, evo), independent_log_G(tree_a, evo), 1e-9);
}

//
// 5. The observation reaches the sampler: incremental and from-scratch posteriors agree
//

namespace {

// A larger synthetic dataset: `num_samples` sequences over `num_sites` sites derived from each other by a few
// substitutions, with a missing block in every third sample.  It has variable sites, invariant sites and
// per-sample missingness, and (unlike the hand-written dataset above) enough mutations for the sampler to
// move mutation histories around.
auto make_big_dataset(int num_samples, Site_index num_sites, uint32_t seed, Site_index max_mut_site = -1)
    -> Dataset {
  if (max_mut_site < 0) { max_mut_site = num_sites; }
  auto rng = std::mt19937{seed};
  auto bases = std::string{"ACGT"};
  auto d = Dataset{};
  d.reference.resize(num_sites);
  for (auto& c : d.reference) { c = bases[rng() % 4]; }
  for (auto i = 0; i != num_samples; ++i) {
    auto seq = (i == 0) ? d.reference : d.sequences[rng() % i];
    for (auto m = 0, num_mut = 1 + static_cast<int>(rng() % 4); m != num_mut; ++m) {
      auto site = static_cast<Site_index>(rng() % max_mut_site);
      auto old = seq[site];
      do { seq[site] = bases[rng() % 4]; } while (seq[site] == old);
    }
    if (i % 3 == 0) {
      auto start = static_cast<Site_index>(rng() % (max_mut_site / 2));
      for (auto l = start; l != start + num_sites / 10; ++l) { seq[l] = 'N'; }
    }
    d.ids.push_back(absl::StrFormat("S%03d", i));
    d.sequences.push_back(std::move(seq));
  }
  return d;
}

// `const_pop` replaces the default exponential population model by a constant one and switches the
// population moves off, which keeps the small trees of these tests away from the collapse attractor of the
// default coalescent grid.  The checkpoint test keeps the exponential model, because that is the one the
// .dphy parameters carry (a constant model is serialized as "no model").
auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, bool fixed_topology,
              bool const_pop = true) -> std::shared_ptr<delphy::Run> {
  auto run = std::make_shared<Run>(pool, std::mt19937{seed}, seed, tree);
  run->set_topology_moves_enabled(not fixed_topology);
  run->set_mu(3e-4);
  run->set_mu_prior_alpha(1.0);
  run->set_mu_prior_beta(0.0);
  run->set_num_parts(1);
  run->set_local_moves_per_global_move(200);
  if (const_pop) {
    run->set_pop_model(std::make_shared<Const_pop_model>(500.0));
    run->set_final_pop_size_move_enabled(false);
    run->set_pop_growth_rate_move_enabled(false);
  }
  return run;
}

auto initial_tree(Maple_file maple, uint32_t seed) -> Phylo_tree {
  auto prng = std::mt19937{seed};
  return build_rough_initial_tree_from_maple(std::move(maple), Init_method::mp_plus_timing, prng);
}

}  // namespace

class Human_observation_mcmc_test : public testing::TestWithParam<bool> {  // parameter: fixed topology
 protected:
  ctpl::thread_pool pool_{1};
};

TEST_P(Human_observation_mcmc_test, the_incremental_likelihood_matches_a_fresh_one_on_completed_input) {
  auto d = make_big_dataset(16, 120, 11);
  auto snp = to_snp_only(d);
  auto config = config_for(snp.dataset);
  config.max_missing_fraction = 1.0;
  config.snp_only_constant_sites = {snp.removed};
  auto result = load_fasta(snp.dataset, config);
  ASSERT_GT(result.audit.num_sites, result.audit.num_input_sites);

  auto run = make_run(pool_, initial_tree(std::move(result.maple), 4), 9, GetParam());
  for (auto batch = 0; batch != 8; ++batch) {
    run->do_mcmc_steps(1000);
    auto incremental = run->log_G();
    auto fresh = run->calc_cur_log_G();
    EXPECT_NEAR(incremental, fresh, 1e-6 * std::max(1.0, std::abs(fresh))) << "batch " << batch;
    EXPECT_NEAR(fresh, independent_log_G(run->tree(), run->evo()), 1e-6 * std::max(1.0, std::abs(fresh)))
        << "batch " << batch;
    assert_phylo_tree_integrity(run->tree(), true);
  }
  // The sampler really moved the mutation history, so the agreement above covers back mutations and sites hit
  // more than once, as well as accepted, rejected and rolled-back moves.
  EXPECT_GT(calc_num_muts(run->tree()), 0);
}

TEST_P(Human_observation_mcmc_test, the_incremental_likelihood_matches_a_fresh_one_with_declared_callable_sites) {
  auto d = make_big_dataset(16, 120, 23, 100);  // no call beyond site 100, so every declaration is consistent
  auto sample_callable = std::string{};
  for (auto i = 0; i != d.num_samples(); ++i) {
    // Every sample declares the first 100 sites, every second one also 101-120.
    absl::StrAppend(&sample_callable, d.ids[i], "\t", i % 2 == 0 ? "1-100" : "1-120", "\n");
  }
  auto config = config_for(d);
  config.max_missing_fraction = 1.0;
  auto result = load_fasta(d, config, with_sample_callable(sample_callable));
  ASSERT_GT(result.audit.num_never_observed_sites, 0);

  auto run = make_run(pool_, initial_tree(std::move(result.maple), 5), 13, GetParam());
  for (auto batch = 0; batch != 8; ++batch) {
    run->do_mcmc_steps(1000);
    auto fresh = run->calc_cur_log_G();
    EXPECT_NEAR(run->log_G(), fresh, 1e-6 * std::max(1.0, std::abs(fresh))) << "batch " << batch;
    EXPECT_NEAR(fresh, independent_log_G(run->tree(), run->evo()), 1e-6 * std::max(1.0, std::abs(fresh)))
        << "batch " << batch;
    assert_phylo_tree_integrity(run->tree(), true);
  }
}

// A checkpoint of a run on completed input restores the observation exactly --- the reference with its
// appended constant block, every tip's differences and missing regions, the site counts and the parameters ---
// and the resumed run keeps its bookkeeping consistent.  Bit-exact continuation of the whole trajectory is not
// asserted here: in this configuration (free exponential population model) the restored run diverges from the
// uninterrupted one after the checkpoint, and it does so in exactly the same way with plain full-length input,
// so it is a property of the existing checkpoint path and not of the observation modes under test.
TEST_P(Human_observation_mcmc_test, a_checkpoint_restores_completed_input_exactly) {
  auto d = make_big_dataset(12, 90, 31);
  auto snp = to_snp_only(d);
  auto config = config_for(snp.dataset);
  config.max_missing_fraction = 1.0;
  config.snp_only_constant_sites = {snp.removed};
  auto loaded = load_fasta(snp.dataset, config);
  auto num_sites = loaded.audit.num_sites;
  auto tip_descs = loaded.maple.tip_descs;
  auto input_reference = loaded.maple.ref_sequence;  // the tree's own reference is rebased to its root
  auto tree = initial_tree(std::move(loaded.maple), 6);

  auto run = make_run(pool_, tree, 17, GetParam(), false);
  run->set_coal_prior_fixed_t_step(50.0);
  run->do_mcmc_steps(2000);

  auto params = run_to_api_params(*run);
  auto resumed = std::make_shared<delphy::Run>(pool_, std::mt19937{1}, 1, run->tree(),
                                               run->coalescent_prior_t_step());
  resumed->set_num_parts(1);
  apply_api_params_to_run(params.data(), *resumed);

  EXPECT_EQ(resumed->step(), run->step());
  EXPECT_EQ(resumed->mu(), run->mu());
  EXPECT_EQ(resumed->tree().num_sites(), num_sites);
  EXPECT_EQ(resumed->log_G(), run->log_G());
  EXPECT_EQ(resumed->tree().ref_sequence, run->tree().ref_sequence);
  assert_phylo_tree_matches_tip_descs(resumed->tree(), input_reference, tip_descs, true);

  resumed->do_mcmc_steps(2000);
  auto fresh = resumed->calc_cur_log_G();
  EXPECT_NEAR(resumed->log_G(), fresh, 1e-6 * std::max(1.0, std::abs(fresh)));
  EXPECT_NEAR(fresh, independent_log_G(resumed->tree(), resumed->evo()), 1e-6 * std::max(1.0, std::abs(fresh)));
  assert_phylo_tree_integrity(resumed->tree(), true);
}

INSTANTIATE_TEST_SUITE_P(Topology, Human_observation_mcmc_test, testing::Values(true, false),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "fixed" : "free";
                         });

}  // namespace delphy
