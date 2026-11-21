#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <functional>
#include <iostream>
#include <random>
#include <sstream>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include "api.h"
#include "cmdline.h"
#include "dates.h"
#include "human_input.h"
#include "phylo_tree.h"

namespace delphy {

namespace {

// Test-only generated alignment: a root sequence plus samples derived from earlier samples with a few
// substitutions each, so that the data have tree structure and invariant sites.
struct Synthetic_alignment {
  std::string reference;
  std::vector<std::string> ids;
  std::vector<std::string> sequences;
};

auto make_synthetic_alignment(int num_samples, int num_sites, uint32_t seed) -> Synthetic_alignment {
  auto rng = std::mt19937{seed};
  auto result = Synthetic_alignment{};
  auto bases = std::string{"ACGT"};
  result.reference.resize(num_sites);
  for (auto& c : result.reference) { c = bases[rng() % 4]; }
  for (auto i = 0; i != num_samples; ++i) {
    auto seq = (i == 0) ? result.reference : result.sequences[rng() % i];
    for (auto m = 0, num_mut = 1 + static_cast<int>(rng() % 4); m != num_mut; ++m) {
      auto site = rng() % num_sites;
      auto old = seq[site];
      do { seq[site] = bases[rng() % 4]; } while (seq[site] == old);
    }
    result.ids.push_back(absl::StrFormat("S%04d", i));
    result.sequences.push_back(std::move(seq));
  }
  return result;
}

auto to_fasta(const Synthetic_alignment& a, bool include_reference = false) -> std::string {
  auto text = std::string{};
  if (include_reference) { absl::StrAppend(&text, ">REF\n", a.reference, "\n"); }
  for (auto i = 0; i != std::ssize(a.ids); ++i) { absl::StrAppend(&text, ">", a.ids[i], "\n", a.sequences[i], "\n"); }
  return text;
}

auto reference_fasta(const Synthetic_alignment& a) -> std::string { return absl::StrCat(">REF\n", a.reference, "\n"); }

auto custom_config(int num_sites) -> Human_input_config {
  auto config = Human_input_config{};
  config.reference_standard = Reference_standard::custom;
  config.expected_length = num_sites;
  return config;
}

auto parse_metadata(const std::string& text) -> Sample_metadata {
  auto is = std::istringstream{text};
  return read_sample_metadata(is);
}

auto load_fasta(const std::string& fasta, const std::string& reference, const Sample_metadata* metadata,
                const Human_input_config& config, const std::string* mask = nullptr) -> Human_input_result {
  auto fasta_is = std::istringstream{fasta};
  auto ref_is = std::istringstream{reference};
  auto mask_is = std::istringstream{mask != nullptr ? *mask : std::string{}};
  return load_human_fasta(fasta_is, config.reference_id.has_value() ? nullptr : &ref_is, metadata,
                          mask != nullptr ? &mask_is : nullptr, config);
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

constexpr auto k_sites = 600;

}  // namespace

//
// Scale: all samples with one uniform date must survive (the pathogen path would drop them for lack of a title date)
//

class Human_input_scale_test : public testing::TestWithParam<int> {};

TEST_P(Human_input_scale_test, uniform_date_keeps_every_sample_from_fasta) {
  auto n = GetParam();
  auto alignment = make_synthetic_alignment(n, 16569, 1234 + n);
  auto config = Human_input_config{};
  config.reference_standard = Reference_standard::rcrs;
  config.uniform_sample_date = "2015-06-01";

  auto result = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config);

  ASSERT_EQ(std::ssize(result.maple.tip_descs), n);
  EXPECT_EQ(std::ssize(result.maple.ref_sequence), 16569);
  EXPECT_EQ(result.audit.num_samples, n);
  EXPECT_EQ(result.audit.num_dates_from_uniform, n);
  EXPECT_TRUE(result.audit.no_time_signal);
  for (auto i = 0; i != n; ++i) {
    EXPECT_EQ(result.maple.tip_descs[i].name, alignment.ids[i]);
    EXPECT_EQ(result.maple.tip_descs[i].t_min, static_cast<float>(parse_iso_date("2015-06-01")));
    EXPECT_EQ(result.maple.tip_descs[i].t_max, static_cast<float>(parse_iso_date("2015-06-01")));
  }
  // Every column is accounted for, so the full length is preserved in the audit
  EXPECT_EQ(result.audit.num_variable_sites + result.audit.num_invariant_observed_sites +
                result.audit.num_never_observed_sites, 16569);
  EXPECT_GT(result.audit.num_invariant_observed_sites, 10000);
  EXPECT_THAT(result.audit.warnings, testing::Contains(testing::HasSubstr("NO TIME SIGNAL")));
}

INSTANTIATE_TEST_SUITE_P(Sizes, Human_input_scale_test, testing::Values(5, 100, 1000));

class Human_input_init_tree_test : public testing::TestWithParam<int> {};

TEST_P(Human_input_init_tree_test, uniform_date_input_reaches_initial_tree) {
  auto n = GetParam();
  auto alignment = make_synthetic_alignment(n, 16569, 99 + n);
  auto config = Human_input_config{};
  config.reference_standard = Reference_standard::rcrs;
  config.uniform_sample_date = "2015-06-01";
  auto result = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config);

  auto prng = std::mt19937{42};
  auto tree = build_rough_initial_tree_from_maple(std::move(result.maple), Init_method::mp_plus_timing, prng);
  assert_phylo_tree_integrity(tree, true);
  EXPECT_EQ((std::ssize(tree) + 1) / 2, n);
  EXPECT_EQ(tree.num_sites(), 16569);
}

INSTANTIATE_TEST_SUITE_P(Sizes, Human_input_init_tree_test, testing::Values(5, 100));

//
// Metadata and dates
//

TEST(Human_input_test, mixed_conventions_from_metadata) {
  auto alignment = make_synthetic_alignment(5, k_sites, 7);
  auto metadata = parse_metadata(
      "sample_id\tdate_convention\tdate_from\tdate_to\tbp_reference_year\tpopulation\n"
      "S0000\tiso\t2015-06-01\t\t\tA\n"
      "S0001\tcal_bp\t3200\t3100\t\tB\n"
      "S0002\tbce\t1500\t\t\tC\n"
      "S0003\tbp\t8000\t\t1950\tD\n"
      "S0004\tce\t1200\t1300\t\tE\n");
  auto result = load_fasta(to_fasta(alignment), reference_fasta(alignment), &metadata, custom_config(k_sites));

  ASSERT_EQ(std::ssize(result.maple.tip_descs), 5);
  auto& tips = result.maple.tip_descs;
  EXPECT_EQ(tips[0].t_min, static_cast<float>(parse_iso_date("2015-06-01")));
  EXPECT_EQ(tips[1].t_min, static_cast<float>(astronomical_year_to_day(1950 - 3200)));
  EXPECT_EQ(tips[1].t_max, static_cast<float>(astronomical_year_to_day(1950 - 3100)));
  EXPECT_EQ(tips[2].t_min, static_cast<float>(astronomical_year_to_day(1 - 1500)));
  EXPECT_EQ(tips[3].t_min, static_cast<float>(astronomical_year_to_day(1950 - 8000)));
  EXPECT_EQ(tips[4].t_max, static_cast<float>(astronomical_year_to_day(1300)));
  EXPECT_FALSE(result.audit.no_time_signal);
  EXPECT_EQ(result.audit.num_dates_from_metadata, 5);
  EXPECT_THAT(result.audit.ignored_metadata_columns, testing::ElementsAre("population"));
  EXPECT_GT(result.audit.midpoint_span_days, 365.0 * 8000);
}

TEST(Human_input_test, metadata_rows_may_be_in_any_order_and_uniform_fills_blanks) {
  auto alignment = make_synthetic_alignment(3, k_sites, 8);
  auto metadata = parse_metadata(
      "sample_id\tdate_convention\tdate_from\n"
      "S0002\tiso\t2010-01-01\n"
      "S0000\t\t\n"
      "S0001\t\t\n");
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015-01-01";
  auto result = load_fasta(to_fasta(alignment), reference_fasta(alignment), &metadata, config);
  auto& tips = result.maple.tip_descs;
  EXPECT_EQ(tips[0].name, "S0000");
  EXPECT_EQ(tips[0].t_min, static_cast<float>(parse_iso_date("2015-01-01")));
  EXPECT_EQ(tips[2].t_min, static_cast<float>(parse_iso_date("2010-01-01")));
  EXPECT_EQ(result.audit.num_dates_from_uniform, 2);
  EXPECT_FALSE(result.audit.no_time_signal);
  EXPECT_THAT(result.audit.warnings, testing::Contains(testing::HasSubstr("2 of 3 samples received the uniform")));
}

TEST(Human_input_test, missing_dates_are_an_error_not_a_drop) {
  auto alignment = make_synthetic_alignment(4, k_sites, 9);
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, custom_config(k_sites)); },
               "4 samples have no sampling date");
  auto metadata = parse_metadata("sample_id\tdate_convention\tdate_from\nS0000\tiso\t2015\n");
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), &metadata, custom_config(k_sites)); },
               "3 samples in the sequence file have no row in the metadata");
}

TEST(Human_input_test, metadata_row_without_sequence_is_an_error) {
  auto alignment = make_synthetic_alignment(3, k_sites, 10);
  auto metadata = parse_metadata(
      "sample_id\tdate_convention\tdate_from\nS0000\tiso\t2015\nS0001\tiso\t2015\nS0002\tiso\t2015\nGHOST\tiso\t2015\n");
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), &metadata, custom_config(k_sites)); },
               "1 metadata row has no sequence in the sequence file: GHOST");
}

TEST(Human_input_test, id_matching_is_exact) {
  auto alignment = make_synthetic_alignment(3, k_sites, 11);
  auto metadata = parse_metadata(
      "sample_id\tdate_convention\tdate_from\ns0000\tiso\t2015\nS0001\tiso\t2015\nS0002\tiso\t2015\n");
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), &metadata, custom_config(k_sites)); },
               "S0000");
}

TEST(Human_input_test, fasta_header_description_is_not_part_of_the_id) {
  auto alignment = make_synthetic_alignment(3, k_sites, 12);
  auto fasta = std::string{};
  for (auto i = 0; i != 3; ++i) {
    absl::StrAppend(&fasta, ">", alignment.ids[i], " Homo sapiens mitochondrion, complete genome\n",
                    alignment.sequences[i], "\n");
  }
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto result = load_fasta(fasta, reference_fasta(alignment), nullptr, config);
  EXPECT_EQ(result.maple.tip_descs[1].name, "S0001");
}

TEST(Human_input_test, date_errors_carry_the_metadata_line) {
  auto alignment = make_synthetic_alignment(3, k_sites, 13);
  auto metadata = parse_metadata(
      "sample_id\tdate_convention\tdate_from\tdate_to\tbp_reference_year\n"
      "S0000\tc14_uncal_bp\t3200\t\t\n"
      "S0001\tbp\t3200\t\t\n"
      "S0002\tbce\t0\t\t\n");
  try {
    load_fasta(to_fasta(alignment), reference_fasta(alignment), &metadata, custom_config(k_sites));
    FAIL();
  } catch (const Human_input_error& e) {
    auto msg = std::string{e.what()};
    EXPECT_THAT(msg, testing::HasSubstr("3 problems"));
    EXPECT_THAT(msg, testing::HasSubstr("Metadata line 2 (sample 'S0000'): Convention 'c14_uncal_bp'"));
    EXPECT_THAT(msg, testing::HasSubstr("Metadata line 3 (sample 'S0001'): Convention 'bp' requires"));
    EXPECT_THAT(msg, testing::HasSubstr("Metadata line 4 (sample 'S0002'): date_from 0 is not a valid BCE year"));
  }
}

TEST(Human_input_test, metadata_format_errors) {
  expect_error([] { parse_metadata(""); }, "empty");
  expect_error([] { parse_metadata("id\tdate_from\nS1\t2015\n"); }, "no 'sample_id' column");
  expect_error([] { parse_metadata("sample_id\tdate_convention\tdate_from\nS1 iso 2015\n"); }, "tab-separated");
  expect_error([] { parse_metadata("sample_id\tdate_convention\tdate_from\nS1\tiso\t2015\ns1\tiso\t2015\n"); },
               "duplicates");
  expect_error([] { parse_metadata("sample_id\tdate_convention\tdate_from\nS1\t\t2015\n"); },
               "date_convention is empty");
  expect_error([] { parse_metadata("sample_id\tdate_convention\tdate_from\nS1\tAD\t2015\n"); },
               "Unknown date convention 'AD'");
  expect_error([] { parse_metadata("sample_id\tdate_convention\tdate_from\tbp_reference_year\nS1\tbp\t5\tx\n"); },
               "bp_reference_year 'x' is not an integer");
  expect_error([] { parse_metadata("sample_id\tsample_id\nS1\tS1\n"); }, "duplicate column");
}

//
// Sequence checks
//

TEST(Human_input_test, duplicate_ids_and_case_collisions) {
  auto alignment = make_synthetic_alignment(3, k_sites, 14);
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto dup = to_fasta(alignment) + ">S0001\n" + alignment.sequences[0] + "\n";
  expect_error([&] { load_fasta(dup, reference_fasta(alignment), nullptr, config); }, "Duplicate sample ID 'S0001'");
  auto collide = to_fasta(alignment) + ">s0001\n" + alignment.sequences[0] + "\n";
  expect_error([&] { load_fasta(collide, reference_fasta(alignment), nullptr, config); }, "differ only in letter case");
}

TEST(Human_input_test, invalid_characters_and_ids) {
  auto alignment = make_synthetic_alignment(3, k_sites, 15);
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto bad = to_fasta(alignment);
  bad[bad.find('\n', bad.find('\n') + 1) - 3] = 'U';
  expect_error([&] { load_fasta(bad, reference_fasta(alignment), nullptr, config); },
               "invalid sequence character 'U' at alignment column");
  auto bad_id = ">S(1)\n" + alignment.sequences[0] + "\n>S2\n" + alignment.sequences[1] + "\n";
  expect_error([&] { load_fasta(bad_id, reference_fasta(alignment), nullptr, config); }, "which is not allowed");
}

TEST(Human_input_test, length_mismatches_are_rejected) {
  auto alignment = make_synthetic_alignment(3, k_sites, 16);
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto short_sample = to_fasta(alignment) + ">SHORT\n" + alignment.sequences[0].substr(0, 100) + "\n";
  expect_error([&] { load_fasta(short_sample, reference_fasta(alignment), nullptr, config); },
               "Sample 'SHORT' has 100 sites but the reference 'REF' has 600");

  auto wrong_expected = custom_config(k_sites + 1);
  wrong_expected.uniform_sample_date = "2015";
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, wrong_expected); },
               "has 600 sites but the declared standard (custom) has 601");
}

TEST(Human_input_test, rcrs_declaration_requires_full_length) {
  auto alignment = make_synthetic_alignment(3, k_sites, 17);
  auto config = Human_input_config{};
  config.reference_standard = Reference_standard::rcrs;
  config.uniform_sample_date = "2015";
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config); },
               "declared standard (rCRS) has 16569");
  config.expected_length = 600;
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config); },
               "Reference standard rCRS has 16569 sites");
  config.reference_standard = Reference_standard::custom;
  config.expected_length.reset();
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config); },
               "requires the expected number of sites");
}

TEST(Human_input_test, snp_only_alignment_is_rejected) {
  // Keep only the columns where some sample differs from the reference: no invariant column remains
  auto alignment = make_synthetic_alignment(30, k_sites, 18);
  auto keep = std::vector<int>{};
  for (auto l = 0; l != k_sites; ++l) {
    for (const auto& s : alignment.sequences) {
      if (s[l] != alignment.reference[l]) { keep.push_back(l); break; }
    }
  }
  ASSERT_LT(std::ssize(keep), k_sites);
  auto snp_only = Synthetic_alignment{};
  snp_only.ids = alignment.ids;
  for (auto l : keep) { snp_only.reference += alignment.reference[l]; }
  for (const auto& s : alignment.sequences) {
    auto t = std::string{};
    for (auto l : keep) { t += s[l]; }
    snp_only.sequences.push_back(t);
  }
  auto config = custom_config(static_cast<int>(std::ssize(keep)));
  config.uniform_sample_date = "2015";
  expect_error([&] { load_fasta(to_fasta(snp_only), reference_fasta(snp_only), nullptr, config); },
               "SNP-only or variable-sites-only alignment");

  // The same data declared as rCRS fail the length check as well
  config.reference_standard = Reference_standard::rcrs;
  config.expected_length.reset();
  expect_error([&] { load_fasta(to_fasta(snp_only), reference_fasta(snp_only), nullptr, config); },
               "declared standard (rCRS)");
}

TEST(Human_input_test, unknown_bases_are_never_written_as_reference_states) {
  auto alignment = make_synthetic_alignment(4, k_sites, 19);
  // Sample 1: N run at 101-110, R at 200, gap at 300-305
  auto& s = alignment.sequences[1];
  for (auto l = 100; l != 110; ++l) { s[l] = 'N'; }
  s[199] = 'R';
  for (auto l = 299; l != 305; ++l) { s[l] = '-'; }
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto result = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config);

  const auto& tip = result.maple.tip_descs[1];
  for (auto l : {100, 105, 109, 199, 299, 304}) {
    EXPECT_TRUE(tip.missations.contains(l)) << "site " << l + 1;
    EXPECT_TRUE(std::ranges::none_of(tip.seq_deltas, [&](const Seq_delta& d) { return d.site == l; }));
  }
  EXPECT_EQ(tip.missations.num_sites(), 10 + 1 + 6);
  const auto& sa = result.audit.samples[1];
  EXPECT_EQ(sa.num_unknown, 10);
  EXPECT_EQ(sa.num_iupac_ambiguous, 1);
  EXPECT_EQ(sa.num_gap, 6);
  EXPECT_EQ(sa.num_missing, 17);
  EXPECT_NEAR(sa.missing_fraction, 17.0 / k_sites, 1e-12);
  EXPECT_THAT(result.audit.warnings, testing::Contains(testing::HasSubstr("IUPAC ambiguity codes")));
  EXPECT_THAT(result.audit.warnings, testing::Contains(testing::HasSubstr("gap characters")));
}

TEST(Human_input_test, excess_missing_data_is_rejected_with_the_sample_named) {
  auto alignment = make_synthetic_alignment(4, k_sites, 20);
  for (auto l = 0; l != 100; ++l) { alignment.sequences[2][l] = 'N'; }
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config); },
               "S0002 (16.7%)");
  config.max_missing_fraction = 0.2;
  EXPECT_NO_THROW(load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config));
  config.max_missing_fraction = 0.0;
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config); },
               "max missing fraction must be in (0, 1]");
}

TEST(Human_input_test, never_observed_sites_are_distinguished_from_invariant_sites) {
  auto alignment = make_synthetic_alignment(4, k_sites, 21);
  for (auto& s : alignment.sequences) { s[49] = 'N'; }  // site 50 missing everywhere
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto result = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config);
  EXPECT_EQ(result.audit.num_never_observed_sites, 1);
  EXPECT_EQ(result.audit.num_variable_sites + result.audit.num_invariant_observed_sites + 1, k_sites);
}

TEST(Human_input_test, mask_makes_sites_missing_everywhere_and_is_audited) {
  auto alignment = make_synthetic_alignment(6, k_sites, 22);
  // Force a difference inside the masked range in sample 2
  auto& base = alignment.reference;
  auto changed = base[309] == 'A' ? 'C' : 'A';
  alignment.sequences[2][309] = changed;
  auto mask = std::string{"# poly-C\n303-315\n310\n400\n"};
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto result = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config, &mask);

  EXPECT_EQ(result.audit.num_masked_sites, 13 + 1);
  for (const auto& tip : result.maple.tip_descs) {
    EXPECT_TRUE(tip.missations.contains(302));
    EXPECT_TRUE(tip.missations.contains(314));
    EXPECT_FALSE(tip.missations.contains(315));
    EXPECT_TRUE(tip.missations.contains(399));
    EXPECT_TRUE(std::ranges::none_of(tip.seq_deltas, [](const Seq_delta& d) { return d.site >= 302 && d.site < 315; }));
  }
  EXPECT_EQ(result.audit.samples[2].num_masked_overrides >= 1, true);
  EXPECT_GE(result.audit.num_never_observed_sites, 14);
}

TEST(Human_input_test, mask_errors) {
  auto is = std::istringstream{"0-5\n"};
  expect_error([&] { read_site_mask(is, 100); }, "outside the valid sites 1-100");
  auto is2 = std::istringstream{"50-40\n"};
  expect_error([&] { read_site_mask(is2, 100); }, "reversed");
  auto is3 = std::istringstream{"abc\n"};
  expect_error([&] { read_site_mask(is3, 100); }, "is not a site");
  auto is4 = std::istringstream{"10-20\n15-30\n31\n5\n"};
  EXPECT_THAT(read_site_mask(is4, 100), testing::ElementsAre(Site_interval{4, 5}, Site_interval{9, 31}));
}

TEST(Human_input_test, duplicate_sequences_are_reported_not_removed) {
  auto alignment = make_synthetic_alignment(5, k_sites, 23);
  alignment.sequences[4] = alignment.sequences[1];
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto result = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config);
  EXPECT_EQ(std::ssize(result.maple.tip_descs), 5);
  ASSERT_EQ(std::ssize(result.audit.duplicate_groups), 1);
  EXPECT_THAT(result.audit.duplicate_groups[0], testing::ElementsAre("S0001", "S0004"));
  EXPECT_EQ(result.audit.samples[1].duplicate_group, 0);
  EXPECT_EQ(result.audit.samples[4].duplicate_group, 0);
  EXPECT_EQ(result.audit.samples[0].duplicate_group, -1);
}

TEST(Human_input_test, reference_inside_input_fasta) {
  auto alignment = make_synthetic_alignment(4, k_sites, 24);
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  config.reference_id = "REF";
  auto result = load_fasta(to_fasta(alignment, true), "", nullptr, config);
  EXPECT_EQ(std::ssize(result.maple.tip_descs), 4);
  EXPECT_EQ(result.audit.reference_id, "REF");

  config.reference_id = "NOPE";
  expect_error([&] { load_fasta(to_fasta(alignment, true), "", nullptr, config); }, "'NOPE' not found");

  // Both ways of giving a reference is ambiguous
  config.reference_id = "REF";
  auto fasta_is = std::istringstream{to_fasta(alignment, true)};
  auto ref_is = std::istringstream{reference_fasta(alignment)};
  expect_error([&] { load_human_fasta(fasta_is, &ref_is, nullptr, nullptr, config); }, "not both");
}

TEST(Human_input_test, reference_must_be_unambiguous) {
  auto alignment = make_synthetic_alignment(3, k_sites, 25);
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  alignment.reference[10] = 'N';
  expect_error([&] { load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config); },
               "cannot be recorded as reference states");
}

TEST(Human_input_test, reference_checksum_is_stable) {
  auto alignment = make_synthetic_alignment(3, k_sites, 26);
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto a = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config);
  auto b = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config);
  EXPECT_EQ(a.audit.reference_checksum_fnv1a64, b.audit.reference_checksum_fnv1a64);
  EXPECT_EQ(a.audit.reference_checksum_fnv1a64.size(), 16u);
  alignment.reference[0] = alignment.reference[0] == 'A' ? 'C' : 'A';
  for (auto& s : alignment.sequences) { s[0] = alignment.reference[0]; }
  auto c = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config);
  EXPECT_NE(a.audit.reference_checksum_fnv1a64, c.audit.reference_checksum_fnv1a64);
}

//
// MAPLE
//

TEST(Human_input_test, maple_matches_fasta_representation) {
  auto alignment = make_synthetic_alignment(6, k_sites, 27);
  alignment.sequences[2][50] = 'N';
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto from_fasta = load_fasta(to_fasta(alignment), reference_fasta(alignment), nullptr, config);

  auto maple = absl::StrCat(">REF\n", alignment.reference, "\n");
  for (auto i = 0; i != 6; ++i) {
    absl::StrAppend(&maple, ">", alignment.ids[i], "\n");
    for (const auto& d : from_fasta.maple.tip_descs[i].seq_deltas) {
      absl::StrAppendFormat(&maple, "%c %d\n", to_char(d.to), d.site + 1);
    }
    for (const auto& [a, b] : from_fasta.maple.tip_descs[i].missations.intervals) {
      absl::StrAppendFormat(&maple, "N %d %d\n", a + 1, b - a);
    }
  }
  auto maple_is = std::istringstream{maple};
  auto from_maple = load_human_maple(maple_is, nullptr, nullptr, config);

  ASSERT_EQ(std::ssize(from_maple.maple.tip_descs), 6);
  EXPECT_EQ(from_maple.maple.ref_sequence, from_fasta.maple.ref_sequence);
  for (auto i = 0; i != 6; ++i) {
    EXPECT_EQ(from_maple.maple.tip_descs[i].name, from_fasta.maple.tip_descs[i].name);
    EXPECT_EQ(from_maple.maple.tip_descs[i].seq_deltas.size(), from_fasta.maple.tip_descs[i].seq_deltas.size());
    EXPECT_EQ(from_maple.maple.tip_descs[i].missations, from_fasta.maple.tip_descs[i].missations);
  }
  EXPECT_EQ(from_maple.audit.samples[0].num_gap, -1);  // not available for MAPLE
  EXPECT_EQ(from_maple.audit.num_variable_sites, from_fasta.audit.num_variable_sites);
}

TEST(Human_input_test, maple_does_not_need_dates_in_ids_and_keeps_all_tips) {
  auto reference = make_synthetic_alignment(1, k_sites, 28).reference;
  auto maple = absl::StrCat(">REF\n", reference, "\n");
  for (auto i = 0; i != 50; ++i) {
    absl::StrAppendFormat(&maple, ">tip%d\n%c %d\n", i, reference[i] == 'A' ? 'C' : 'A', i + 1);
  }
  auto maple_is = std::istringstream{maple};
  auto config = custom_config(k_sites);
  config.uniform_sample_date = "2015";
  auto result = load_human_maple(maple_is, nullptr, nullptr, config);
  EXPECT_EQ(std::ssize(result.maple.tip_descs), 50);
  EXPECT_EQ(result.audit.num_variable_sites, 50);
}

TEST(Human_input_test, maple_errors) {
  auto reference = make_synthetic_alignment(1, 100, 29).reference;
  auto config = custom_config(100);
  config.uniform_sample_date = "2015";
  auto run = [&](const std::string& body) {
    auto is = std::istringstream{absl::StrCat(">REF\n", reference, "\n", body)};
    return load_human_maple(is, nullptr, nullptr, config);
  };
  auto other = [&](int site) { return reference[site - 1] == 'A' ? 'C' : 'A'; };

  expect_error([&] { run(absl::StrFormat(">a\n%c 500\n>b\n%c 3\n", other(1), other(3))); }, "bad or out-of-range site");
  expect_error([&] { run(absl::StrFormat(">a\n%c 3\n>b\n%c 4\n", reference[2], other(4))); }, "equals the reference state");
  expect_error([&] { run(absl::StrFormat(">a\n%c 3\n%c 3\n>b\n%c 4\n", other(3), other(3), other(4))); },
               "listed more than once");
  expect_error([&] { run(absl::StrFormat(">a\n%c 3\nN 2 5\n>b\n%c 4\n", other(3), other(4))); },
               "both mutated and missing");
  expect_error([&] { run(">a\nN 98 10\n>b\nN 1\n"); }, "outside the 100-site reference");
  expect_error([&] { run(">a\nU 3\n>b\nN 1\n"); }, "cannot parse state letter");
  expect_error([&] {
    auto is = std::istringstream{absl::StrCat(">REF\nAC", "R", reference.substr(3), "\n>a\nN 1\n>b\nN 2\n")};
    load_human_maple(is, nullptr, nullptr, config);
  }, "reference must be fully resolved");
  expect_error([&] {
    auto is = std::istringstream{"ACGT\n>a\nN 1\n"};
    load_human_maple(is, nullptr, nullptr, config);
  }, "first line must be '>'");
  config.reference_id = "REF";
  expect_error([&] { run(">a\nN 1\n>b\nN 2\n"); }, "applies to FASTA input only");
}

//
// Pathogen path is unchanged
//

TEST(Human_input_test, pathogen_path_still_drops_undated_sequences) {
  auto fasta = std::vector<Fasta_entry>{
    {"tipA|2024-01-01", {Seq_letters::A, Seq_letters::C, Seq_letters::G}},
    {"tipB", {Seq_letters::A, Seq_letters::C, Seq_letters::T}},
  };
  auto dropped = std::vector<std::string>{};
  auto maple = fasta_to_maple(fasta, [](int, int) {}, [&](const std::string& id, const Sequence_warning&) {
    dropped.push_back(id);
  });
  EXPECT_EQ(std::ssize(maple.tip_descs), 1);
  EXPECT_THAT(dropped, testing::ElementsAre("tipB"));
}

//
// Audit report
//

TEST(Human_input_test, audit_json_is_well_formed_and_deterministic) {
  auto alignment = make_synthetic_alignment(4, k_sites, 30);
  auto metadata = parse_metadata(
      "sample_id\tdate_convention\tdate_from\tdate_to\tbp_reference_year\n"
      "S0000\tcal_bp\t3200\t3100\t\nS0001\tiso\t2015-06-01\t\t\nS0002\tbce\t1500\t\t\nS0003\tbp\t800\t\t1950\n");
  auto config = custom_config(k_sites);
  auto r1 = load_fasta(to_fasta(alignment), reference_fasta(alignment), &metadata, config);
  auto r2 = load_fasta(to_fasta(alignment), reference_fasta(alignment), &metadata, config);
  auto s1 = std::ostringstream{};
  auto s2 = std::ostringstream{};
  write_human_audit_json(r1.audit, s1);
  write_human_audit_json(r2.audit, s2);
  EXPECT_EQ(s1.str(), s2.str());
  EXPECT_THAT(s1.str(), testing::HasSubstr("\"report_version\": 3"));
  EXPECT_THAT(s1.str(), testing::HasSubstr("\"observation_mode\": \"full_alignment\""));
  EXPECT_THAT(s1.str(), testing::HasSubstr("\"no_time_signal\": false"));
  EXPECT_THAT(s1.str(), testing::HasSubstr("\"date\": \"cal_bp 3200..3100\""));
  EXPECT_THAT(s1.str(), testing::HasSubstr("\"date\": \"bp 800 (baseline 1950 CE)\""));
}

//
// Deep time: float32 limits in tip bounds and in .dphy round trips
//

TEST(Human_input_test, ancient_tip_bounds_survive_float32_storage_within_one_day) {
  for (auto years_bp : {1000, 8000, 12000, 30000, 45000, 90000}) {
    SCOPED_TRACE(years_bp);
    auto b = convert_sample_date({.convention = Date_convention::cal_bp, .from = std::to_string(years_bp),
                                  .to = std::to_string(years_bp - 100), .bp_reference_year = {}});
    auto lo = static_cast<float>(b.t_min);
    auto hi = static_cast<float>(b.t_max);
    EXPECT_LE(lo, hi);
    EXPECT_LE(std::abs(static_cast<double>(lo) - b.t_min), 1.0);
    EXPECT_LE(std::abs(static_cast<double>(hi) - b.t_max), 1.0);
  }
}

namespace {

struct Roundtrip_measurement {
  double max_abs_t = 0.0;
  double max_node_error = 0.0;
  int order_violations = 0;      // child before parent, or mutation outside its branch, after the round trip
  int tolerance_violations = 0;  // same, by more than the 1e-2 day slack of assert_phylo_tree_integrity
  bool tip_bounds_preserved = true;
};

// Builds an initial tree from ancient-dated synthetic data and sends it through the .dphy (api) representation,
// which stores node and mutation times as float32.
auto measure_dphy_roundtrip(int num_samples, const std::function<std::string(int)>& date_cells) -> Roundtrip_measurement {
  auto alignment = make_synthetic_alignment(num_samples, k_sites, 31);
  auto metadata_text = std::string{"sample_id\tdate_convention\tdate_from\tdate_to\n"};
  for (auto i = 0; i != num_samples; ++i) {
    absl::StrAppend(&metadata_text, alignment.ids[i], "\t", date_cells(i), "\n");
  }
  auto metadata = parse_metadata(metadata_text);
  auto result = load_fasta(to_fasta(alignment), reference_fasta(alignment), &metadata, custom_config(k_sites));
  auto prng = std::mt19937{5};
  auto tree = build_rough_initial_tree_from_maple(std::move(result.maple), Init_method::mp_plus_timing, prng);
  assert_phylo_tree_integrity(tree, true);

  auto tree_fb = phylo_tree_to_api_tree(tree);
  auto info_fb = phylo_tree_to_api_tree_info(tree);
  auto reloaded = api_tree_and_tree_info_to_phylo_tree(tree_fb.data(), info_fb.data());
  EXPECT_EQ(std::ssize(reloaded), std::ssize(tree));

  auto m = Roundtrip_measurement{};
  for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
    m.max_node_error = std::max(m.max_node_error, std::abs(reloaded.at(node).t - tree.at(node).t));
    m.max_abs_t = std::max(m.max_abs_t, std::abs(tree.at(node).t));
    if (tree.at(node).is_tip() &&
        (reloaded.at(node).t_min != tree.at(node).t_min || reloaded.at(node).t_max != tree.at(node).t_max)) {
      m.tip_bounds_preserved = false;
    }

    const auto& n = reloaded.at(node);
    auto t_prev = n.parent != k_no_node ? reloaded.at(n.parent).t : -std::numeric_limits<double>::infinity();
    if (n.parent != k_no_node) {
      if (n.t < t_prev) { ++m.order_violations; }
      if (n.t < t_prev - 1e-2) { ++m.tolerance_violations; }
    }
    for (const auto& mut : n.mutations) {
      if (mut.t < t_prev) { ++m.order_violations; }
      if (mut.t < t_prev - 1e-2) { ++m.tolerance_violations; }
      t_prev = mut.t;
    }
    if (n.parent != k_no_node && not n.mutations.empty() && n.t < t_prev) { ++m.order_violations; }
  }
  return m;
}

}  // namespace

TEST(Human_input_test, dphy_api_roundtrip_float32_error_with_spread_ancient_dates) {
  auto m = measure_dphy_roundtrip(30, [](int i) { return absl::StrFormat("cal_bp\t%d\t%d", 9000 + 10 * i, 8990 + 10 * i); });
  std::cout << absl::StreamFormat(
      "[float32] spread dates: max |t| = %.0f d, max node time error = %.4f d, order violations = %d, "
      "beyond-1e-2 violations = %d\n", m.max_abs_t, m.max_node_error, m.order_violations, m.tolerance_violations);
  // float32 spacing at |t| ~ 3.3e6 days is 0.25 day, so the rounding error is at most 0.125 day
  EXPECT_GT(m.max_abs_t, 3.0e6);
  EXPECT_LT(m.max_abs_t, 4.0e6);
  EXPECT_LE(m.max_node_error, 0.125 + 1e-12);
  EXPECT_TRUE(m.tip_bounds_preserved);
  EXPECT_EQ(m.order_violations, 0);
}

TEST(Human_input_test, dphy_api_roundtrip_float32_error_with_contemporaneous_ancient_samples) {
  // Short branches (the initializer uses a minimum branch length of 0.1 day) are below the float32 resolution
  auto m = measure_dphy_roundtrip(60, [](int) { return std::string{"cal_bp\t9000\t"}; });
  std::cout << absl::StreamFormat(
      "[float32] contemporaneous dates: max |t| = %.0f d, max node time error = %.4f d, "
      "order violations = %d, beyond-1e-2 violations = %d\n",
      m.max_abs_t, m.max_node_error, m.order_violations, m.tolerance_violations);
  EXPECT_LE(m.max_node_error, 0.125 + 1e-12);
  EXPECT_TRUE(m.tip_bounds_preserved);
}

}  // namespace delphy
