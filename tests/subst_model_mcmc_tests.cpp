#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include "ctpl_stl.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "api.h"
#include "evo_subst_model.h"
#include "independent_seq_density.h"
#include "phylo_tree_calc.h"
#include "pop_model.h"
#include "run.h"
#include "site_partition.h"

namespace delphy {

namespace {

constexpr auto k_mu = 2e-4;    // per site per day
constexpr auto k_n0 = 1000.0;  // days

// A Kingman tree over `num_tips` tips at one date, with mutations drawn under an HKY-like process, so
// that the mutation-type counts M_ab are not symmetric and every rate matrix entry is exercised.
auto make_tree(int num_tips, Site_index num_sites, uint64_t seed) -> Phylo_tree {
  auto g = std::mt19937_64{seed};
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(num_sites, Real_seq_letter::A);
  auto active = std::vector<Node_index>{};
  for (auto i = 0; i != num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = 0.0;
    tree.at(i).name = absl::StrFormat("tip%d", i);
    active.push_back(i);
  }
  auto t = 0.0;
  auto next = num_tips;
  while (std::ssize(active) > 1) {
    auto k = std::ssize(active);
    auto rate = k * (k - 1) / (2 * k_n0);
    t -= std::exponential_distribution<double>{rate}(g);
    auto pick = std::uniform_int_distribution<int>{0, static_cast<int>(k) - 1};
    auto i = pick(g);
    auto j = i;
    while (j == i) { j = pick(g); }
    auto left = active[i];
    auto right = active[j];
    auto node = next++;
    tree.at(node).t = t;
    tree.at(node).children = {left, right};
    tree.at(left).parent = node;
    tree.at(right).parent = node;
    active.erase(active.begin() + std::max(i, j));
    active.erase(active.begin() + std::min(i, j));
    active.push_back(node);
  }
  tree.root = active.front();
  tree.at(tree.root).parent = k_no_node;
  // An inner node's age is unconstrained, which the tree integrity check in a debug build requires.
  for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
    if (tree.at(node).is_inner_node()) {
      tree.at(node).t_min = -std::numeric_limits<float>::max();
      tree.at(node).t_max = +std::numeric_limits<float>::max();
    }
  }

  // Mutations, drawn site by site down the tree with transition-biased rates.
  auto uniform = std::uniform_real_distribution<double>{0.0, 1.0};
  auto stack = std::vector<std::pair<Node_index, Real_sequence>>{};
  stack.push_back({tree.root, tree.ref_sequence});
  while (not stack.empty()) {
    auto [p, state] = std::move(stack.back());
    stack.pop_back();
    for (auto c : tree.at(p).children) {
      auto st = state;
      auto branch = tree.at(c).t - tree.at(p).t;
      auto times = std::vector<std::pair<double, Site_index>>{};
      for (auto site = Site_index{0}; site != num_sites; ++site) {
        auto events = std::poisson_distribution<int>{k_mu * branch}(g);
        for (auto e = 0; e != events; ++e) {
          times.push_back({tree.at(p).t + uniform(g) * branch, site});
        }
      }
      // The states are assigned in time order, so that every mutation's `from` is the state the site
      // is actually in at that moment and the tree stays mutation-consistent.
      std::ranges::sort(times);
      for (auto [tm, site] : times) {
        auto from = st[site];
        auto to = from;
        while (to == from) {
          to = real_seq_letter_from_index(std::uniform_int_distribution<int>{0, 3}(g));
        }
        tree.at(c).mutations.push_back(Mutation{from, site, to, tm});
        st[site] = to;
      }
      stack.push_back({c, std::move(st)});
    }
  }
  return tree;
}

auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, bool fixed_topology,
              bool const_pop = true) -> std::shared_ptr<Run> {
  auto run = std::make_shared<Run>(pool, std::mt19937{seed}, seed, tree);
  run->set_topology_moves_enabled(not fixed_topology);
  run->set_mu(k_mu);
  run->set_mu_prior_alpha(1.0);
  run->set_mu_prior_beta(0.0);
  run->set_local_moves_per_global_move(200);
  if (const_pop) {
    run->set_pop_model(std::make_shared<Const_pop_model>(k_n0));
    run->set_final_pop_size_move_enabled(false);
    run->set_pop_growth_rate_move_enabled(false);
  }
  return run;
}

auto delphy_log_G(const Phylo_tree& tree, const Global_evo_model& evo) -> double {
  return calc_log_root_prior(tree, evo) + calc_log_G_below_root(tree, evo);
}

// The run's evolutionary model with the rate matrix of `model` substituted into every partition,
// which is how an independent full recomputation is set up for a proposed model.
auto evo_with_model(const Global_evo_model& evo, const Subst_model& model,
                    const std::vector<double>& partition_rates) -> Global_evo_model {
  auto result = evo;
  auto site_evo = model.derive_site_evo_model();
  for (auto p = Partition_index{0}; p != result.num_partitions(); ++p) {
    result.partition_evo_model[p] = site_evo;
    result.partition_evo_model[p].mu = model.mu * partition_rates[p];
  }
  return result;
}

auto read_site_partition_config_text(const std::string& text) -> Site_partition_config {
  auto is = std::istringstream{text};
  return read_site_partition_config(is);
}

auto make_model(std::string_view spec, std::vector<double> rates, std::array<double, 4> pi,
                double mu = k_mu) -> Subst_model {
  auto model = parse_subst_model(spec);
  if (not rates.empty()) { model.class_rate = std::move(rates); }
  model.mu = mu;
  if (model.frequency_mode != Frequency_mode::equal) {
    model.pi_a[Real_seq_letter::A] = pi[0];
    model.pi_a[Real_seq_letter::C] = pi[1];
    model.pi_a[Real_seq_letter::G] = pi[2];
    model.pi_a[Real_seq_letter::T] = pi[3];
  }
  return model;
}

// log of the conditional density of the rate matrix given the tree, the mutational history and every
// other parameter, up to an additive constant that does not depend on the matrix: the exposure, the
// root-state counts and the mutation-type counts.  Written from the maintained statistics only, and
// validated against full recomputations by the first tests below.
auto log_conditional_of_matrix(const Run& run, const Subst_model& model) -> double {
  auto site_evo = model.derive_site_evo_model();
  auto result = 0.0;
  for (auto p = Partition_index{0}; p != run.evo().num_partitions(); ++p) {
    for (auto a : k_all_real_seq_letters) {
      result -= model.mu * run.site_partition_rates()[p] * site_evo.q_a(a)
          * run.Ttwiddle_beta_a()[p][a];
    }
  }
  auto counts = run.state_frequencies_of_ref_sequence();
  for (const auto& m : run.tree().at_root().mutations) {
    --counts[m.from];
    ++counts[m.to];
  }
  for (const auto& [site, from] :
       run.tree().at_root().missations.slow_elements(run.tree().ref_sequence)) {
    --counts[from];
  }
  for (auto a : k_all_real_seq_letters) {
    if (counts[a] > 0) { result += counts[a] * std::log(site_evo.pi_a[a]); }
  }
  for (auto a : k_all_real_seq_letters) {
    for (auto b : k_all_real_seq_letters) {
      if (a != b && run.num_muts_ab()[a][b] > 0) {
        result += run.num_muts_ab()[a][b] * std::log(site_evo.q_ab[a][b]);
      }
    }
  }
  return result;
}

// The log-normal prior of the free exchangeabilities of `model` under the run's prior settings.
auto log_exchangeability_prior(const Run& run, const Subst_model& model) -> double {
  auto sigma = run.subst_exchangeability_prior_sd();
  auto result = 0.0;
  for (auto k = 0; k != model.grouping.num_classes(); ++k) {
    if (k == model.grouping.reference_class()) { continue; }
    auto all_transitions = true;
    for (auto index : model.grouping.members_of(k)) {
      if (index != 1 && index != 4) { all_transitions = false; }
    }
    auto m = all_transitions ? run.subst_exchangeability_prior_log_mean_transition()
                             : run.subst_exchangeability_prior_log_mean_other();
    auto s = model.class_rate[k];
    result += -std::pow(std::log(s) - m, 2) / (2 * sigma * sigma)
        - 0.5 * std::log(2 * M_PI * sigma * sigma) - std::log(s);
  }
  return result;
}

}  // namespace

//
// 1. The incremental difference of log_G against independent full recomputations
//

TEST(Subst_model_mcmc_test, the_delta_of_an_exchangeability_change_matches_two_independent_computations) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(10, 300, 4711);
  auto run = make_run(pool, tree, 13, true);

  for (const auto* spec : {"HKY85", "TN93", "GTR", "012032", "TVM", "012210"}) {
    auto base = make_model(spec, {}, {0.31, 0.19, 0.17, 0.33});
    for (auto k = 0; k != base.grouping.num_classes(); ++k) {
      if (k == base.grouping.reference_class()) { continue; }
      base.class_rate[k] = 1.0 + 0.7 * k;
    }
    run->set_subst_model(base);
    auto old_log_G = run->log_G();
    EXPECT_NEAR(old_log_G, delphy_log_G(run->tree(), run->evo()), 1e-8 * std::abs(old_log_G));
    EXPECT_NEAR(old_log_G, independent_log_G(run->tree(), run->evo()), 1e-8 * std::abs(old_log_G));

    for (auto k = 0; k != base.grouping.num_classes(); ++k) {
      if (k == base.grouping.reference_class()) { continue; }
      for (auto scale : {0.8, 1.25, 3.0}) {
        auto proposed = base;
        proposed.class_rate[k] *= scale;
        auto delta = run->calc_delta_log_G_of_subst_model(base.derive_site_evo_model(),
                                                          proposed.derive_site_evo_model());
        auto proposed_evo = evo_with_model(run->evo(), proposed, run->site_partition_rates());
        auto fresh = delphy_log_G(run->tree(), proposed_evo) - old_log_G;
        auto independent = independent_log_G(run->tree(), proposed_evo) - old_log_G;
        EXPECT_NEAR(delta, fresh, 1e-7 * std::max(1.0, std::abs(fresh)))
            << spec << " class " << k << " scale " << scale;
        EXPECT_NEAR(delta, independent, 1e-7 * std::max(1.0, std::abs(independent)))
            << spec << " class " << k << " scale " << scale;
      }
    }
    // After the proposals the run must be exactly where it started: nothing was applied.
    EXPECT_EQ(run->log_G(), old_log_G) << spec;
    EXPECT_EQ(run->subst_model().class_rate, base.class_rate) << spec;
  }
}

TEST(Subst_model_mcmc_test, the_delta_of_a_frequency_change_matches_two_independent_computations) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(10, 300, 991);
  auto run = make_run(pool, tree, 17, true);

  for (const auto* spec : {"HKY85", "GTR", "TN93"}) {
    auto base = make_model(spec, {}, {0.31, 0.19, 0.17, 0.33});
    for (auto k = 0; k != base.grouping.num_classes(); ++k) {
      if (k != base.grouping.reference_class()) { base.class_rate[k] = 2.3; }
    }
    run->set_subst_model(base);
    auto old_log_G = run->log_G();
    for (auto pi : std::vector<std::array<double, 4>>{
             {0.25, 0.25, 0.25, 0.25}, {0.4, 0.1, 0.2, 0.3}, {0.1, 0.4, 0.4, 0.1}}) {
      auto proposed = base;
      proposed.pi_a[Real_seq_letter::A] = pi[0];
      proposed.pi_a[Real_seq_letter::C] = pi[1];
      proposed.pi_a[Real_seq_letter::G] = pi[2];
      proposed.pi_a[Real_seq_letter::T] = pi[3];
      auto delta = run->calc_delta_log_G_of_subst_model(base.derive_site_evo_model(),
                                                        proposed.derive_site_evo_model());
      auto proposed_evo = evo_with_model(run->evo(), proposed, run->site_partition_rates());
      auto fresh = delphy_log_G(run->tree(), proposed_evo) - old_log_G;
      auto independent = independent_log_G(run->tree(), proposed_evo) - old_log_G;
      EXPECT_NEAR(delta, fresh, 1e-7 * std::max(1.0, std::abs(fresh))) << spec;
      EXPECT_NEAR(delta, independent, 1e-7 * std::max(1.0, std::abs(independent))) << spec;
    }
    EXPECT_EQ(run->log_G(), old_log_G) << spec;
  }
}

TEST(Subst_model_mcmc_test, the_closed_form_conditional_tracks_log_G_up_to_a_constant) {
  // The conditional used by the exact-conditional tests below differs from log_G by a constant that
  // does not depend on the rate matrix; here that is checked against full recomputations.
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 200, 55);
  auto run = make_run(pool, tree, 23, true);
  auto base = make_model("GTR", {1.0, 2.0, 0.7, 1.3, 3.1, 1.0}, {0.31, 0.19, 0.17, 0.33});
  run->set_subst_model(base);
  auto offset = run->log_G() - log_conditional_of_matrix(*run, base);
  for (auto k = 0; k != 5; ++k) {
    auto proposed = base;
    proposed.class_rate[k] *= 1.9;
    proposed.pi_a[Real_seq_letter::A] += 0.03;
    proposed.pi_a[Real_seq_letter::T] -= 0.03;
    auto proposed_evo = evo_with_model(run->evo(), proposed, run->site_partition_rates());
    auto fresh = delphy_log_G(run->tree(), proposed_evo);
    EXPECT_NEAR(log_conditional_of_matrix(*run, proposed) + offset, fresh,
                1e-7 * std::abs(fresh)) << "class " << k;
  }
}

//
// 2. The stationary distribution of the moves
//

// A long chain of exchangeability moves alone must reproduce the exact conditional density of the
// free exchangeability, which is known in closed form once the tree, the history and the remaining
// parameters are frozen.
auto check_exact_conditional_of_class(Run& run, const Subst_model& start, int free_class,
                                      const char* label) -> void {
  run.set_subst_model(start);
  auto log_target = [&](double value) {
    if (not (value > 0.0)) { return -std::numeric_limits<double>::infinity(); }
    auto model = run.subst_model();
    model.class_rate[free_class] = value;
    return log_conditional_of_matrix(run, model) + log_exchangeability_prior(run, model);
  };

  // The exact distribution on a geometric grid of log(s), from which equal-probability bin edges are
  // read off.  The log-normal prior keeps the density integrable at both ends.
  constexpr auto k_grid = 60001;
  constexpr auto k_log_min = -8.0;
  constexpr auto k_log_max = 8.0;
  auto value_at = [&](int i) {
    return std::exp(k_log_min + (k_log_max - k_log_min) * i / (k_grid - 1));
  };
  auto density = std::vector<double>(k_grid, 0.0);
  auto max_log = -std::numeric_limits<double>::infinity();
  for (auto i = 0; i != k_grid; ++i) { max_log = std::max(max_log, log_target(value_at(i))); }
  for (auto i = 0; i != k_grid; ++i) {
    // The grid is uniform in log(s), so the measure carries a factor of s.
    density[i] = std::exp(log_target(value_at(i)) - max_log) * value_at(i);
  }
  auto cdf = std::vector<double>(k_grid, 0.0);
  for (auto i = 1; i != k_grid; ++i) { cdf[i] = cdf[i - 1] + density[i]; }
  for (auto& c : cdf) { c /= cdf.back(); }
  constexpr auto k_bins = 20;
  auto edges = std::vector<double>{};
  {
    auto i = 0;
    for (auto b = 1; b != k_bins; ++b) {
      auto target = static_cast<double>(b) / k_bins;
      while (i < k_grid - 1 && cdf[i] < target) { ++i; }
      edges.push_back(value_at(i));
    }
  }

  constexpr auto k_keep = 6000;
  constexpr auto k_thin = 60;
  auto counts = std::vector<int>(k_bins, 0);
  for (auto i = 0; i != k_keep * k_thin; ++i) {
    run.subst_exchangeability_move(free_class);
    if (i % k_thin == 0) {
      auto s = run.subst_model().class_rate[free_class];
      auto bin = static_cast<int>(std::ranges::upper_bound(edges, s) - edges.begin());
      ++counts[bin];
    }
  }
  auto expected = static_cast<double>(k_keep) / k_bins;
  auto chi2 = 0.0;
  for (auto c : counts) { chi2 += (c - expected) * (c - expected) / expected; }
  // 19 degrees of freedom: the 0.1% critical value is 43.8
  EXPECT_LT(chi2, 43.8) << label << " class " << free_class << ", counts "
                        << absl::StrJoin(counts, ",");
}

TEST(Subst_model_mcmc_test, the_exchangeability_move_samples_the_exact_conditional_for_HKY85) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(10, 400, 321);
  auto run = make_run(pool, tree, 29, true);
  auto model = make_model("HKY85", {}, {0.31, 0.19, 0.17, 0.33});
  check_exact_conditional_of_class(*run, model, model.grouping.class_of(1), "HKY85");
}

TEST(Subst_model_mcmc_test, the_exchangeability_move_samples_the_exact_conditional_for_TN93_and_GTR) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(10, 400, 322);
  auto run = make_run(pool, tree, 31, true);
  for (const auto* spec : {"TN93", "GTR"}) {
    auto model = make_model(spec, {}, {0.31, 0.19, 0.17, 0.33});
    for (auto k = 0; k != model.grouping.num_classes(); ++k) {
      if (k == model.grouping.reference_class()) { continue; }
      check_exact_conditional_of_class(*run, model, k, spec);
    }
  }
}

TEST(Subst_model_mcmc_test, the_exchangeability_move_samples_the_exact_conditional_under_a_changed_prior) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(10, 400, 323);
  auto run = make_run(pool, tree, 37, true);
  run->set_subst_exchangeability_prior(-0.5, 0.25, 0.6);
  auto model = make_model("TN93", {}, {0.31, 0.19, 0.17, 0.33});
  for (auto k = 0; k != model.grouping.num_classes(); ++k) {
    if (k == model.grouping.reference_class()) { continue; }
    check_exact_conditional_of_class(*run, model, k, "TN93 with a changed prior");
  }
}

TEST(Subst_model_mcmc_test, the_frequency_move_samples_the_exact_conditional) {
  // The conditional of pi is three-dimensional; it is computed exactly on a grid over the simplex and
  // compared with the chain's marginal for each base.
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 400, 324);
  auto run = make_run(pool, tree, 41, true);
  auto model = make_model("HKY85", {1.0, 2.5}, {0.31, 0.19, 0.17, 0.33});
  run->set_subst_model(model);

  constexpr auto k_cells = 80;  // per coordinate
  auto log_target = [&](std::array<double, 4> pi) {
    auto proposed = run->subst_model();
    for (auto i = 0; i != 4; ++i) { proposed.pi_a[real_seq_letter_from_index(i)] = pi[i]; }
    return log_conditional_of_matrix(*run, proposed);
  };
  // Marginal of each coordinate by summing the density over the simplex grid.
  auto marginal = std::array<std::vector<double>, 4>{};
  for (auto& m : marginal) { m.assign(k_cells, 0.0); }
  auto max_log = -std::numeric_limits<double>::infinity();
  auto cells = std::vector<std::pair<std::array<int, 3>, double>>{};
  for (auto ia = 0; ia != k_cells; ++ia) {
    for (auto ic = 0; ic != k_cells; ++ic) {
      for (auto ig = 0; ig != k_cells; ++ig) {
        auto pa = (ia + 0.5) / k_cells;
        auto pc = (ic + 0.5) / k_cells;
        auto pg = (ig + 0.5) / k_cells;
        auto pt = 1.0 - pa - pc - pg;
        if (pt <= 0.0) { continue; }
        auto value = log_target({pa, pc, pg, pt});
        max_log = std::max(max_log, value);
        cells.push_back({{ia, ic, ig}, value});
      }
    }
  }
  auto total = 0.0;
  for (const auto& [index, value] : cells) {
    auto weight = std::exp(value - max_log);
    total += weight;
    marginal[0][index[0]] += weight;
    marginal[1][index[1]] += weight;
    marginal[2][index[2]] += weight;
    auto it = static_cast<int>((1.0 - (index[0] + 0.5) / k_cells - (index[1] + 0.5) / k_cells
                                - (index[2] + 0.5) / k_cells) * k_cells);
    if (it >= 0 && it < k_cells) { marginal[3][it] += weight; }
  }
  for (auto& m : marginal) { for (auto& v : m) { v /= total; } }

  constexpr auto k_keep = 20000;
  constexpr auto k_thin = 40;
  auto counts = std::array<std::vector<int>, 4>{};
  for (auto& c : counts) { c.assign(k_cells, 0); }
  for (auto i = 0; i != k_keep * k_thin; ++i) {
    run->subst_frequencies_move();
    if (i % k_thin == 0) {
      for (auto a = 0; a != 4; ++a) {
        auto cell = static_cast<int>(run->subst_model().pi_a[real_seq_letter_from_index(a)] * k_cells);
        cell = std::clamp(cell, 0, k_cells - 1);
        ++counts[a][cell];
      }
    }
  }
  // The exact marginal mean of each coordinate against the chain's mean, to within the Monte Carlo
  // error of the chain (the autocorrelation of a delta-exchange chain is high, so the tolerance is
  // generous but still far below the width of the posterior).
  for (auto a = 0; a != 4; ++a) {
    auto exact_mean = 0.0;
    auto exact_var = 0.0;
    for (auto cell = 0; cell != k_cells; ++cell) {
      exact_mean += marginal[a][cell] * (cell + 0.5) / k_cells;
    }
    for (auto cell = 0; cell != k_cells; ++cell) {
      auto x = (cell + 0.5) / k_cells - exact_mean;
      exact_var += marginal[a][cell] * x * x;
    }
    auto chain_mean = 0.0;
    for (auto cell = 0; cell != k_cells; ++cell) {
      chain_mean += counts[a][cell] * (cell + 0.5) / k_cells;
    }
    chain_mean /= k_keep;
    EXPECT_NEAR(chain_mean, exact_mean, 0.25 * std::sqrt(exact_var))
        << "base " << real_seq_letter_from_index(a) << " exact sd " << std::sqrt(exact_var);
  }
}

//
// 3. Nesting, fixed parameters and compatibility
//

TEST(Subst_model_mcmc_test, a_constrained_GTR_has_the_same_posterior_as_the_model_it_nests) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(12, 500, 777);
  auto run = make_run(pool, tree, 43, true);
  auto pi = std::array<double, 4>{0.31, 0.19, 0.17, 0.33};
  auto kappa = 2.75;

  run->set_subst_model(make_model("HKY85", {1.0, kappa}, pi));
  auto hky_log_G = run->log_G();
  run->set_subst_model(make_model("GTR", {1.0, kappa, 1.0, 1.0, kappa, 1.0}, pi));
  EXPECT_NEAR(run->log_G(), hky_log_G, 1e-9 * std::abs(hky_log_G));
  run->set_subst_model(make_model("TN93", {1.0, kappa, kappa}, pi));
  EXPECT_NEAR(run->log_G(), hky_log_G, 1e-9 * std::abs(hky_log_G));

  run->set_subst_model(make_model("JC69", {}, pi));
  auto jc_log_G = run->log_G();
  run->set_subst_model(make_model("GTR+FQ", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, pi));
  EXPECT_NEAR(run->log_G(), jc_log_G, 1e-9 * std::abs(jc_log_G));
}

TEST(Subst_model_mcmc_test, a_model_without_free_parameters_keeps_them_fixed) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 200, 99);
  auto run = make_run(pool, tree, 47, true);
  run->set_subst_model(make_model("JC69", {}, {0.25, 0.25, 0.25, 0.25}));
  auto before = run->subst_model();
  for (auto i = 0; i != 500; ++i) {
    run->subst_exchangeability_moves();
    run->subst_frequencies_move();
  }
  EXPECT_EQ(run->subst_model().class_rate, before.class_rate);
  for (auto a : k_all_real_seq_letters) { EXPECT_EQ(run->subst_model().pi_a[a], before.pi_a[a]); }

  // SYM has free exchangeabilities but equal frequencies, which must not move.
  run->set_subst_model(make_model("SYM", {1.0, 2.0, 0.5, 1.5, 3.0, 1.0}, {0.25, 0.25, 0.25, 0.25}));
  for (auto i = 0; i != 500; ++i) { run->subst_frequencies_move(); }
  for (auto a : k_all_real_seq_letters) { EXPECT_EQ(run->subst_model().pi_a[a], 0.25); }
  auto moved = false;
  for (auto i = 0; i != 500; ++i) {
    run->subst_exchangeability_moves();
    if (run->subst_model().class_rate[1] != 2.0) { moved = true; }
  }
  EXPECT_TRUE(moved) << "the exchangeabilities of SYM must still be sampled";
  EXPECT_EQ(run->subst_model().class_rate[run->subst_model().grouping.reference_class()], 1.0);
}

TEST(Subst_model_mcmc_test, fixed_frequencies_stay_at_the_given_values) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 200, 101);
  auto run = make_run(pool, tree, 53, true);
  auto model = make_model("GTR", {1.0, 2.0, 0.5, 1.5, 3.0, 1.0}, {0.4, 0.1, 0.2, 0.3});
  model.frequency_mode = Frequency_mode::fixed;
  run->set_subst_model(model);
  for (auto i = 0; i != 500; ++i) { run->subst_frequencies_move(); }
  EXPECT_EQ(run->subst_model().pi_a[Real_seq_letter::A], 0.4);
  EXPECT_EQ(run->subst_model().pi_a[Real_seq_letter::C], 0.1);
}

TEST(Subst_model_mcmc_test, the_reference_exchangeability_never_moves) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 200, 103);
  auto run = make_run(pool, tree, 59, true);
  for (const auto* spec : {"GTR", "TN93", "TVM", "010232"}) {
    run->set_subst_model(make_model(spec, {}, {0.31, 0.19, 0.17, 0.33}));
    auto reference = run->subst_model().grouping.reference_class();
    for (auto i = 0; i != 2000; ++i) { run->subst_exchangeability_moves(); }
    EXPECT_EQ(run->subst_model().class_rate[reference], 1.0) << spec;
  }
}

class Subst_model_run_test : public testing::TestWithParam<bool> {  // parameter: fixed topology
 protected:
  auto fixed_topology() const -> bool { return GetParam(); }
};

TEST_P(Subst_model_run_test, incremental_and_fresh_posteriors_agree_over_long_runs) {
  auto pool = ctpl::thread_pool{1};
  for (const auto* spec : {"HKY85", "GTR", "JC69", "TN93", "012032", "SYM"}) {
    auto tree = make_tree(12, 400, 2024);
    auto run = make_run(pool, tree, 61, fixed_topology());
    run->set_subst_model(make_model(spec, {}, {0.31, 0.19, 0.17, 0.33}));
    for (auto batch = 0; batch != 4; ++batch) {
      run->do_mcmc_steps(2000);
      auto fresh = run->calc_cur_log_G();
      EXPECT_NEAR(run->log_G(), fresh, 1e-6 * std::max(1.0, std::abs(fresh)))
          << spec << " batch " << batch;
      auto fresh_priors = run->calc_cur_log_other_priors();
      EXPECT_NEAR(run->log_other_priors(), fresh_priors,
                  1e-6 * std::max(1.0, std::abs(fresh_priors))) << spec << " batch " << batch;
      auto independent = independent_log_G(run->tree(), run->evo());
      EXPECT_NEAR(run->log_G(), independent, 1e-6 * std::max(1.0, std::abs(independent)))
          << spec << " batch " << batch;
      run->subst_model().validate();
    }
  }
}

TEST_P(Subst_model_run_test, a_checkpoint_restores_the_model_and_its_parameters) {
  auto pool = ctpl::thread_pool{1};
  for (const auto* spec : {"GTR", "JC69", "TN93", "012032", "SYM", "HKY85"}) {
    auto tree = make_tree(10, 300, 606);
    auto run = make_run(pool, tree, 67, fixed_topology(), /*const_pop=*/false);
    auto model = make_model(spec, {}, {0.31, 0.19, 0.17, 0.33});
    for (auto k = 0; k != model.grouping.num_classes(); ++k) {
      if (k != model.grouping.reference_class()) { model.class_rate[k] = 1.0 + 0.5 * k; }
    }
    run->set_subst_model(model);
    run->set_subst_exchangeability_prior(0.5, -0.25, 0.9);
    run->do_mcmc_steps(1000);

    auto saved = run_to_api_params(*run);
    // The resumed run starts from the tree the checkpoint describes, which is the run's current tree.
    auto restored = std::make_shared<delphy::Run>(pool, std::mt19937{71}, 71, run->tree(),
                                          run->coalescent_prior_t_step());
    restored->set_num_parts(1);
    apply_api_params_to_run(saved.data(), *restored);

    EXPECT_EQ(restored->subst_model().grouping.code(), run->subst_model().grouping.code()) << spec;
    EXPECT_EQ(restored->subst_model().frequency_mode, run->subst_model().frequency_mode) << spec;
    EXPECT_EQ(restored->subst_model().class_rate, run->subst_model().class_rate) << spec;
    for (auto a : k_all_real_seq_letters) {
      EXPECT_EQ(restored->subst_model().pi_a[a], run->subst_model().pi_a[a]) << spec;
    }
    if (std::string_view{spec} != "HKY85") {
      // HKY85 with estimated frequencies is the default and is written the historical way, without
      // the prior fields; every other model carries them.
      EXPECT_EQ(restored->subst_exchangeability_prior_log_mean_transition(), 0.5) << spec;
      EXPECT_EQ(restored->subst_exchangeability_prior_log_mean_other(), -0.25) << spec;
      EXPECT_EQ(restored->subst_exchangeability_prior_sd(), 0.9) << spec;
    }
    EXPECT_NEAR(restored->log_G(), run->log_G(), 1e-9 * std::abs(run->log_G())) << spec;
    EXPECT_NEAR(restored->log_G(), restored->calc_cur_log_G(), 1e-6) << spec;
    restored->do_mcmc_steps(1000);
    EXPECT_NEAR(restored->log_G(), restored->calc_cur_log_G(), 1e-6) << spec;
  }
}

INSTANTIATE_TEST_SUITE_P(Topology, Subst_model_run_test, testing::Values(true, false),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "fixed" : "free"; });

TEST_P(Subst_model_run_test, the_model_combines_with_site_partitions_site_rates_and_tree_parts) {
  // Everything at once: a general matrix shared by two site partitions with their own relative
  // rates, continuous Gamma site rates, four tree parts (so the Subruns cut across the site
  // partitions) and repartitioning between global steps.  The incremental posterior must keep
  // agreeing with a fresh computation and with the independent per-site density, and the matrix must
  // stay the same in every partition while the rates stay normalized.
  auto pool = ctpl::thread_pool{4};
  auto tree = make_tree(16, 400, 31415);
  auto run = make_run(pool, tree, 97, fixed_topology());
  run->set_num_parts(4);
  run->set_alpha_move_enabled(true);
  run->set_alpha(0.9);

  auto config = read_site_partition_config_text(
      "version 1\n\npartition left\n  sites 1-200\nend\n\npartition right\n  sites 201-400\nend\n");
  run->set_site_partition(config);
  run->set_site_partition_rates({0.5, 1.5});

  auto model = make_model("GTR", {1.0, 2.5, 0.8, 1.2, 3.1, 1.0}, {0.31, 0.19, 0.17, 0.33});
  run->set_subst_model(model);

  for (auto batch = 0; batch != 4; ++batch) {
    run->do_mcmc_steps(3000);
    auto fresh = run->calc_cur_log_G();
    EXPECT_NEAR(run->log_G(), fresh, 1e-6 * std::max(1.0, std::abs(fresh))) << "batch " << batch;
    auto independent = independent_log_G(run->tree(), run->evo());
    EXPECT_NEAR(run->log_G(), independent, 1e-6 * std::max(1.0, std::abs(independent)))
        << "batch " << batch;
    auto priors = run->calc_cur_log_other_priors();
    EXPECT_NEAR(run->log_other_priors(), priors, 1e-6 * std::max(1.0, std::abs(priors)))
        << "batch " << batch;

    // One matrix, two rates: the partitions differ only in mu.
    const auto& a = run->evo().partition_evo_model[0];
    const auto& b = run->evo().partition_evo_model[1];
    for (auto x : k_all_real_seq_letters) {
      EXPECT_EQ(a.pi_a[x], b.pi_a[x]) << "batch " << batch;
      for (auto y : k_all_real_seq_letters) {
        EXPECT_EQ(a.q_ab[x][y], b.q_ab[x][y]) << "batch " << batch;
      }
    }
    for (auto p = Partition_index{0}; p != 2; ++p) {
      EXPECT_EQ(run->evo().partition_evo_model[p].mu,
                run->mu() * run->site_partition_rates()[p]) << "batch " << batch;
    }
    EXPECT_NEAR(weighted_mean_site_partition_rate(run->site_partition(),
                                                  run->site_partition_rates()), 1.0, 1e-9)
        << "batch " << batch;
    EXPECT_NO_THROW(run->subst_model().validate()) << "batch " << batch;
  }
}

TEST(Subst_model_mcmc_test, a_file_without_the_fields_restores_HKY85) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 200, 808);
  auto run = make_run(pool, tree, 73, true, /*const_pop=*/false);
  run->set_hky_kappa(3.5);
  run->set_hky_pi({0.3, 0.2, 0.2, 0.3});
  auto saved = run_to_api_params(*run);
  EXPECT_EQ(flatbuffers::GetSizePrefixedRoot<api::Params>(saved.data())->subst_model_code(), nullptr)
      << "the default model must not be written, so that existing files stay byte-identical";

  auto restored = std::make_shared<delphy::Run>(pool, std::mt19937{79}, 79, run->tree(),
                                        run->coalescent_prior_t_step());
  restored->set_num_parts(1);
  apply_api_params_to_run(saved.data(), *restored);
  EXPECT_TRUE(restored->is_hky());
  EXPECT_EQ(restored->subst_model().frequency_mode, Frequency_mode::estimated);
  EXPECT_EQ(restored->hky_kappa(), 3.5);
  EXPECT_EQ(restored->subst_model().pi_a[Real_seq_letter::A], 0.3);
}

TEST(Subst_model_mcmc_test, the_model_is_validated_when_it_is_set) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(6, 100, 909);
  auto run = make_run(pool, tree, 83, true);
  auto bad = make_model("GTR", {1.0, 2.0, 0.5, 1.5, 3.0, 2.0}, {0.31, 0.19, 0.17, 0.33});
  EXPECT_THROW(run->set_subst_model(bad), std::invalid_argument);  // G-T not held at one
  auto negative = make_model("GTR", {1.0, -2.0, 0.5, 1.5, 3.0, 1.0}, {0.31, 0.19, 0.17, 0.33});
  EXPECT_THROW(run->set_subst_model(negative), std::invalid_argument);
  auto unnormalized = make_model("GTR", {}, {0.5, 0.19, 0.17, 0.33});
  EXPECT_THROW(run->set_subst_model(unnormalized), std::invalid_argument);
  EXPECT_THROW(run->set_subst_exchangeability_prior(1.0, 0.0, 0.0), std::invalid_argument);
  EXPECT_THROW(run->set_subst_exchangeability_prior(1.0, 0.0, -1.0), std::invalid_argument);
}

TEST(Subst_model_mcmc_test, every_one_of_the_203_groupings_runs_and_keeps_its_posterior_consistent) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 200, 1234);
  auto run = make_run(pool, tree, 89, true);
  for (const auto& grouping : all_exchangeability_groupings()) {
    auto model = Subst_model{};
    model.mu = k_mu;
    model.grouping = grouping;
    model.frequency_mode = Frequency_mode::estimated;
    model.reset_class_rates();
    for (auto k = 0; k != grouping.num_classes(); ++k) {
      if (k != grouping.reference_class()) { model.class_rate[k] = 0.6 + 0.4 * k; }
    }
    model.pi_a = Seq_vector<double>{0.31, 0.19, 0.17, 0.33};
    run->set_subst_model(model);
    run->do_mcmc_steps(200);
    auto fresh = run->calc_cur_log_G();
    EXPECT_NEAR(run->log_G(), fresh, 1e-6 * std::max(1.0, std::abs(fresh))) << grouping.code();
    auto independent = independent_log_G(run->tree(), run->evo());
    EXPECT_NEAR(run->log_G(), independent, 1e-6 * std::max(1.0, std::abs(independent)))
        << grouping.code();
    auto fresh_priors = run->calc_cur_log_other_priors();
    EXPECT_NEAR(run->log_other_priors(), fresh_priors,
                1e-6 * std::max(1.0, std::abs(fresh_priors))) << grouping.code();
    EXPECT_NO_THROW(run->subst_model().validate()) << grouping.code();
  }
}

}  // namespace delphy
