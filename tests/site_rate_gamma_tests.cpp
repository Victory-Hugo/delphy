#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "ctpl_stl.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "evo_subst_model.h"
#include "independent_seq_density.h"
#include "phylo_tree_calc.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

constexpr auto k_mu = 2e-4;    // per site per day
constexpr auto k_n0 = 1000.0;  // days

// Poisson draw that allows a zero mean, which std::poisson_distribution does not
auto draw_poisson(double mean, std::mt19937_64& g) -> int {
  if (not std::isfinite(mean) or mean < 0.0) {
    throw std::invalid_argument{"draw_poisson: mean must be finite and non-negative"};
  }
  if (mean == 0.0) { return 0; }
  return std::poisson_distribution<int>{mean}(g);
}

// A Kingman tree over `num_tips` tips at one date whose site rates differ: site l gets `rate(l)`
// times the clock rate, so the mutation counts per site are genuinely heterogeneous.
auto make_tree(int num_tips, Site_index num_sites, uint64_t seed,
               const std::function<double(Site_index)>& rate = [](Site_index) { return 1.0; },
               std::vector<Site_interval> missing_in_first_tip = {}) -> Phylo_tree {
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
    t -= std::exponential_distribution<double>{k * (k - 1) / (2 * k_n0)}(g);
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
        auto events = draw_poisson(k_mu * rate(site) * branch, g);
        for (auto e = 0; e != events; ++e) {
          times.push_back({tree.at(p).t + uniform(g) * branch, site});
        }
      }
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

  for (const auto& iv : missing_in_first_tip) {
    for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
      std::erase_if(tree.at(node).mutations,
                    [&](const Mutation& m) { return m.site >= iv.first && m.site < iv.second; });
    }
    tree.at(0).missations.intervals.insert(iv);
  }
  return tree;
}

auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, bool fixed_topology)
    -> std::shared_ptr<delphy::Run> {
  auto run = std::make_shared<delphy::Run>(pool, std::mt19937{seed}, seed, tree);
  run->set_topology_moves_enabled(not fixed_topology);
  run->set_mu(k_mu);
  run->set_mu_prior_alpha(1.0);
  run->set_mu_prior_beta(0.0);
  run->set_local_moves_per_global_move(200);
  run->set_pop_model(std::make_shared<Const_pop_model>(k_n0));
  run->set_final_pop_size_move_enabled(false);
  run->set_pop_growth_rate_move_enabled(false);
  run->set_alpha_move_enabled(true);
  return run;
}

// An independent evaluation of the density of alpha with every nu_l integrated out, up to a constant
// that does not depend on alpha.  For each site,
//   int_0^inf nu^{M_l} e^{-nu mu_l Ttwiddle_l} * alpha^alpha / Gamma(alpha) nu^{alpha-1} e^{-alpha nu} dnu
//     = alpha^alpha / Gamma(alpha) * Gamma(M_l + alpha) / (mu_l Ttwiddle_l + alpha)^{M_l + alpha},
// written here term by term with no sharing of code with calc_log_p_alpha.
auto independent_log_p_alpha(const Run& run, double alpha) -> double {
  auto Ttwiddle_l = calc_Ttwiddle_l(run.tree(), run.evo());
  auto M_l = calc_num_muts_l(run.tree());
  auto total = 0.0;
  for (auto l = Site_index{0}; l != run.tree().num_sites(); ++l) {
    auto rate = run.evo().mu_l(l) * Ttwiddle_l[l] + alpha;
    total += alpha * std::log(alpha) - std::lgamma(alpha)
        + std::lgamma(M_l[l] + alpha) - (M_l[l] + alpha) * std::log(rate);
  }
  return total;
}

// Numerical quadrature of the same integral, as a second independent route.
auto quadrature_log_p_alpha(const Run& run, double alpha) -> double {
  auto Ttwiddle_l = calc_Ttwiddle_l(run.tree(), run.evo());
  auto M_l = calc_num_muts_l(run.tree());
  auto total = 0.0;
  for (auto l = Site_index{0}; l != run.tree().num_sites(); ++l) {
    auto rate = run.evo().mu_l(l) * Ttwiddle_l[l] + alpha;
    auto shape = M_l[l] + alpha;
    // Substitute nu = exp(u) and integrate over u on a wide grid; the integrand is unimodal.
    auto mode = std::log(std::max(shape - 1.0, 1e-8) / rate);
    auto half_width = 40.0 / std::sqrt(std::max(shape, 0.05));
    constexpr auto k_steps = 20000;
    auto lo = mode - half_width;
    auto hi = mode + half_width;
    auto step = (hi - lo) / k_steps;
    auto max_log = -std::numeric_limits<double>::infinity();
    auto values = std::vector<double>(k_steps + 1);
    for (auto i = 0; i <= k_steps; ++i) {
      auto u = lo + i * step;
      values[i] = shape * u - rate * std::exp(u);
      max_log = std::max(max_log, values[i]);
    }
    auto sum = 0.0;
    for (auto i = 0; i <= k_steps; ++i) {
      auto w = (i == 0 || i == k_steps) ? 0.5 : 1.0;
      sum += w * std::exp(values[i] - max_log);
    }
    total += alpha * std::log(alpha) - std::lgamma(alpha) + max_log + std::log(sum * step);
  }
  return total;
}

}  // namespace

//
// 1. The model as implemented
//

TEST(Site_rate_gamma_test, the_integrated_density_of_alpha_matches_two_independent_computations) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(10, 300, 2468,
                        [](Site_index l) { return 0.2 + 2.0 * ((l % 5) / 4.0); });
  auto run = make_run(pool, tree, 11, true);
  for (auto alpha : {0.05, 0.2, 0.5, 1.0, 2.5, 10.0, 100.0}) {
    auto theirs = run->calc_cur_log_p_alpha(alpha);
    auto mine = independent_log_p_alpha(*run, alpha);
    auto quad = quadrature_log_p_alpha(*run, alpha);
    // calc_log_p_alpha drops the terms that are the same for every alpha only in the sense of the
    // lgamma(alpha) of the sites with no mutations, which cancel exactly; the two must agree to
    // floating-point accuracy.
    EXPECT_NEAR(theirs, mine, 1e-8 * std::max(1.0, std::abs(mine))) << "alpha " << alpha;
    // The quadrature is the weaker of the two routes: at alpha = 0.05 the integrand of a site
    // without mutations is extremely peaked, so it is held to an absolute tolerance in nats.
    EXPECT_NEAR(theirs, quad, 0.05) << "alpha " << alpha;
  }
}

TEST(Site_rate_gamma_test, the_site_rate_gibbs_draw_matches_its_exact_conditional) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(10, 60, 1357, [](Site_index l) { return 0.3 + 1.5 * (l % 4); });
  auto run = make_run(pool, tree, 13, true);
  run->set_alpha(0.7);

  auto Ttwiddle_l = calc_Ttwiddle_l(run->tree(), run->evo());
  auto M_l = calc_num_muts_l(run->tree());
  auto alpha = run->alpha();

  // The conditional of nu_l is Gamma(M_l + alpha, mu_l Ttwiddle_l + alpha); a long chain of Gibbs
  // sweeps gives independent draws of every site rate, so each site's draws are checked against the
  // exact quantiles of its own Gamma.
  constexpr auto k_draws = 4000;
  constexpr auto k_bins = 10;
  auto counts = std::vector<std::vector<int>>(run->tree().num_sites(),
                                              std::vector<int>(k_bins, 0));
  // Equal-probability bin edges of a Gamma(shape, rate), by bisection on its regularized incomplete
  // gamma function, computed here from a series that shares nothing with the sampler.
  auto lower_gamma_cdf = [](double shape, double x) {
    if (x <= 0.0) { return 0.0; }
    // Series for P(shape, x) when x < shape + 1, continued fraction otherwise (Numerical Recipes).
    if (x < shape + 1.0) {
      auto sum = 1.0 / shape;
      auto term = sum;
      for (auto n = 1; n != 2000; ++n) {
        term *= x / (shape + n);
        sum += term;
        if (std::abs(term) < std::abs(sum) * 1e-16) { break; }
      }
      return sum * std::exp(-x + shape * std::log(x) - std::lgamma(shape));
    }
    auto b = x + 1.0 - shape;
    auto c = 1.0 / 1e-300;
    auto d = 1.0 / b;
    auto h = d;
    for (auto i = 1; i != 2000; ++i) {
      auto an = -i * (i - shape);
      b += 2.0;
      d = an * d + b;
      if (std::abs(d) < 1e-300) { d = 1e-300; }
      c = b + an / c;
      if (std::abs(c) < 1e-300) { c = 1e-300; }
      d = 1.0 / d;
      auto del = d * c;
      h *= del;
      if (std::abs(del - 1.0) < 1e-16) { break; }
    }
    return 1.0 - std::exp(-x + shape * std::log(x) - std::lgamma(shape)) * h;
  };
  auto quantile = [&](double shape, double rate, double p) {
    auto lo = 1e-30;
    auto hi = 1.0;
    while (lower_gamma_cdf(shape, hi * rate) < p) { hi *= 2.0; if (hi > 1e30) { break; } }
    for (auto i = 0; i != 200; ++i) {
      auto mid = 0.5 * (lo + hi);
      if (lower_gamma_cdf(shape, mid * rate) < p) { lo = mid; } else { hi = mid; }
    }
    return 0.5 * (lo + hi);
  };
  auto edges = std::vector<std::vector<double>>{};
  for (auto l = Site_index{0}; l != run->tree().num_sites(); ++l) {
    auto shape = M_l[l] + alpha;
    auto rate = run->evo().mu_l(l) * Ttwiddle_l[l] + alpha;
    auto e = std::vector<double>{};
    for (auto b = 1; b != k_bins; ++b) {
      e.push_back(quantile(shape, rate, static_cast<double>(b) / k_bins));
    }
    edges.push_back(std::move(e));
  }

  for (auto draw = 0; draw != k_draws; ++draw) {
    run->gibbs_sample_all_nus();
    for (auto l = Site_index{0}; l != run->tree().num_sites(); ++l) {
      auto bin = static_cast<int>(std::ranges::upper_bound(edges[l], run->nu()[l]) - edges[l].begin());
      ++counts[l][bin];
    }
  }
  auto expected = static_cast<double>(k_draws) / k_bins;
  auto failures = 0;
  for (auto l = Site_index{0}; l != run->tree().num_sites(); ++l) {
    auto chi2 = 0.0;
    for (auto c : counts[l]) { chi2 += (c - expected) * (c - expected) / expected; }
    // 9 degrees of freedom: the 0.1% critical value is 27.9.  Over 60 sites a single excursion is
    // expected about once in sixteen runs, so the test counts them instead of failing on the first.
    if (chi2 > 27.9) { ++failures; }
  }
  EXPECT_LE(failures, 1) << failures << " of " << run->tree().num_sites()
                         << " sites deviate from their exact conditional";
}

TEST(Site_rate_gamma_test, with_no_exposure_the_site_rates_are_drawn_from_their_prior) {
  // With mu = 0 nothing is observed, so the conditional of nu_l collapses to the prior
  // Gamma(alpha, alpha), whose mean is one: this is the normalization that lets mu carry the scale.
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(6, 200, 99);
  for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
    tree.at(node).mutations.clear();
  }
  auto run = make_run(pool, tree, 17, true);
  // A tiny but positive rate: the exposure mu_l Ttwiddle_l is then negligible against alpha, so the
  // conditional is the prior.  Exactly zero is not used because the default Gamma(1, 0) prior on mu
  // evaluates 0 * log(0) there.
  run->set_mu(1e-20);
  for (auto alpha : {0.5, 1.0, 4.0}) {
    run->set_alpha(alpha);
    auto sum = 0.0;
    auto sum_sq = 0.0;
    auto n = 0;
    for (auto draw = 0; draw != 200; ++draw) {
      run->gibbs_sample_all_nus();
      for (auto nu : run->nu()) { sum += nu; sum_sq += nu * nu; ++n; }
    }
    auto mean = sum / n;
    auto var = sum_sq / n - mean * mean;
    EXPECT_NEAR(mean, 1.0, 0.05) << "alpha " << alpha;        // prior mean 1
    EXPECT_NEAR(var, 1.0 / alpha, 0.15 / alpha) << "alpha " << alpha;  // prior variance 1/alpha
  }
}

TEST(Site_rate_gamma_test, the_alpha_move_samples_the_exact_conditional) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(12, 150, 8642, [](Site_index l) { return 0.1 + 2.5 * ((l % 7) / 6.0); });
  auto run = make_run(pool, tree, 19, true);

  // The conditional of alpha given the history is the integrated density times its exponential
  // prior with mean one; the nu_l do not enter, so a chain of alpha moves alone targets it exactly.
  auto log_target = [&](double alpha) {
    if (not (alpha > 0.0)) { return -std::numeric_limits<double>::infinity(); }
    return independent_log_p_alpha(*run, alpha) - alpha;
  };
  constexpr auto k_grid = 40001;
  constexpr auto k_log_min = -7.0;
  constexpr auto k_log_max = 7.0;
  auto value_at = [&](int i) {
    return std::exp(k_log_min + (k_log_max - k_log_min) * i / (k_grid - 1));
  };
  auto max_log = -std::numeric_limits<double>::infinity();
  for (auto i = 0; i != k_grid; ++i) { max_log = std::max(max_log, log_target(value_at(i))); }
  auto density = std::vector<double>(k_grid, 0.0);
  for (auto i = 0; i != k_grid; ++i) {
    density[i] = std::exp(log_target(value_at(i)) - max_log) * value_at(i);  // uniform in log(alpha)
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
  auto counts = std::vector<int>(k_bins, 0);
  for (auto i = 0; i != k_keep; ++i) {
    run->alpha_moves();  // ten scaling proposals plus the nu Gibbs sweep, as in a real run
    auto bin = static_cast<int>(std::ranges::upper_bound(edges, run->alpha()) - edges.begin());
    ++counts[bin];
  }
  auto expected = static_cast<double>(k_keep) / k_bins;
  auto chi2 = 0.0;
  for (auto c : counts) { chi2 += (c - expected) * (c - expected) / expected; }
  // 19 degrees of freedom: the 0.1% critical value is 43.8
  EXPECT_LT(chi2, 43.8) << "counts " << absl::StrJoin(counts, ",");
}

//
// 2. Combination with the substitution model and the site partitions
//

TEST(Site_rate_gamma_test, the_site_rate_multiplies_the_partition_rate_in_the_posterior) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 200, 1111, [](Site_index l) { return l < 100 ? 0.5 : 2.0; });
  auto run = make_run(pool, tree, 23, true);
  run->set_alpha(1.5);
  run->gibbs_sample_all_nus();

  // log_G must agree with the independent per-site density, which multiplies mu, the partition rate
  // and nu_l for every site.
  auto independent = independent_log_G(run->tree(), run->evo());
  EXPECT_NEAR(run->log_G(), independent, 1e-7 * std::abs(independent));
  for (auto l = Site_index{0}; l != run->tree().num_sites(); ++l) {
    EXPECT_EQ(run->evo().nu_l[l], run->nu()[l]);
    EXPECT_EQ(run->evo().mu_l(l), run->mu());  // one partition: mu_l is mu, and nu_l is separate
  }
}

TEST(Site_rate_gamma_test, continuous_gamma_works_with_every_substitution_model) {
  auto pool = ctpl::thread_pool{1};
  for (const auto* spec : {"HKY85", "GTR", "JC69", "TN93", "SYM", "012032"}) {
    auto tree = make_tree(10, 300, 2222, [](Site_index l) { return 0.2 + 2.0 * ((l % 5) / 4.0); });
    auto run = make_run(pool, tree, 29, true);
    auto model = parse_subst_model(spec);
    model.mu = k_mu;
    if (model.frequency_mode != Frequency_mode::equal) {
      model.pi_a = Seq_vector<double>{0.31, 0.19, 0.17, 0.33};
    }
    run->set_subst_model(model);
    run->set_alpha(0.8);
    for (auto batch = 0; batch != 3; ++batch) {
      run->do_mcmc_steps(2000);
      auto fresh = run->calc_cur_log_G();
      EXPECT_NEAR(run->log_G(), fresh, 1e-6 * std::max(1.0, std::abs(fresh)))
          << spec << " batch " << batch;
      auto independent = independent_log_G(run->tree(), run->evo());
      EXPECT_NEAR(run->log_G(), independent, 1e-6 * std::max(1.0, std::abs(independent)))
          << spec << " batch " << batch;
      auto priors = run->calc_cur_log_other_priors();
      EXPECT_NEAR(run->log_other_priors(), priors, 1e-6 * std::max(1.0, std::abs(priors)))
          << spec << " batch " << batch;
      EXPECT_GT(run->alpha(), 0.0) << spec;
    }
  }
}

class Site_rate_gamma_run_test : public testing::TestWithParam<bool> {  // parameter: fixed topology
 protected:
  auto fixed_topology() const -> bool { return GetParam(); }
};

TEST_P(Site_rate_gamma_run_test, incremental_and_fresh_posteriors_agree_over_long_runs) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(12, 400, 3333, [](Site_index l) { return 0.1 + 3.0 * ((l % 9) / 8.0); });
  auto run = make_run(pool, tree, 31, fixed_topology());
  auto model = parse_subst_model("GTR");
  model.mu = k_mu;
  model.pi_a = Seq_vector<double>{0.31, 0.19, 0.17, 0.33};
  run->set_subst_model(model);
  for (auto batch = 0; batch != 5; ++batch) {
    run->do_mcmc_steps(4000);
    auto fresh = run->calc_cur_log_G();
    EXPECT_NEAR(run->log_G(), fresh, 1e-6 * std::max(1.0, std::abs(fresh))) << "batch " << batch;
    auto priors = run->calc_cur_log_other_priors();
    EXPECT_NEAR(run->log_other_priors(), priors, 1e-6 * std::max(1.0, std::abs(priors)))
        << "batch " << batch;
  }
}

INSTANTIATE_TEST_SUITE_P(Topology, Site_rate_gamma_run_test, testing::Values(true, false),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "fixed" : "free"; });

//
// 3. Numerical behaviour at the edges
//

TEST(Site_rate_gamma_test, sites_without_mutations_and_missing_data_stay_finite) {
  auto pool = ctpl::thread_pool{1};
  // Half the sites have no mutations at all and a block is missing in the first tip.
  auto tree = make_tree(8, 400, 4444, [](Site_index l) { return l < 200 ? 0.0 : 2.0; },
                        {{Site_index{300}, Site_index{360}}});
  auto run = make_run(pool, tree, 37, true);
  for (auto alpha : {1e-3, 0.01, 0.1, 1.0, 50.0, 1000.0}) {
    run->set_alpha(alpha);
    run->gibbs_sample_all_nus();
    EXPECT_TRUE(std::isfinite(run->log_G())) << "alpha " << alpha;
    EXPECT_TRUE(std::isfinite(run->log_other_priors())) << "alpha " << alpha;
    EXPECT_TRUE(std::isfinite(run->calc_cur_log_p_alpha(alpha))) << "alpha " << alpha;
    auto fresh = run->calc_cur_log_G();
    EXPECT_NEAR(run->log_G(), fresh, 1e-6 * std::max(1.0, std::abs(fresh))) << "alpha " << alpha;
    auto priors = run->calc_cur_log_other_priors();
    EXPECT_NEAR(run->log_other_priors(), priors, 1e-6 * std::max(1.0, std::abs(priors)))
        << "alpha " << alpha;
    for (auto nu : run->nu()) {
      EXPECT_GT(nu, 0.0) << "alpha " << alpha;
      EXPECT_TRUE(std::isfinite(nu)) << "alpha " << alpha;
    }
  }
}

TEST(Site_rate_gamma_test, the_floor_on_the_site_rates_is_reached_only_at_tiny_alpha) {
  // gibbs_sample_all_nus clamps nu_l at 1e-50 because a very concentrated Gamma can round to zero.
  // The clamp is a documented deviation from the continuous Gamma model, so this test records when it
  // actually bites: only sites without mutations, and only at an alpha far below any sampled value.
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 400, 5555, [](Site_index l) { return l < 200 ? 0.0 : 2.0; });
  auto run = make_run(pool, tree, 41, true);
  auto M_l = calc_num_muts_l(run->tree());
  for (auto alpha : {1e-3, 0.01, 0.1, 0.5, 1.0}) {
    run->set_alpha(alpha);
    run->gibbs_sample_all_nus();
    auto floored = 0;
    for (auto l = Site_index{0}; l != run->tree().num_sites(); ++l) {
      if (run->nu()[l] <= 1e-50) {
        ++floored;
        EXPECT_EQ(M_l[l], 0) << "a site with mutations was floored, at alpha " << alpha;
      }
    }
    if (alpha >= 0.1) {
      EXPECT_EQ(floored, 0) << "alpha " << alpha << ": " << floored << " sites floored";
    }
  }
}

TEST(Site_rate_gamma_test, the_alpha_move_leaves_a_run_without_mutations_alone) {
  // Every site has zero mutations and zero exposure: the integrated density is flat in alpha except
  // for its prior, so the chain must stay finite and alpha must stay positive.
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(6, 100, 6666);
  for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
    tree.at(node).mutations.clear();
  }
  auto run = make_run(pool, tree, 43, true);
  run->set_mu(1e-20);
  for (auto i = 0; i != 200; ++i) {
    run->alpha_moves();
    ASSERT_GT(run->alpha(), 0.0) << "iteration " << i;
    ASSERT_TRUE(std::isfinite(run->log_other_priors())) << "iteration " << i;
  }
}

}  // namespace delphy
