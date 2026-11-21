#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <random>
#include <sstream>

#include "ctpl_stl.h"

#include "calibration.h"
#include "coalescent_grid.h"
#include "estd.h"
#include "phylo_tree.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

constexpr auto k_n0 = 1000.0;  // days

// Kingman tree with every tip at t = 0, no mutations (one invariant site).
auto make_kingman_tree(int num_tips, std::mt19937_64& g, double n0 = k_n0) -> Phylo_tree {
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(1, Real_seq_letter::A);
  auto active = std::vector<Node_index>{};
  for (auto i = 0; i != num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = 0.0;
    tree.at(i).name = "S" + std::to_string(i);
    active.push_back(i);
  }
  auto age = 0.0;
  auto next = num_tips;
  while (std::ssize(active) > 1) {
    auto k = static_cast<double>(std::ssize(active));
    age += std::exponential_distribution<double>{k * (k - 1) / 2.0 / n0}(g);
    auto i = g() % active.size(); auto x = active[i]; active[i] = active.back(); active.pop_back();
    auto j = g() % active.size(); auto y = active[j]; active[j] = active.back(); active.pop_back();
    auto& node = tree.at(next);
    node.t = -age;
    node.t_min = -std::numeric_limits<float>::max();
    node.t_max = +std::numeric_limits<float>::max();
    node.name = "n" + std::to_string(next);
    node.children = {x, y};
    tree.at(x).parent = next;
    tree.at(y).parent = next;
    active.push_back(next++);
  }
  tree.root = next - 1;
  tree.at(tree.root).parent = k_no_node;
  return tree;
}

// A run on such a tree, with the global moves that would otherwise change the target of the test switched off
auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, double t_step, bool topology_moves,
              int parts = 1, int batch = 5) -> std::unique_ptr<delphy::Run> {
  auto run = std::make_unique<delphy::Run>(pool, std::mt19937{seed}, seed, tree, t_step);
  run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0, 0.0, 0.0));
  run->set_mu(1e-13);
  run->set_mu_move_enabled(false);
  run->set_alpha_move_enabled(false);
  run->set_final_pop_size_move_enabled(false);
  run->set_pop_growth_rate_move_enabled(false);
  run->set_num_parts(parts);
  run->set_local_moves_per_global_move(batch);
  run->set_topology_moves_enabled(topology_moves);
  run->set_coal_prior_fixed_t_step(t_step);
  return run;
}


// Tip names under an inner node
auto tips_under(const Phylo_tree& tree, Node_index node) -> std::vector<std::string> {
  auto out = std::vector<std::string>{};
  auto stack = std::vector<Node_index>{node};
  while (not stack.empty()) {
    auto v = stack.back(); stack.pop_back();
    if (tree.at(v).is_tip()) { out.push_back(tree.at(v).name); continue; }
    for (auto c : tree.at(v).children) { stack.push_back(c); }
  }
  std::ranges::sort(out);
  return out;
}

// A calibration configuration: the root and the inner node `node`, lognormal with a median 15% above the true age
// and hard bounds at 0.4 and 4 times it
auto calibration_config_for(const Phylo_tree& tree, Node_index node) -> std::string {
  auto os = std::ostringstream{};
  auto block = [&](const std::string& name, const std::string& taxa, double age_days) {
    auto age_years = age_days / 365.0;
    os << "calibration " << name << "\n  taxa " << taxa << "\n  age_units years_bp\n  age_reference_year 2020\n"
       << "  distribution lognormal\n  median " << 1.15 * age_years << "\n  sigma 0.2\n"
       << "  oldest " << 4.0 * age_years << "\n  youngest " << 0.4 * age_years << "\nend\n";
  };
  os << "version 1\n\n";
  block("root", "all", -tree.at(tree.root).t);
  auto taxa = std::string{};
  for (const auto& name : tips_under(tree, node)) { taxa += name + " "; }
  block("clade", taxa, -tree.at(node).t);
  return os.str();
}

// The non-root inner node with the most tips below it
auto largest_non_root_node(const Phylo_tree& tree) -> Node_index {
  auto best = k_no_node; auto best_size = 0;
  for (auto v = 0; v != std::ssize(tree); ++v) {
    if (tree.at(v).is_tip() || v == tree.root) { continue; }
    auto size = static_cast<int>(tips_under(tree, v).size());
    if (size > best_size) { best_size = size; best = v; }
  }
  return best;
}

}  // namespace

TEST(Exact_coalescent_correction_test, it_is_off_by_default_and_changes_nothing_then) {
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{1};
  auto tree = make_kingman_tree(8, g);
  auto run = make_run(pool, tree, 7, 250.0, true);
  EXPECT_FALSE(run->exact_coalescent_correction());
  EXPECT_EQ(run->log_coalescent_prior(), run->calc_cur_log_coalescent_prior());   // bit for bit
  EXPECT_EQ(run->calc_cur_log_coalescent_target(), run->calc_cur_log_coalescent_prior());
  run->do_mcmc_steps(200);
  EXPECT_EQ(run->exact_coalescent_batches_proposed(), 0);
  EXPECT_EQ(run->log_coalescent_prior(), run->calc_cur_log_coalescent_prior());
}

TEST(Exact_coalescent_correction_test, unsupported_configurations_are_refused) {
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{2};
  auto tree = make_kingman_tree(8, g);

  {  // free topology with several tree partition parts (one part is accepted; see exact_free_topology_tests.cpp)
    auto run = make_run(pool, tree, 7, 250.0, true, 2);
    EXPECT_THROW(run->set_exact_coalescent_correction(true), std::invalid_argument);
    EXPECT_FALSE(run->exact_coalescent_correction());
  }
  {  // grid not pinned
    auto run = std::make_unique<delphy::Run>(pool, std::mt19937{7}, 7, tree, 250.0);
    run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0, 0.0, 0.0));
    run->set_topology_moves_enabled(false);
    EXPECT_THROW(run->set_exact_coalescent_correction(true), std::invalid_argument);
  }
  {  // the constant and Skygrid population models are supported as well as the exponential one;
     // see `exact_skygrid_mcmc_tests.cpp` for the Skygrid moveset's own acceptance
    auto run = make_run(pool, tree, 7, 250.0, false);
    run->set_pop_model(std::make_shared<Const_pop_model>(k_n0));
    EXPECT_NO_THROW(run->set_exact_coalescent_correction(true));
    run->set_exact_coalescent_correction(false);
    auto x = std::vector<double>{-3.0 * k_n0, -2.0 * k_n0, -1.0 * k_n0, 0.0};
    auto gamma = std::vector<double>(4, std::log(k_n0));
    run->set_pop_model(std::make_shared<Skygrid_pop_model>(x, gamma, Skygrid_pop_model::Type::k_staircase));
    EXPECT_NO_THROW(run->set_exact_coalescent_correction(true));
  }
  {  // heterochronous samples with exact dates are supported: the exact density treats a tip as a
     // lineage-count change, and the correction's normalization argument does not use a shared date
    auto other = tree;
    other.at(3).t = other.at(3).t_min = other.at(3).t_max = -10.0;
    auto run = make_run(pool, other, 7, 250.0, false);
    EXPECT_NO_THROW(run->set_exact_coalescent_correction(true));
  }
  {  // tip with date uncertainty
    auto other = tree;
    other.at(3).t_min = -5.0f;
    auto run = make_run(pool, other, 7, 250.0, false);
    EXPECT_THROW(run->set_exact_coalescent_correction(true), std::invalid_argument);
  }
  {  // a supported configuration, then changes that would break it are refused
    auto run = make_run(pool, tree, 7, 250.0, false, 2);
    ASSERT_NO_THROW(run->set_exact_coalescent_correction(true));
    EXPECT_TRUE(run->exact_coalescent_correction());
    EXPECT_THROW(run->set_topology_moves_enabled(true), std::invalid_argument);
    EXPECT_THROW(run->set_coal_prior_fixed_t_step(0.0), std::invalid_argument);
    EXPECT_THROW(run->retarget_coal_prior_grid(), std::invalid_argument);
    EXPECT_NO_THROW(run->set_pop_model(std::make_shared<Const_pop_model>(k_n0)));
    {
      auto x = std::vector<double>{-3.0 * k_n0, -2.0 * k_n0, -1.0 * k_n0, 0.0};
      auto gamma = std::vector<double>(4, std::log(k_n0));
      EXPECT_NO_THROW(run->set_pop_model(
          std::make_shared<Skygrid_pop_model>(x, gamma, Skygrid_pop_model::Type::k_staircase)));
    }
    run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0, 0.0, 0.0));
    run->set_exact_coalescent_correction(false);
    EXPECT_FALSE(run->exact_coalescent_correction());
  }
}

TEST(Exact_coalescent_correction_test, the_reported_coalescent_term_is_the_exact_density_and_stays_consistent_through_rejections) {
  auto pool = ctpl::thread_pool{2};
  auto g = std::mt19937_64{3};
  auto tree = make_kingman_tree(20, g);
  for (auto parts : {1, 2}) {
    auto run = make_run(pool, tree, 11, 2.0 * k_n0, false, parts, 40);   // a grid so coarse that batches are often rejected
    run->set_exact_coalescent_correction(true);
    auto exact = [&](const delphy::Run& r) {
      return calc_exact_coalescent_log_prior(coalescent_event_times_of(r.tree()), r.pop_model());
    };
    EXPECT_NEAR(run->log_coalescent_prior(), exact(*run), 1e-9);

    for (auto i = 0; i != 120; ++i) {
      run->do_mcmc_steps(40);
      ASSERT_NEAR(run->log_coalescent_prior(), exact(*run), 1e-8) << "parts " << parts << " batch " << i;

      // a Run rebuilt from the current genealogy agrees on every derived quantity, whether the last batch was
      // accepted or rolled back
      auto fresh = make_run(pool, run->tree(), 5, 2.0 * k_n0, false, parts, 40);
      fresh->set_exact_coalescent_correction(true);
      fresh->set_hky_kappa(run->hky_kappa());
      fresh->set_hky_pi(run->hky_pi());
      ASSERT_NEAR(fresh->log_G(), run->log_G(), 1e-6 * std::max(1.0, std::abs(run->log_G())));
      ASSERT_NEAR(fresh->log_coalescent_prior(), run->log_coalescent_prior(), 1e-8);
    }
    EXPECT_EQ(run->exact_coalescent_batches_proposed(), 120);
    EXPECT_GT(run->exact_coalescent_batches_accepted(), 0);
    EXPECT_LT(run->exact_coalescent_batches_accepted(), 120);   // some batches were rolled back
  }
}

TEST(Exact_coalescent_correction_test, calibrations_are_accepted_with_any_number_of_partitions_in_either_order) {
  auto pool = ctpl::thread_pool{4};
  auto g = std::mt19937_64{21};
  auto tree = make_kingman_tree(20, g);
  auto config = calibration_config_for(tree, largest_non_root_node(tree));
  auto read = [&] { auto is = std::istringstream{config}; return read_calibration_config(is); };

  for (auto parts : {1, 2, 4}) {
    auto run = make_run(pool, tree, 7, 250.0, false, parts);
    run->set_calibrations(read());
    EXPECT_NO_THROW(run->set_exact_coalescent_correction(true)) << parts << " parts";
    EXPECT_TRUE(run->exact_coalescent_correction());
    EXPECT_NO_THROW(run->set_num_parts(parts == 1 ? 2 : 1));   // changing the number of parts afterwards is accepted too
    auto run2 = make_run(pool, tree, 7, 250.0, false, parts);
    run2->set_exact_coalescent_correction(true);
    EXPECT_NO_THROW(run2->set_calibrations(read())) << parts << " parts";
  }
}

// With calibrations and the correction, after every batch (accepted or rolled back) the genealogy is inside every hard support,
// the calibration prior kept by the sampler is the one recomputed from the tree, and a Run rebuilt from the genealogy agrees on the
// whole target: log G, the exact coalescent density, the calibration prior and the other priors.
TEST(Exact_coalescent_correction_test, with_calibrations_every_batch_leaves_a_consistent_state_inside_the_supports) {
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{22};
  auto tree = make_kingman_tree(20, g);
  auto node = largest_non_root_node(tree);
  auto config = calibration_config_for(tree, node);
  auto read = [&] { auto is = std::istringstream{config}; return read_calibration_config(is); };
  auto cal = read();
  auto clade_taxa = tips_under(tree, node);

  auto run = make_run(pool, tree, 11, 2.0 * k_n0, false, 1, 40);   // a coarse grid: many batches are rejected
  run->set_calibrations(read());
  run->set_exact_coalescent_correction(true);

  auto exact = [&](const delphy::Run& r) {
    return calc_exact_coalescent_log_prior(coalescent_event_times_of(r.tree()), r.pop_model());
  };
  for (auto i = 0; i != 150; ++i) {
    run->do_mcmc_steps(40);
    // inside the supports
    for (auto k = 0; k != cal.size(); ++k) {
      auto age_years = -run->tree().at(run->calibration_nodes()[k]).t / 365.0;
      ASSERT_GE(age_years, cal.calibrations[k].prior.support_youngest_age()) << "batch " << i;
      ASSERT_LE(age_years, cal.calibrations[k].prior.support_oldest_age()) << "batch " << i;
    }
    ASSERT_NEAR(run->log_calibration_prior(), run->calc_cur_log_calibration_prior(), 1e-9);
    ASSERT_NEAR(run->log_coalescent_prior(), exact(*run), 1e-8);

    auto fresh = make_run(pool, run->tree(), 5, 2.0 * k_n0, false, 1, 40);
    fresh->set_calibrations(read());
    fresh->set_exact_coalescent_correction(true);
    fresh->set_hky_kappa(run->hky_kappa());
    fresh->set_hky_pi(run->hky_pi());
    ASSERT_NEAR(fresh->log_calibration_prior(), run->log_calibration_prior(), 1e-9);
    ASSERT_NEAR(fresh->log_posterior(), run->log_posterior(), 1e-6 * std::max(1.0, std::abs(run->log_posterior())));
  }
  EXPECT_GT(run->exact_coalescent_batches_accepted(), 0);
  EXPECT_LT(run->exact_coalescent_batches_accepted(), 150);
  // the calibrated clade is still the same clade
  auto resolved = tips_under(run->tree(), run->calibration_nodes()[1]);
  EXPECT_EQ(resolved, clade_taxa);
}

// The same consistency checks on several parts of a larger tree: the calibrated nodes are inside their supports after every batch
// (accepted or rolled back), the calibration prior and the coalescent terms kept by the sampler equal the ones recomputed from the
// tree, a one-part Run rebuilt from the genealogy agrees on the whole target, and parts really were cut.  Debug builds also run the
// per-batch comparison of the global and the parts' totals (frozen calibrations included) inside do_mcmc_steps.
TEST(Exact_coalescent_correction_test, with_calibrations_and_several_parts_every_batch_leaves_a_consistent_state) {
  for (auto parts : {2, 4}) {
    auto pool = ctpl::thread_pool{parts};
    auto g = std::mt19937_64{23u + parts};
    auto tree = make_kingman_tree(120, g);
    auto node = largest_non_root_node(tree);
    auto config = calibration_config_for(tree, node);
    auto read = [&] { auto is = std::istringstream{config}; return read_calibration_config(is); };
    auto cal = read();
    auto clade_taxa = tips_under(tree, node);

    auto run = make_run(pool, tree, 13, 2.0 * k_n0, false, parts, 120);   // a coarse grid: many batches are rejected
    run->set_calibrations(read());
    run->set_exact_coalescent_correction(true);

    auto batches = estd::is_debug_enabled ? 40 : 150;
    auto multipart_batches = 0;
    auto exact = [&](const delphy::Run& r) {
      return calc_exact_coalescent_log_prior(coalescent_event_times_of(r.tree()), r.pop_model());
    };
    for (auto i = 0; i != batches; ++i) {
      run->do_mcmc_steps(120);
      if (std::ssize(run->partition().parts()) > 1) { ++multipart_batches; }
      for (auto k = 0; k != cal.size(); ++k) {
        auto age_years = -run->tree().at(run->calibration_nodes()[k]).t / 365.0;
        ASSERT_GE(age_years, cal.calibrations[k].prior.support_youngest_age()) << parts << " parts, batch " << i;
        ASSERT_LE(age_years, cal.calibrations[k].prior.support_oldest_age()) << parts << " parts, batch " << i;
      }
      ASSERT_NEAR(run->log_calibration_prior(), run->calc_cur_log_calibration_prior(), 1e-9);
      ASSERT_NEAR(run->log_coalescent_prior(), exact(*run), 1e-8);

      auto one_pool = ctpl::thread_pool{1};
      auto fresh = make_run(one_pool, run->tree(), 5, 2.0 * k_n0, false, 1, 120);
      fresh->set_calibrations(read());
      fresh->set_exact_coalescent_correction(true);
      fresh->set_hky_kappa(run->hky_kappa());
      fresh->set_hky_pi(run->hky_pi());
      ASSERT_NEAR(fresh->log_calibration_prior(), run->log_calibration_prior(), 1e-9);
      ASSERT_NEAR(fresh->log_posterior(), run->log_posterior(), 1e-6 * std::max(1.0, std::abs(run->log_posterior())));
    }
    EXPECT_GT(multipart_batches, batches / 4) << parts << " parts requested";
    EXPECT_EQ(run->exact_coalescent_batches_proposed(), batches);
    EXPECT_GT(run->exact_coalescent_batches_accepted(), 0);
    EXPECT_LT(run->exact_coalescent_batches_accepted(), batches);   // some batches were rolled back
    EXPECT_EQ(tips_under(run->tree(), run->calibration_nodes()[1]), clade_taxa);
  }
}

// Without the correction the sampler's first coalescence is too old on a coarse grid; with it the mean agrees with Kingman.
TEST(Exact_coalescent_correction_test, the_correction_removes_the_grid_bias_of_the_first_coalescence_age) {
  auto num_tips = 8;
  // Debug and sanitizer builds are 40 times slower: fewer chains, and a threshold matched to the lower power
  auto chains = estd::is_debug_enabled ? 160 : 500;
  auto iterations = estd::is_debug_enabled ? 20 : 60;
  auto steps = estd::is_debug_enabled ? 40 : 5;   // local moves per batch (a longer relaxation per iteration in debug)
  auto expected = k_n0 / (num_tips * (num_tips - 1) / 2.0);   // mean age of the first coalescence

  auto z_score = [&](bool correct) {
    auto pool = ctpl::thread_pool{1};
    auto g = std::mt19937_64{correct ? 101u : 102u};
    auto means = std::vector<double>{};
    for (auto c = 0; c != chains; ++c) {
      auto tree = make_kingman_tree(num_tips, g);   // exact draw: stationary start
      auto run = make_run(pool, tree, static_cast<uint32_t>(g()), 1.0 * k_n0, false, 1, steps);
      if (correct) { run->set_exact_coalescent_correction(true); }
      auto sum = 0.0;
      for (auto it = 0; it != iterations; ++it) {
        run->do_mcmc_steps(steps);
        auto first = std::numeric_limits<double>::max();
        for (const auto& node : index_order_traversal(run->tree())) {
          if (not run->tree().at(node).is_tip()) { first = std::min(first, -run->tree().at(node).t); }
        }
        sum += first / iterations;
      }
      means.push_back(sum);
    }
    auto mean = 0.0; for (auto m : means) { mean += m; } mean /= chains;
    auto var = 0.0; for (auto m : means) { var += (m - mean) * (m - mean); } var /= (chains - 1);
    return (mean - expected) / std::sqrt(var / chains);
  };

  EXPECT_LT(std::abs(z_score(true)), 4.0);
  EXPECT_GT(std::abs(z_score(false)), estd::is_debug_enabled ? 3.0 : 6.0);
}

}  // namespace delphy
