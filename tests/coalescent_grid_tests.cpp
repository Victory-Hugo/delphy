#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <random>

#include "coalescent.h"
#include "coalescent_grid.h"
#include "scalable_coalescent.h"

namespace delphy {

namespace {

constexpr auto k_year = 365.0;  // Delphy's calendar convention: 1 year = 365 days

// Composite Simpson's rule in long double, used as an independent check on the closed-form
// integrals in `Pop_model`.  `n` must be even.
template<typename F>
auto simpson(F f, double a, double b, int n) -> double {
  auto h = (static_cast<long double>(b) - a) / n;
  auto sum = static_cast<long double>(f(a)) + f(b);
  for (auto i = 1; i != n; ++i) {
    auto t = a + i * h;
    sum += (i % 2 == 0 ? 2.0L : 4.0L) * f(static_cast<double>(t));
  }
  return static_cast<double>(sum * h / 3.0L);
}

// A genealogy with every tip at `t_ref` and coalescences drawn from the Kingman coalescent with a
// constant coalescent timescale.  Deterministic given the seed.
auto kingman_events(int num_tips, double pop, double t_ref, uint32_t seed) -> Coalescent_event_times {
  auto prng = std::mt19937{seed};
  auto result = Coalescent_event_times{};
  result.tip_times.assign(num_tips, t_ref);
  auto t = t_ref;
  for (auto k = num_tips; k >= 2; --k) {
    auto rate = static_cast<double>(k) * (k - 1) / 2.0 / pop;  // not `k * (k - 1)`: int overflow from k = 46,342
    t -= std::exponential_distribution<double>{rate}(prng);
    result.coalescence_times.push_back(t);
  }
  return result;
}

// `Scalable_coalescent_prior` is Delphy's production coalescent prior.  Drive it the way `Run` does
// and read off its log prior for a given cell width.
auto cell_log_prior(const Coalescent_event_times& events,
                    std::shared_ptr<const Pop_model> pop_model,
                    double t_ref,
                    double t_step)
    -> double {
  auto num_tips = std::ssize(events.tip_times);
  auto num_coalescences = std::ssize(events.coalescence_times);
  auto prior = Scalable_coalescent_prior{pop_model, static_cast<Node_index>(num_tips + num_coalescences),
                                         t_ref, t_step};
  for (auto i = 0; i != num_coalescences; ++i) { prior.mark_as_coalescence(i); }
  for (auto i = num_coalescences; i != num_coalescences + num_tips; ++i) { prior.mark_as_tip(i); }
  for (auto i = 0; i != num_coalescences; ++i) {
    prior.displace_coalescence(i, events.coalescence_times[i]);
  }
  for (auto i = 0; i != num_tips; ++i) {
    prior.displace_tip(num_coalescences + i, events.tip_times[i]);
  }
  return prior.calc_log_prior();
}

// The three population models a human analysis can be run with, at a coalescent timescale of the
// same order, so that the grid comparison is not dominated by a difference in scale.
auto const_pop_model() -> std::shared_ptr<const Pop_model> {
  return std::make_shared<Const_pop_model>(75'000 * k_year);
}

auto exp_pop_model() -> std::shared_ptr<const Pop_model> {
  // Growing towards the present by a factor of e every 100 ka; the floor is 100 years, which is a
  // stated input and not a Delphy default.
  return std::make_shared<Exp_pop_model>(0.0, 75'000 * k_year, 1.0 / (100'000 * k_year), 100 * k_year);
}

auto skygrid_pop_model(Skygrid_pop_model::Type type) -> std::shared_ptr<const Pop_model> {
  // Knots every 10 ka back to 400 ka, with a log-population curve that bends rather than merely
  // tilting, so that terms (II) and (III) of the decomposition in coalescent_grid.h are exercised.
  auto num_knots = 41;
  auto x = std::vector<double>{};
  auto gamma = std::vector<double>{};
  for (auto k = 0; k != num_knots; ++k) {
    auto frac = static_cast<double>(k) / (num_knots - 1);
    x.push_back((-400'000.0 + frac * 400'000.0) * k_year);
    gamma.push_back(std::log(75'000 * k_year) + 0.8 * std::sin(6.0 * frac));
  }
  return std::make_shared<Skygrid_pop_model>(std::move(x), std::move(gamma), type);
}

struct Named_pop_model {
  const char* name;
  std::shared_ptr<const Pop_model> model;
};

auto all_pop_models() -> std::vector<Named_pop_model> {
  return {{"constant", const_pop_model()},
          {"exponential", exp_pop_model()},
          {"skygrid-staircase", skygrid_pop_model(Skygrid_pop_model::Type::k_staircase)},
          {"skygrid-log-linear", skygrid_pop_model(Skygrid_pop_model::Type::k_log_linear)}};
}

}  // namespace

// A two-tip genealogy has a coalescent density that can be written down by hand:
// with a constant coalescent timescale N, one pair of lineages waits (t_tip - t_coal) at rate 1/N,
// and the single coalescence contributes a factor 1/N.
TEST(Coalescent_grid_test, exact_log_prior_matches_a_hand_calculation_on_a_two_tip_genealogy) {
  auto pop = 75'000 * k_year;
  auto pop_model = Const_pop_model{pop};
  auto events = Coalescent_event_times{.tip_times = {0.0, 0.0}, .coalescence_times = {-140'000 * k_year}};

  auto expected = -std::log(pop) - (140'000 * k_year) / pop;

  EXPECT_NEAR(calc_exact_coalescent_log_prior(events, pop_model), expected, 1e-12 * std::abs(expected));
}

TEST(Coalescent_grid_test, exact_log_prior_matches_a_hand_calculation_with_heterochronous_tips) {
  auto pop = 50'000 * k_year;
  auto pop_model = Const_pop_model{pop};
  // Two modern tips and one tip 40 ka old; coalescences at 60 ka and 100 ka.
  auto events = Coalescent_event_times{
    .tip_times = {0.0, 0.0, -40'000 * k_year},
    .coalescence_times = {-60'000 * k_year, -100'000 * k_year}};

  // Walking forwards: 1 lineage before 100 ka; 2 in [100 ka, 60 ka); 3 in [60 ka, 40 ka);
  // 2 in [40 ka, 0).
  auto expected = 0.0
      - (2 * 1) / 2.0 * (100'000 - 60'000) * k_year / pop
      - (3 * 2) / 2.0 * (60'000 - 40'000) * k_year / pop
      - (2 * 1) / 2.0 * (40'000 - 0) * k_year / pop
      - 2 * std::log(pop);

  EXPECT_NEAR(calc_exact_coalescent_log_prior(events, pop_model), expected, 1e-12 * std::abs(expected));
}

// The diagnostic must report the difference that Delphy's production cell prior actually has against
// the exact Kingman density -- for every population model, not only for a constant one.
TEST(Coalescent_grid_test, the_measured_excess_matches_the_production_cell_prior_for_every_pop_model) {
  for (const auto& [name, pop_model] : all_pop_models()) {
    for (auto num_tips : {2, 5, 20, 200, 1'000}) {
      auto events = kingman_events(num_tips, 75'000 * k_year, 0.0, 20250101u + num_tips);
      auto exact = calc_exact_coalescent_log_prior(events, *pop_model);
      auto span = -*std::ranges::min_element(events.coalescence_times);

      for (auto num_cells : {37, 400, 4'000, 40'000}) {
        auto t_step = span / num_cells;
        auto cells = cell_log_prior(events, pop_model, 0.0, t_step);
        auto measured = calc_coalescent_grid_excess(events, *pop_model, 0.0, t_step);

        EXPECT_NEAR(cells - exact, measured.excess, 1e-6 + 1e-9 * std::abs(exact))
            << name << " num_tips=" << num_tips << " num_cells=" << num_cells;
        // The variance term is sign-definite whatever the population model does
        EXPECT_GE(measured.variance_term, 0.0) << name;
      }
    }
  }
}

// `calc_cell_coalescent_log_prior` must agree with the production class it stands in for, so that
// the two can be compared without building a `Scalable_coalescent_prior`.
TEST(Coalescent_grid_test, the_standalone_cell_prior_agrees_with_the_production_class) {
  for (const auto& [name, pop_model] : all_pop_models()) {
    auto events = kingman_events(50, 75'000 * k_year, 0.0, 31u);
    auto span = -*std::ranges::min_element(events.coalescence_times);
    for (auto num_cells : {37, 400, 4'000}) {
      auto t_step = span / num_cells;
      auto expected = cell_log_prior(events, pop_model, 0.0, t_step);
      auto actual = calc_cell_coalescent_log_prior(events, *pop_model, 0.0, t_step);
      EXPECT_NEAR(actual, expected, 1e-6 + 1e-10 * std::abs(expected))
          << name << " num_cells=" << num_cells;
    }
  }
}

// Condition under which the closed form of coalescent_grid.h holds: the variance term equals the
// whole difference only when N(t) is constant across each cell.
TEST(Coalescent_grid_test, the_variance_closed_form_is_exact_for_a_constant_population_only) {
  auto events = kingman_events(200, 75'000 * k_year, 0.0, 41u);
  auto span = -*std::ranges::min_element(events.coalescence_times);
  auto t_step = span / 400;

  auto constant = calc_coalescent_grid_excess(events, *const_pop_model(), 0.0, t_step);
  EXPECT_NEAR(constant.excess, constant.variance_term, 1e-9 * std::abs(constant.excess));
  EXPECT_GT(constant.excess, 0.0);

  // With a population size that varies inside the cells, the variance term is only the leading
  // contribution: the terms that involve 1/N are not negligible and need not even share its sign.
  auto num_disagreements = 0;
  for (const auto& [name, pop_model] : all_pop_models()) {
    if (std::string{name} == "constant") { continue; }
    auto measured = calc_coalescent_grid_excess(events, *pop_model, 0.0, t_step);
    EXPECT_GE(measured.variance_term, 0.0) << name;
    if (std::abs(measured.excess - measured.variance_term)
        > 1e-6 * std::max(1.0, std::abs(measured.variance_term))) {
      ++num_disagreements;
    }
  }
  EXPECT_GT(num_disagreements, 0)
      << "the closed form must not be assumed to hold for a varying population size";
}

// What a Metropolis-Hastings ratio consumes is the *difference* of the log prior between two
// genealogies, and much of the absolute discretization error cancels in that difference.  The
// residual is the quantity that actually distorts the sampled posterior, so it is measured directly
// against the exact event-driven delta.
TEST(Coalescent_grid_test, the_residual_error_of_a_proposal_delta_is_far_below_the_absolute_error) {
  for (const auto& [name, pop_model] : all_pop_models()) {
    auto events = kingman_events(200, 75'000 * k_year, 0.0, 43u);
    auto span = -*std::ranges::min_element(events.coalescence_times);
    auto t_step = span / 400;

    auto absolute = std::abs(calc_coalescent_grid_excess(events, *pop_model, 0.0, t_step).excess);
    EXPECT_GT(absolute, 0.1) << name;  // The single-tree error at 400 cells is far from negligible

    auto worst_delta_error = 0.0;
    auto prng = std::mt19937{47u};
    for (auto i = 0; i != std::ssize(events.coalescence_times); ++i) {
      auto old_t = events.coalescence_times[i];
      // A proposal of the size `inner_node_displace` makes: a fraction of the tree span.  The
      // proposal is kept below the youngest tip, so that it is a genealogy and every lineage count
      // stays positive.
      auto new_t = std::min(old_t + std::uniform_real_distribution<double>{-0.02 * span, 0.02 * span}(prng),
                            -1e-3 * span);

      auto exact_delta = calc_exact_delta_log_prior_after_displace_coalescence(
          events, *pop_model, old_t, new_t);

      auto moved = events;
      moved.coalescence_times[i] = new_t;
      auto cell_delta = calc_cell_coalescent_log_prior(moved, *pop_model, 0.0, t_step)
          - calc_cell_coalescent_log_prior(events, *pop_model, 0.0, t_step);

      worst_delta_error = std::max(worst_delta_error, std::abs(cell_delta - exact_delta));
    }

    // The per-proposal residual is what biases the sampler, and for every population model it is
    // more than an order of magnitude below the absolute error of either tree on its own: most of
    // the discretization error is common to the two genealogies and cancels.  A cell count chosen
    // from the absolute error is therefore far stricter than the sampler needs; see
    // doc/extended/coalescent_grid.md section 4.
    EXPECT_LT(worst_delta_error, 0.2 * absolute) << name;
    EXPECT_LT(worst_delta_error, 1.0) << name;
  }
}

// The exact event-driven delta has to agree with recomputing the exact density from scratch, or it
// cannot serve as the reference for the cell prior's incremental update.
TEST(Coalescent_grid_test, the_exact_proposal_delta_agrees_with_recomputing_the_exact_density) {
  for (const auto& [name, pop_model] : all_pop_models()) {
    auto events = kingman_events(60, 75'000 * k_year, 0.0, 53u);
    auto span = -*std::ranges::min_element(events.coalescence_times);
    auto before = calc_exact_coalescent_log_prior(events, *pop_model);
    auto prng = std::mt19937{59u};

    for (auto i = 0; i != std::ssize(events.coalescence_times); ++i) {
      auto old_t = events.coalescence_times[i];
      auto new_t = std::min(old_t + std::uniform_real_distribution<double>{-0.05 * span, 0.05 * span}(prng),
                            -1e-3 * span);

      auto delta = calc_exact_delta_log_prior_after_displace_coalescence(
          events, *pop_model, old_t, new_t);

      auto moved = events;
      moved.coalescence_times[i] = new_t;
      auto after = calc_exact_coalescent_log_prior(moved, *pop_model);

      EXPECT_NEAR(delta, after - before, 1e-8 + 1e-10 * std::abs(before))
          << name << " node=" << i;
    }
  }
}

TEST(Coalescent_grid_test, the_exact_proposal_delta_is_zero_for_a_move_that_changes_nothing) {
  auto events = kingman_events(8, 75'000 * k_year, 0.0, 61u);
  auto t = events.coalescence_times.front();
  EXPECT_EQ(calc_exact_delta_log_prior_after_displace_coalescence(events, *const_pop_model(), t, t),
            0.0);
}

// The excess is what makes a coarse grid unsafe at human sample sizes, and it grows with the number
// of tips at a fixed cell count rather than with the time span.
TEST(Coalescent_grid_test, the_excess_grows_with_the_number_of_tips_at_a_fixed_cell_count) {
  auto pop_model = Const_pop_model{75'000 * k_year};
  auto previous = -1.0;
  for (auto num_tips : {20, 100, 1'000}) {
    auto events = kingman_events(num_tips, 75'000 * k_year, 0.0, 7u);
    auto span = -*std::ranges::min_element(events.coalescence_times);
    auto excess = calc_coalescent_grid_excess(events, pop_model, 0.0, span / 400).excess;
    EXPECT_GT(excess, previous) << "num_tips=" << num_tips;
    previous = excess;
  }
  // At 400 cells -- Delphy's default -- a thousand same-date tips already cost tens of nats.
  EXPECT_GT(previous, 10.0);
}

// Refining the grid drives the excess to zero, so the cell prior does converge to the exact
// density; it just does not do so at the default resolution.
TEST(Coalescent_grid_test, refining_the_grid_drives_the_excess_to_zero_at_human_depths) {
  auto pop_model = std::make_shared<Const_pop_model>(75'000 * k_year);
  auto events = kingman_events(200, 75'000 * k_year, 0.0, 11u);
  auto exact = calc_exact_coalescent_log_prior(events, *pop_model);
  auto span = -*std::ranges::min_element(events.coalescence_times);

  auto coarse = std::abs(cell_log_prior(events, pop_model, 0.0, span / 400) - exact);
  auto fine = std::abs(cell_log_prior(events, pop_model, 0.0, span / 400'000) - exact);

  EXPECT_LT(fine, 1e-3);
  EXPECT_LT(fine, coarse / 100.0);
}

// Deep time: the ages a human mtDNA or chrY analysis has to represent.
TEST(Coalescent_grid_test, excess_and_exact_density_are_well_defined_at_deep_human_ages) {
  auto pop_model = std::make_shared<Const_pop_model>(75'000 * k_year);

  for (auto age_ka : {50.0, 100.0, 140.0, 200.0, 352.0}) {
    // A caterpillar genealogy whose coalescences are spread evenly over the whole depth, so that
    // every cell is exercised rather than just the recent ones.
    auto num_tips = 16;
    auto events = Coalescent_event_times{};
    events.tip_times.assign(num_tips, 0.0);
    for (auto i = 0; i != num_tips - 1; ++i) {
      events.coalescence_times.push_back(-age_ka * 1000.0 * k_year * (i + 1) / (num_tips - 1));
    }

    auto exact = calc_exact_coalescent_log_prior(events, *pop_model);
    EXPECT_TRUE(std::isfinite(exact)) << "age_ka=" << age_ka;

    auto span = age_ka * 1000.0 * k_year;
    for (auto num_cells : {400, 4'000, 40'000}) {
      auto t_step = span / num_cells;
      auto cells = cell_log_prior(events, pop_model, 0.0, t_step);
      auto predicted = calc_coalescent_grid_excess(events, *pop_model, 0.0, t_step);
      EXPECT_NEAR(cells - exact, predicted.excess, 1e-6 + 1e-9 * std::abs(exact))
          << "age_ka=" << age_ka << " num_cells=" << num_cells;
    }
  }
}

TEST(Coalescent_grid_test, the_grid_chooser_meets_its_tolerance_or_reports_the_remainder) {
  auto pop_model = Const_pop_model{75'000 * k_year};
  auto events = kingman_events(200, 75'000 * k_year, 0.0, 13u);

  auto generous = choose_coalescent_grid_t_step(events, pop_model, 0.0, 1e-2, 400, 4'000'000);
  EXPECT_LE(generous.excess, 1e-2);
  EXPECT_GT(generous.num_cells, 400);

  // A tolerance that cannot be met within the cell cap returns the finest grid allowed, and its
  // excess, rather than silently claiming success.
  auto impossible = choose_coalescent_grid_t_step(events, pop_model, 0.0, 1e-12, 400, 800);
  EXPECT_GT(impossible.excess, 1e-12);
  EXPECT_LE(impossible.num_cells, 1'600);

  // A tolerance the coarsest grid already meets must not refine the grid at all.
  auto trivial = choose_coalescent_grid_t_step(events, pop_model, 0.0, 1e6, 400, 4'000'000);
  EXPECT_LE(trivial.num_cells, 400);
}

TEST(Coalescent_grid_test, the_grid_chooser_rejects_invalid_arguments) {
  auto pop_model = Const_pop_model{75'000 * k_year};
  auto events = kingman_events(8, 75'000 * k_year, 0.0, 17u);

  EXPECT_ANY_THROW(choose_coalescent_grid_t_step(events, pop_model, 0.0, 0.0, 400, 4'000));
  EXPECT_ANY_THROW(choose_coalescent_grid_t_step(events, pop_model, 0.0, 1.0, 0, 4'000));
  EXPECT_ANY_THROW(choose_coalescent_grid_t_step(events, pop_model, 0.0, 1.0, 4'000, 400));
  EXPECT_ANY_THROW(calc_coalescent_grid_excess(events, pop_model, 0.0, 0.0));
}

// Units audit: the coalescent timescale is a time, so a genealogy whose ages are expressed in days
// against a timescale in days must give the same density as the same genealogy and timescale scaled
// together.  Only the `sum_i log N(t_i)` term carries the unit, and it does so once per coalescence.
TEST(Coalescent_grid_test, rescaling_time_and_the_coalescent_timescale_together_shifts_the_density_by_a_known_amount) {
  auto events_days = kingman_events(32, 75'000 * k_year, 0.0, 23u);
  auto log_prior_days = calc_exact_coalescent_log_prior(events_days, Const_pop_model{75'000 * k_year});

  auto events_years = events_days;
  for (auto& t : events_years.tip_times) { t /= k_year; }
  for (auto& t : events_years.coalescence_times) { t /= k_year; }
  auto log_prior_years = calc_exact_coalescent_log_prior(events_years, Const_pop_model{75'000.0});

  auto num_coalescences = std::ssize(events_days.coalescence_times);
  EXPECT_NEAR(log_prior_days, log_prior_years - num_coalescences * std::log(k_year), 1e-9);
}

// The closed-form integrals in `Pop_model` are what the coalescent prior consumes, so they are
// checked against high-precision quadrature at human depths rather than only near the tips.
TEST(Pop_model_deep_time_test, exponential_model_integrals_match_quadrature_at_human_depths) {
  for (auto growth_per_year : {-2e-4, 0.0, 5e-5, 2e-4}) {
    auto pop_model = Exp_pop_model{0.0, 75'000 * k_year, growth_per_year / k_year, 0.0};
    for (auto age_ka : {50.0, 100.0, 140.0, 200.0, 352.0}) {
      auto a = -age_ka * 1000.0 * k_year;
      auto b = a + 1000.0 * k_year;  // A one-kiloyear window at that depth

      auto quad_pop = simpson([&](double t) { return pop_model.pop_at_time(t); }, a, b, 20'000);
      auto quad_intensity = simpson([&](double t) { return 1.0 / pop_model.pop_at_time(t); }, a, b, 20'000);

      EXPECT_NEAR(pop_model.pop_integral(a, b), quad_pop, 1e-9 * std::abs(quad_pop))
          << "g=" << growth_per_year << " age_ka=" << age_ka;
      EXPECT_NEAR(pop_model.intensity_integral(a, b), quad_intensity, 1e-9 * std::abs(quad_intensity))
          << "g=" << growth_per_year << " age_ka=" << age_ka;
    }
  }
}

TEST(Pop_model_deep_time_test, skygrid_model_integrals_match_quadrature_at_human_depths) {
  // Knots every 10 ka from 200 ka to today, with a log-population curve that varies by a factor of
  // e over the whole range.
  auto num_knots = 21;
  auto x = std::vector<double>{};
  auto gamma = std::vector<double>{};
  for (auto k = 0; k != num_knots; ++k) {
    x.push_back(-200'000 * k_year + k * (200'000.0 / (num_knots - 1)) * k_year);
    gamma.push_back(std::log(75'000 * k_year) + static_cast<double>(k) / (num_knots - 1));
  }

  for (auto type : {Skygrid_pop_model::Type::k_staircase, Skygrid_pop_model::Type::k_log_linear}) {
    auto pop_model = Skygrid_pop_model{x, gamma, type};
    for (auto age_ka : {50.0, 100.0, 140.0, 200.0, 352.0}) {
      // A one-kiloyear window placed strictly inside a knot interval.  A staircase population is
      // discontinuous at a knot, and Simpson's rule is not a valid reference across a jump, so the
      // window must not touch one.
      auto a = -(age_ka * 1000.0 - 2'500.0) * k_year;
      auto b = a + 1000.0 * k_year;

      auto quad_pop = simpson([&](double t) { return pop_model.pop_at_time(t); }, a, b, 20'000);
      auto quad_intensity = simpson([&](double t) { return 1.0 / pop_model.pop_at_time(t); }, a, b, 20'000);

      EXPECT_NEAR(pop_model.pop_integral(a, b), quad_pop, 1e-8 * std::abs(quad_pop))
          << "type=" << static_cast<int>(type) << " age_ka=" << age_ka;
      EXPECT_NEAR(pop_model.intensity_integral(a, b), quad_intensity, 1e-8 * std::abs(quad_intensity))
          << "type=" << static_cast<int>(type) << " age_ka=" << age_ka;
    }
  }
}

// A Skygrid curve whose knots span human depths must still reproduce a constant population exactly,
// including across the open-ended first and last intervals.
TEST(Pop_model_deep_time_test, skygrid_reproduces_a_constant_population_across_deep_knots) {
  auto num_knots = 51;
  auto x = std::vector<double>{};
  auto gamma = std::vector<double>{};
  for (auto k = 0; k != num_knots; ++k) {
    x.push_back(-352'000 * k_year + k * (352'000.0 / (num_knots - 1)) * k_year);
    gamma.push_back(std::log(75'000 * k_year));
  }

  for (auto type : {Skygrid_pop_model::Type::k_staircase, Skygrid_pop_model::Type::k_log_linear}) {
    auto pop_model = Skygrid_pop_model{x, gamma, type};
    auto a = -500'000 * k_year;  // Older than the first knot
    auto b = +1'000 * k_year;    // Later than the last knot
    EXPECT_NEAR(pop_model.pop_integral(a, b), (b - a) * 75'000 * k_year, 1e-9 * (b - a) * 75'000 * k_year);
    EXPECT_NEAR(pop_model.intensity_integral(a, b), (b - a) / (75'000 * k_year),
                1e-9 * (b - a) / (75'000 * k_year));
  }
}


// `(k * (k - 1)) / 2` used to be evaluated in `int`, which overflows from k = 46,342 lineages.  The reference below uses
// summation by parts, H = sum_{j=1}^{n-1} (n - j) Lambda(a_j) over the coalescence ages a_1 < ... < a_{n-1} of n tips
// sampled at one time, which shares no code with the walk over merged lineage changes in the functions under test.
TEST(Coalescent_grid_test, the_exact_prior_does_not_overflow_the_pair_count_at_tens_of_thousands_of_tips) {
  auto n0 = 75'000 * k_year;
  auto growth = 1.0 / (100'000 * k_year);  // per day

  for (auto num_tips : {46'341, 46'342, 50'000, 52'930}) {
    auto events = kingman_events(num_tips, n0, 0.0, 12345 + num_tips);
    auto ages = std::vector<long double>{};
    for (auto t : events.coalescence_times) { ages.push_back(-static_cast<long double>(t)); }
    std::ranges::sort(ages);
    ASSERT_EQ(std::ssize(ages), num_tips - 1);

    for (auto exponential : {false, true}) {
      auto pop_model = std::shared_ptr<const Pop_model>{};
      if (exponential) {
        pop_model = std::make_shared<Exp_pop_model>(0.0, n0, growth, 100 * k_year);
      } else {
        pop_model = std::make_shared<Const_pop_model>(n0);
      }

      // Independent reference: Lambda(a) = int_0^a dx / N(x), N(x) = n0 exp(-g x) (floor not reached: checked below)
      auto lambda = [&](long double a) -> long double {
        return exponential ? std::expm1(static_cast<long double>(growth) * a) / (static_cast<long double>(growth) * n0)
                           : a / n0;
      };
      auto hazard = 0.0L;
      auto log_pop_sum = 0.0L;
      for (auto j = 1; j <= num_tips - 1; ++j) {
        hazard += static_cast<long double>(num_tips - j) * lambda(ages[j - 1]);
        log_pop_sum += std::log(static_cast<long double>(n0)) - (exponential ? static_cast<long double>(growth) * ages[j - 1] : 0.0L);
      }
      if (exponential) {
        ASSERT_GT(n0 * std::exp(-static_cast<long double>(growth) * ages.back()), 100 * k_year);  // floor not reached
      }
      auto expected = static_cast<double>(-hazard - log_pop_sum);

      auto from_function = calc_exact_coalescent_log_prior(events, *pop_model);
      EXPECT_NEAR(from_function, expected, 1e-9 * std::abs(expected))
          << "calc_exact_coalescent_log_prior, tips=" << num_tips << " exponential=" << exponential;

      auto prior = Coalescent_prior{pop_model.get(), num_tips};
      for (auto t : events.coalescence_times) { prior.displace_coalescence(0.0, t); }
      EXPECT_NEAR(prior.calc_log_prior(), expected, 1e-9 * std::abs(expected))
          << "Coalescent_prior::calc_log_prior, tips=" << num_tips << " exponential=" << exponential;
    }
  }
}

}  // namespace delphy
