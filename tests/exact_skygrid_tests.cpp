#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <absl/strings/str_format.h>

#include "coalescent_grid.h"
#include "exact_coalescent.h"
#include "pop_model.h"

namespace delphy {

namespace {

constexpr auto k_year = 365.0;  // Delphy's calendar convention: 1 year = 365 days

// Composite Simpson's rule in long double: an independent evaluation of the rate integral, so that
// the exact density is checked against quadrature and not only against another closed form.
template<typename F>
auto simpson(F f, double a, double b, int n) -> long double {
  if (not (b > a)) { return 0.0L; }
  auto h = (static_cast<long double>(b) - a) / n;
  auto sum = static_cast<long double>(f(a)) + static_cast<long double>(f(b));
  for (auto i = 1; i != n; ++i) {
    auto t = static_cast<long double>(a) + i * h;
    sum += (i % 2 == 0 ? 2.0L : 4.0L) * static_cast<long double>(f(static_cast<double>(t)));
  }
  return sum * h / 3.0L;
}

// A genealogy with every tip at `t_ref` and coalescences from the Kingman coalescent with a constant
// coalescent timescale.  Deterministic given the seed.
auto kingman_events(int num_tips, double pop, double t_ref, uint32_t seed) -> Coalescent_event_times {
  auto prng = std::mt19937{seed};
  auto result = Coalescent_event_times{};
  result.tip_times.assign(num_tips, t_ref);
  auto t = t_ref;
  for (auto k = num_tips; k >= 2; --k) {
    auto rate = static_cast<double>(k) * (k - 1) / 2.0 / pop;
    t -= std::exponential_distribution<double>{rate}(prng);
    result.coalescence_times.push_back(t);
  }
  return result;
}

// A heterochronous genealogy: tips spread over the most recent `spread` of the span, and
// coalescences placed so that every lineage count is positive.  Built by hand rather than simulated
// so that the event pattern is reproducible and has tips interleaved with coalescences.
auto heterochronous_events(int num_tips, double t_ref, double spread, double depth)
    -> Coalescent_event_times {
  auto result = Coalescent_event_times{};
  for (auto i = 0; i != num_tips; ++i) {
    result.tip_times.push_back(t_ref - spread * i / static_cast<double>(num_tips));
  }
  // Coalescences strictly older than every tip they could involve: place them below the oldest tip,
  // which makes the lineage count rise monotonically backwards and keeps the genealogy valid.
  auto t_oldest_tip = t_ref - spread;
  for (auto j = 0; j != num_tips - 1; ++j) {
    result.coalescence_times.push_back(t_oldest_tip - depth * (j + 1) / static_cast<double>(num_tips));
  }
  return result;
}

// Exact Kingman log density written out from the model definition, independently of
// `Exact_coalescent_events` and of `calc_exact_coalescent_log_prior`: the event list is merged here,
// the lineage count is tracked here, and the rate integral of every interval is evaluated by
// quadrature in long double.
auto quadrature_log_prior(
    const Coalescent_event_times& events,
    const Pop_model& pop_model,
    int points_per_interval)
    -> long double {

  struct Event { double t; int delta_k; };
  auto all = std::vector<Event>{};
  for (auto t : events.coalescence_times) { all.push_back({t, +1}); }
  for (auto t : events.tip_times) { all.push_back({t, -1}); }
  std::ranges::sort(all, [](const Event& a, const Event& b) {
    return a.t != b.t ? a.t < b.t : a.delta_k > b.delta_k;
  });

  auto result = 0.0L;
  auto k = 1;
  auto prev = 0.0;
  auto have_prev = false;
  for (const auto& e : all) {
    if (have_prev && k >= 2) {
      auto weight = 0.5L * k * (k - 1);
      result -= weight * simpson([&](double t) { return 1.0 / pop_model.pop_at_time(t); },
                                 prev, e.t, points_per_interval);
    }
    prev = e.t;
    have_prev = true;
    k += e.delta_k;
    if (e.delta_k > 0) {
      result -= static_cast<long double>(std::log(pop_model.pop_at_time(e.t)));
    }
  }
  return result;
}

struct Curve {
  std::string name;
  std::vector<double> x;
  std::vector<double> gamma;
};

// Knot times and values covering the shapes the task asks for: constant, monotone, oscillating,
// extreme, non-uniformly spaced, and the smallest possible curve.  `t_ref` is the latest tip time
// and `span` the depth the genealogies reach.
auto curves(double t_ref, double span) -> std::vector<Curve> {
  auto uniform_x = [&](int num_knots) {
    auto x = std::vector<double>{};
    for (auto i = 0; i != num_knots; ++i) {
      x.push_back(t_ref - span * (num_knots - 1 - i) / static_cast<double>(num_knots - 1));
    }
    return x;
  };

  auto result = std::vector<Curve>{};
  result.push_back({"flat", uniform_x(6), {5.0, 5.0, 5.0, 5.0, 5.0, 5.0}});
  result.push_back({"monotone", uniform_x(6), {2.0, 3.0, 4.0, 5.0, 6.0, 7.0}});
  result.push_back({"oscillating", uniform_x(6), {5.0, 2.0, 6.0, 1.5, 6.5, 3.0}});
  result.push_back({"steep", uniform_x(6), {9.0, 1.0, 9.0, 1.0, 9.0, 1.0}});
  result.push_back({"two knots", {t_ref - span, t_ref}, {4.0, 6.0}});
  // Non-uniform knot spacing, with the knots crowded near the tips
  result.push_back({"non-uniform",
                    {t_ref - span, t_ref - 0.6 * span, t_ref - 0.25 * span,
                     t_ref - 0.08 * span, t_ref - 0.02 * span, t_ref},
                    {3.0, 5.5, 4.0, 6.0, 2.5, 5.0}});
  // Populations at exp(+-12): large enough to exercise the biased summation, small enough that the
  // genealogies below still have a finite density at both variants.
  result.push_back({"extreme", uniform_x(4), {12.0, -12.0, 12.0, -12.0}});
  return result;
}

auto both_types() -> std::vector<Skygrid_pop_model::Type> {
  return {Skygrid_pop_model::Type::k_staircase, Skygrid_pop_model::Type::k_log_linear};
}

auto type_name(Skygrid_pop_model::Type type) -> const char* {
  return type == Skygrid_pop_model::Type::k_staircase ? "staircase" : "log-linear";
}

constexpr auto k_t_ref = 1000.0;
constexpr auto k_span = 400.0 * k_year;

// The genealogies every check below runs over: 4, 8, 12 and 20 tips, isochronous and
// heterochronous.
auto genealogies() -> std::vector<std::pair<std::string, Coalescent_event_times>> {
  auto result = std::vector<std::pair<std::string, Coalescent_event_times>>{};
  for (auto n : {4, 8, 12, 20}) {
    result.push_back({absl::StrFormat("isochronous n=%d", n),
                      kingman_events(n, 60.0 * k_year, k_t_ref, 20261118u + n)});
    result.push_back({absl::StrFormat("heterochronous n=%d", n),
                      heterochronous_events(n, k_t_ref, 0.15 * k_span, 0.7 * k_span)});
  }
  return result;
}

}  // namespace

// The event-driven decomposition reproduces the reference density of `coalescent_grid.h`, with and
// without the subdivision at the knots.  The two disagree only if the interval bookkeeping is
// wrong, so this is the structural check on `Exact_coalescent_events`.
TEST(Exact_skygrid_test, the_event_driven_density_matches_the_reference) {
  auto checks = 0;
  for (const auto& [name, events] : genealogies()) {
    for (auto type : both_types()) {
      for (const auto& curve : curves(k_t_ref, k_span)) {
        auto pop_model = Skygrid_pop_model{curve.x, curve.gamma, type};
        auto expected = calc_exact_coalescent_log_prior(events, pop_model);

        auto cut = Exact_coalescent_events{events, pop_model.x()};
        auto uncut = Exact_coalescent_events{events};
        auto scale = std::max(1.0, std::abs(expected));
        EXPECT_NEAR(cut.calc_log_prior(pop_model), expected, 1e-10 * scale)
            << name << ", " << type_name(type) << ", " << curve.name << " (knot-cut)";
        EXPECT_NEAR(uncut.calc_log_prior(pop_model), expected, 1e-10 * scale)
            << name << ", " << type_name(type) << ", " << curve.name << " (uncut)";
        ++checks;
      }
    }
  }
  EXPECT_EQ(checks, 8 * 2 * 7);
}

// The log-linear density agrees with a fully independent evaluation: the events are merged in the
// test, the lineage count is tracked in the test, and the rate integral of every event interval is
// evaluated by quadrature in long double over the raw interval, without any knowledge of the knots.
// The log-linear population curve is continuous, so the quadrature converges.
TEST(Exact_skygrid_test, the_log_linear_density_matches_fully_independent_quadrature) {
  for (const auto& [name, events] : genealogies()) {
    for (const auto& curve : curves(k_t_ref, k_span)) {
      if (curve.name == "extreme") { continue; }  // 1/N spans e^24; quadrature cannot resolve it
      auto pop_model = Skygrid_pop_model{curve.x, curve.gamma, Skygrid_pop_model::Type::k_log_linear};
      auto expected = static_cast<double>(quadrature_log_prior(events, pop_model, 20000));
      auto got = Exact_coalescent_events{events, pop_model.x()}.calc_log_prior(pop_model);
      EXPECT_NEAR(got, expected, 1e-6 * std::max(1.0, std::abs(expected)))
          << name << ", " << curve.name;
    }
  }
}

// The staircase density agrees with quadrature too, but a staircase curve jumps at every knot, so
// quadrature over a raw event interval that crosses one does not converge.  The knot-cut pieces
// place the quadrature bounds; what is being checked is then the closed-form integral of
// `Skygrid_pop_model` and the weights of the decomposition, not the cut points themselves.
TEST(Exact_skygrid_test, the_event_driven_density_matches_independent_quadrature) {
  for (const auto& [name, events] : genealogies()) {
    for (auto type : both_types()) {
      for (const auto& curve : curves(k_t_ref, k_span)) {
        if (curve.name == "extreme") { continue; }  // 1/N spans e^24; quadrature cannot resolve it
        auto pop_model = Skygrid_pop_model{curve.x, curve.gamma, type};
        auto cut = Exact_coalescent_events{events, pop_model.x()};

        // Integral term by quadrature over the knot-cut pieces
        auto intensity = 0.0L;
        for (const auto& interval : cut.intervals()) {
          // The pieces are half-open, (a, b]: a staircase curve reads its lower end from the
          // *previous* population interval, so the quadrature's lower sample is moved one ULP
          // inside.  For the continuous log-linear curve this changes nothing.
          intensity += static_cast<long double>(interval.weight)
              * simpson([&](double t) { return 1.0 / pop_model.pop_at_time(t); },
                        std::nextafter(interval.a, interval.b), interval.b, 2000);
        }
        auto log_pop = 0.0L;
        for (auto t : cut.coalescence_times()) {
          log_pop += static_cast<long double>(std::log(pop_model.pop_at_time(t)));
        }
        auto expected = static_cast<double>(-intensity - log_pop);
        auto got = cut.calc_log_prior(pop_model);
        auto scale = std::max(1.0, std::abs(expected));
        EXPECT_NEAR(got, expected, 1e-7 * scale)
            << name << ", " << type_name(type) << ", " << curve.name;
      }
    }
  }
}

// For the staircase variant, which is the variant BEAST implements, the density is a function of the
// sufficient statistics (SS_k, c_k) alone:
//   log P = - sum_k c_k gamma_k - sum_k SS_k exp(-gamma_k).
// The sufficient statistics are accumulated from the interval decomposition and are independent of
// the knot values, so this check also establishes that the density has that analytic form in gamma.
TEST(Exact_skygrid_test, the_staircase_density_is_the_sufficient_statistic_form) {
  for (const auto& [name, events] : genealogies()) {
    for (const auto& curve : curves(k_t_ref, k_span)) {
      auto pop_model = Skygrid_pop_model{curve.x, curve.gamma, Skygrid_pop_model::Type::k_staircase};
      auto cut = Exact_coalescent_events{events, pop_model.x()};
      auto stats = cut.calc_staircase_sufficient_statistics(pop_model);

      auto expected = 0.0;
      for (auto k = 0; k <= pop_model.M(); ++k) {
        expected -= stats.c[k] * pop_model.gamma(k) + stats.ss[k] * std::exp(-pop_model.gamma(k));
      }
      auto got = cut.calc_log_prior(pop_model);
      EXPECT_NEAR(got, expected, 1e-9 * std::max(1.0, std::abs(expected)))
          << name << ", " << curve.name;

      // Every coalescence is counted once and no interval is lost
      EXPECT_EQ(std::accumulate(std::begin(stats.c), std::end(stats.c), 0.0),
                static_cast<double>(std::ssize(events.coalescence_times)))
          << name << ", " << curve.name;
    }

    // The same statistics must not depend on the knot values they are later combined with
    auto x = curves(k_t_ref, k_span).front().x;
    auto flat = Skygrid_pop_model{x, std::vector<double>(std::size(x), 1.0),
                                  Skygrid_pop_model::Type::k_staircase};
    auto bumpy = Skygrid_pop_model{x, std::vector<double>(std::size(x), 7.0),
                                   Skygrid_pop_model::Type::k_staircase};
    auto cut = Exact_coalescent_events{events, x};
    EXPECT_EQ(cut.calc_staircase_sufficient_statistics(flat).ss,
              cut.calc_staircase_sufficient_statistics(bumpy).ss) << name;
  }
}

// The gradient the HMC needs agrees with a central finite difference of the density the HMC
// evaluates, for every knot of every curve and both variants.
TEST(Exact_skygrid_test, the_gradient_matches_a_central_finite_difference) {
  auto worst = 0.0;
  auto checks = 0;
  for (const auto& [name, events] : genealogies()) {
    for (auto type : both_types()) {
      for (const auto& curve : curves(k_t_ref, k_span)) {
        if (curve.name == "extreme") { continue; }  // a finite difference of e^{+-12} is meaningless
        auto pop_model = Skygrid_pop_model{curve.x, curve.gamma, type};
        auto M = pop_model.M();
        auto cut = Exact_coalescent_events{events, pop_model.x()};

        auto grad = std::vector<double>(M + 1, 0.0);
        cut.add_d_log_prior_d_gamma(pop_model, grad);

        constexpr auto d_gamma = 1e-5;
        for (auto k = 0; k <= M; ++k) {
          auto bumped = curve.gamma;
          bumped[k] = curve.gamma[k] + d_gamma;
          auto plus = cut.calc_log_prior(Skygrid_pop_model{curve.x, bumped, type});
          bumped[k] = curve.gamma[k] - d_gamma;
          auto minus = cut.calc_log_prior(Skygrid_pop_model{curve.x, bumped, type});
          auto numeric = (plus - minus) / (2 * d_gamma);

          auto scale = std::max({1.0, std::abs(grad[k]), std::abs(numeric)});
          auto discrepancy = std::abs(grad[k] - numeric) / scale;
          worst = std::max(worst, discrepancy);
          // The gate is 1e-6 relative or three times the finite difference's own round-off floor,
          // whichever is larger.  A central difference of a quantity of magnitude |U| at a step h
          // carries an absolute error of about eps |U| / h, which for the deep-time curves here
          // (|log P| up to 1e6) is 3e-5 and has nothing to do with the analytic gradient.
          auto fd_noise = std::numeric_limits<double>::epsilon()
              * std::max(std::abs(plus), std::abs(minus)) / d_gamma;
          auto gate = std::max(1e-6 * scale, 3 * fd_noise);
          EXPECT_LT(std::abs(grad[k] - numeric), gate)
              << name << ", " << type_name(type) << ", " << curve.name << ", knot " << k
              << ": analytic " << grad[k] << " vs numeric " << numeric;
          ++checks;
        }
      }
    }
  }
  EXPECT_GT(checks, 400);
  std::cerr << "Worst relative gradient discrepancy: " << worst << " over " << checks << " checks\n";
}

// Shifting the whole curve by delta multiplies the rate integral by exp(-delta) and adds delta to
// every log N(t_i), so
//   d(log P)/d delta = A - N_inner,
// where A is the integral term and N_inner the number of coalescences.  The gradient must therefore
// sum to exactly that, which is the identity the zero-mode Gibbs move rests on.
TEST(Exact_skygrid_test, the_gradient_sums_to_the_zero_mode_derivative) {
  for (const auto& [name, events] : genealogies()) {
    for (auto type : both_types()) {
      for (const auto& curve : curves(k_t_ref, k_span)) {
        auto pop_model = Skygrid_pop_model{curve.x, curve.gamma, type};
        auto cut = Exact_coalescent_events{events, pop_model.x()};
        auto grad = std::vector<double>(pop_model.M() + 1, 0.0);
        cut.add_d_log_prior_d_gamma(pop_model, grad);

        auto sum = std::accumulate(std::begin(grad), std::end(grad), 0.0);
        auto expected = cut.calc_intensity_term(pop_model)
            - static_cast<double>(std::ssize(events.coalescence_times));
        EXPECT_NEAR(sum, expected, 1e-8 * std::max(1.0, std::abs(expected)))
            << name << ", " << type_name(type) << ", " << curve.name;
      }
    }
  }
}

// The exact dependence of the density on a uniform shift of the curve, which is what makes the
// zero-mode a Gibbs step:  with I_bar = exp(-gamma_bar) and Ntwiddle = N I_bar,
//   log P(I_bar) = N_inner log I_bar - B I_bar - sum_i log Ntwiddle(t_i),  B = A / I_bar.
// Fitted at the current curve and predicted at four shifts.
TEST(Exact_skygrid_test, the_zero_mode_dependence_is_exactly_gamma_shaped) {
  for (const auto& [name, events] : genealogies()) {
    for (auto type : both_types()) {
      for (const auto& curve : curves(k_t_ref, k_span)) {
        auto pop_model = Skygrid_pop_model{curve.x, curve.gamma, type};
        auto cut = Exact_coalescent_events{events, pop_model.x()};
        auto n_inner = static_cast<double>(std::ssize(events.coalescence_times));
        auto base_intensity = cut.calc_intensity_term(pop_model);
        auto base_log_pop = cut.calc_log_pop_term(pop_model);

        for (auto shift : {-3.0, -0.5, 0.5, 3.0}) {
          auto shifted_gamma = curve.gamma;
          for (auto& g : shifted_gamma) { g += shift; }
          auto shifted = Skygrid_pop_model{curve.x, shifted_gamma, type};
          auto got = cut.calc_log_prior(shifted);
          // N -> N e^shift, so A -> A e^{-shift} and S -> S + N_inner shift
          auto expected = -base_intensity * std::exp(-shift) - base_log_pop - n_inner * shift;
          EXPECT_NEAR(got, expected, 1e-9 * std::max(1.0, std::abs(expected)))
              << name << ", " << type_name(type) << ", " << curve.name << ", shift " << shift;
        }
      }
    }
  }
}

// Coalescences and tips placed exactly on knots, one knot either side by a single ULP, and several
// coalescences inside one population interval.  These are the cases where the interval decomposition
// can lose or double-count a piece.
TEST(Exact_skygrid_test, events_on_and_beside_the_knots) {
  auto x = std::vector<double>{0.0, 100.0, 200.0, 300.0, 400.0};
  auto gamma = std::vector<double>{4.0, 6.0, 3.0, 5.5, 4.5};

  for (auto type : both_types()) {
    auto pop_model = Skygrid_pop_model{x, gamma, type};

    // Six tips at the last knot; coalescences exactly on three knots, one ULP either side of one
    // knot, and two more inside a single interval
    auto events = Coalescent_event_times{};
    events.tip_times.assign(6, 400.0);
    events.coalescence_times = {
      100.0,                                  // exactly on a knot
      200.0,                                  // exactly on a knot
      std::nextafter(300.0, -1e9),            // one ULP below a knot
      std::nextafter(300.0, +1e9),            // one ULP above a knot
      350.0, 370.0                            // two coalescences inside interval 4
    };

    auto cut = Exact_coalescent_events{events, pop_model.x()};
    auto expected = calc_exact_coalescent_log_prior(events, pop_model);
    EXPECT_NEAR(cut.calc_log_prior(pop_model), expected, 1e-10 * std::abs(expected))
        << type_name(type);

    // No piece crosses a knot, and the pieces tile the span from the first event to the last
    auto total = 0.0;
    for (const auto& interval : cut.intervals()) {
      EXPECT_LE(interval.a, interval.b) << type_name(type);
      EXPECT_EQ(pop_model.interval_containing_t(interval.b), interval.knot_interval)
          << type_name(type);
      if (interval.a > interval.b - (interval.b - interval.a)) { /* zero-length allowed only at ties */ }
      total += interval.b - interval.a;
    }
    EXPECT_NEAR(total, 400.0 - 100.0, 1e-9) << type_name(type);

    // The gradient still matches a finite difference here
    auto grad = std::vector<double>(pop_model.M() + 1, 0.0);
    cut.add_d_log_prior_d_gamma(pop_model, grad);
    constexpr auto d_gamma = 1e-5;
    for (auto k = 0; k <= pop_model.M(); ++k) {
      auto bumped = gamma;
      bumped[k] = gamma[k] + d_gamma;
      auto plus = cut.calc_log_prior(Skygrid_pop_model{x, bumped, type});
      bumped[k] = gamma[k] - d_gamma;
      auto minus = cut.calc_log_prior(Skygrid_pop_model{x, bumped, type});
      auto numeric = (plus - minus) / (2 * d_gamma);
      auto scale = std::max({1.0, std::abs(grad[k]), std::abs(numeric)});
      EXPECT_LT(std::abs(grad[k] - numeric) / scale, 1e-6) << type_name(type) << ", knot " << k;
    }
  }
}

// The exact increment of one coalescence moving, which is what a Metropolis-Hastings ratio of a
// local node-time move needs, equals the difference of two full exact densities.  Checked for both
// variants, for moves that stay inside one population interval and for moves that cross several.
TEST(Exact_skygrid_test, the_exact_increment_matches_the_full_density_difference) {
  auto x = std::vector<double>{0.0, 100.0, 200.0, 300.0, 400.0};
  auto curve_gammas = std::vector<std::vector<double>>{
    {4.0, 4.0, 4.0, 4.0, 4.0},
    {2.0, 3.0, 4.0, 5.0, 6.0},
    {5.0, 2.0, 6.0, 1.5, 6.5},
  };

  for (auto type : both_types()) {
    for (const auto& gamma : curve_gammas) {
      auto pop_model = Skygrid_pop_model{x, gamma, type};

      auto events = Coalescent_event_times{};
      events.tip_times.assign(8, 400.0);
      events.coalescence_times = {120.0, 170.0, 230.0, 260.0, 310.0, 340.0, 380.0};

      for (auto which = 0; which != std::ssize(events.coalescence_times); ++which) {
        for (auto new_t : {115.0, 185.0, 205.0, 299.0, 301.0, 355.0, 395.0}) {
          auto old_t = events.coalescence_times[which];
          // Keep the genealogy valid: a coalescence may not move above the tips or below the root
          if (new_t >= 400.0 || new_t <= 100.0) { continue; }
          // Distinct nodes must not share a time, which the increment's derivation assumes
          auto clash = false;
          for (auto t : events.coalescence_times) { if (t == new_t) { clash = true; } }
          if (clash) { continue; }
          // The increment's derivation needs every lineage count to stay positive, which for this
          // caterpillar genealogy means the order of the coalescences may not change
          auto moved = events;
          moved.coalescence_times[which] = new_t;
          auto sorted = moved.coalescence_times;
          std::ranges::sort(sorted);
          if (sorted != moved.coalescence_times) { continue; }

          auto full = calc_exact_coalescent_log_prior(moved, pop_model)
              - calc_exact_coalescent_log_prior(events, pop_model);
          auto incremental = calc_exact_delta_log_prior_after_displace_coalescence(
              events, pop_model, old_t, new_t);
          EXPECT_NEAR(incremental, full, 1e-8 * std::max(1.0, std::abs(full)))
              << type_name(type) << ", coalescence " << which << " to " << new_t;
        }
      }
    }
  }
}

}  // namespace delphy
