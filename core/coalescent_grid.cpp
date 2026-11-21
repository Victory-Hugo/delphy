#include "coalescent_grid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <absl/log/check.h>
#include <absl/strings/str_format.h>

namespace delphy {

auto merged_lineage_changes(const Coalescent_event_times& events) -> std::vector<Lineage_change> {
  auto changes = std::vector<Lineage_change>{};
  changes.reserve(std::size(events.tip_times) + std::size(events.coalescence_times));
  for (auto t : events.coalescence_times) { changes.push_back({.t = t, .delta_k = +1}); }
  for (auto t : events.tip_times) { changes.push_back({.t = t, .delta_k = -1}); }

  // Coalescences sort before tips at the same time: a lineage must exist before it can be sampled,
  // which is the same tie-breaking rule as in `Coalescent_prior::calc_log_prior`.
  std::ranges::sort(changes, [](const Lineage_change& a, const Lineage_change& b) {
    return a.t != b.t ? a.t < b.t : a.delta_k > b.delta_k;
  });
  return changes;
}

namespace {

// Number of lineages just after every change at or before `t`.
auto lineages_at(const std::vector<Lineage_change>& changes, double t) -> int {
  auto k = 1;
  for (const auto& change : changes) {
    if (change.t > t) { break; }
    k += change.delta_k;
  }
  return k;
}

// `Scalable_coalescent_prior::ensure_space` substitutes this for a cell population size that
// underflows to zero, so anything that stands in for the production cell prior must do the same.
inline constexpr auto k_popsize_bar_floor = 1e-100;

// Per-cell quantities both sides of the comparison need.
struct Cell_moments {
  double kbar = 0.0;             // <k>_c
  double k2bar = 0.0;            // <k^2>_c
  double exact_integral = 0.0;   // int_c dt (k^2 - k) / N
  double intensity = 0.0;        // J_c = int_c dt / N
  double popsize_bar = 0.0;      // Nbar_c
};

// Walks the events inside [a, b), accumulating the moments of k(t) and the exact rate integral.
// `k` is the lineage count entering the cell and is advanced past every change inside it.
auto cell_moments(
    const std::vector<Lineage_change>& changes,
    std::size_t& next_change,
    int& k,
    const Pop_model& pop_model,
    double a,
    double b)
    -> Cell_moments {

  auto t_step = b - a;
  auto result = Cell_moments{};
  auto m1 = 0.0;
  auto m2 = 0.0;
  auto prev = a;
  while (next_change != std::size(changes) && changes[next_change].t < b) {
    auto t = std::max(a, changes[next_change].t);
    m1 += k * (t - prev);
    m2 += static_cast<double>(k) * k * (t - prev);
    result.exact_integral += static_cast<double>(k) * (k - 1) * pop_model.intensity_integral(prev, t);
    prev = t;
    k += changes[next_change].delta_k;
    ++next_change;
  }
  m1 += k * (b - prev);
  m2 += static_cast<double>(k) * k * (b - prev);
  result.exact_integral += static_cast<double>(k) * (k - 1) * pop_model.intensity_integral(prev, b);

  result.kbar = m1 / t_step;
  result.k2bar = m2 / t_step;
  result.intensity = pop_model.intensity_integral(a, b);
  result.popsize_bar = pop_model.pop_integral(a, b) / t_step;
  return result;
}

// Index of the cell holding the earliest event, as a negative offset from the cell that starts at
// `t_ref`.  Cell c covers [t_ref + c t_step, t_ref + (c+1) t_step).
auto first_cell_of(const std::vector<Lineage_change>& changes, double t_ref, double t_step)
    -> std::int64_t {
  return static_cast<std::int64_t>(std::floor((changes.front().t - t_ref) / t_step));
}

}  // namespace

auto coalescent_event_times_of(const Phylo_tree& tree) -> Coalescent_event_times {
  auto result = Coalescent_event_times{};
  for (const auto& node : index_order_traversal(tree)) {
    if (tree.at(node).is_tip()) {
      result.tip_times.push_back(tree.at(node).t);
    } else {
      result.coalescence_times.push_back(tree.at(node).t);
    }
  }
  return result;
}

auto calc_coalescent_grid_excess(
    const Coalescent_event_times& events,
    const Pop_model& pop_model,
    double t_ref,
    double t_step)
    -> Coalescent_grid_excess {

  if (not (t_step > 0.0)) {
    throw std::invalid_argument(absl::StrFormat("Coalescent grid cell width must be positive, not %g", t_step));
  }

  auto result = Coalescent_grid_excess{.t_step = t_step};
  auto changes = merged_lineage_changes(events);
  if (changes.empty()) { return result; }

  auto first_cell = first_cell_of(changes, t_ref, t_step);
  result.num_cells = static_cast<int>(-first_cell);

  auto next_change = std::size_t{0};
  auto k = 1;  // Above the root there is one lineage, and it coalesces with nothing
  for (auto cell = first_cell; cell < 0; ++cell) {
    auto a = t_ref + static_cast<double>(cell) * t_step;
    auto m = cell_moments(changes, next_change, k, pop_model, a, a + t_step);
    if (not (m.popsize_bar > 0.0) || not std::isfinite(m.exact_integral)
        || not std::isfinite(m.intensity)) {
      ++result.num_degenerate_cells;
      continue;
    }

    auto cells_c = 0.5 * t_step * (m.kbar * m.kbar - m.kbar) / m.popsize_bar;
    auto exact_c = 0.5 * m.exact_integral;
    auto cell_excess = exact_c - cells_c;  // log P = -integral, so the excess flips the sign

    result.excess += cell_excess;
    result.variance_term += 0.5 * m.intensity * std::max(0.0, m.k2bar - m.kbar * m.kbar);
    if (std::abs(cell_excess) > std::abs(result.worst_cell_excess)) {
      result.worst_cell_excess = cell_excess;
      result.worst_cell_t_lo = a;
    }
  }

  return result;
}

auto calc_exact_coalescent_log_prior(
    const Coalescent_event_times& events,
    const Pop_model& pop_model)
    -> double {

  auto changes = merged_lineage_changes(events);
  auto result = 0.0;
  auto k = 1;
  auto prev = 0.0;
  auto have_prev = false;
  for (const auto& change : changes) {
    if (have_prev && k >= 2) {
      result -= 0.5 * static_cast<double>(k) * (k - 1) * pop_model.intensity_integral(prev, change.t);
    }
    prev = change.t;
    have_prev = true;
    k += change.delta_k;
    if (change.delta_k > 0) {
      result -= std::log(pop_model.pop_at_time(change.t));
    }
  }
  return result;
}

auto calc_cell_coalescent_log_prior(
    const Coalescent_event_times& events,
    const Pop_model& pop_model,
    double t_ref,
    double t_step)
    -> double {

  if (not (t_step > 0.0)) {
    throw std::invalid_argument(absl::StrFormat("Coalescent grid cell width must be positive, not %g", t_step));
  }

  auto changes = merged_lineage_changes(events);
  auto result = 0.0;
  for (auto t : events.coalescence_times) {
    result -= std::log(pop_model.pop_at_time(t));
  }
  if (changes.empty()) { return result; }

  auto next_change = std::size_t{0};
  auto k = 1;
  for (auto cell = first_cell_of(changes, t_ref, t_step); cell < 0; ++cell) {
    auto a = t_ref + static_cast<double>(cell) * t_step;
    auto m = cell_moments(changes, next_change, k, pop_model, a, a + t_step);
    auto popsize_bar = m.popsize_bar > 0.0 ? m.popsize_bar : k_popsize_bar_floor;
    result -= 0.5 * t_step * (m.kbar * m.kbar - m.kbar) / popsize_bar;
  }
  return result;
}

auto calc_exact_delta_log_prior_after_displace_coalescence(
    const Coalescent_event_times& events,
    const Pop_model& pop_model,
    double old_t,
    double new_t)
    -> double {

  if (old_t == new_t) { return 0.0; }

  auto changes = merged_lineage_changes(events);
  auto min_t = std::min(old_t, new_t);
  auto max_t = std::max(old_t, new_t);
  auto adding_lineages = new_t < old_t;

  // The change in the integral term is
  //
  //   -/+ int_{min_t}^{max_t} dt k_lo(t) / N(t),
  //
  // where k_lo(t) is the lineage count over the crossed interval with the moved coalescence at the
  // *later* of its two positions, because C(k+1,2) - C(k,2) = k.  Starting the walk strictly below
  // `min_t` and skipping every change at `min_t` gives exactly k_lo: when the node moves back in
  // time `min_t` is the new position and carries no event yet, and when it moves forward `min_t` is
  // the old position, whose own +1 is then correctly left out.  Distinct nodes are assumed not to
  // share a time, which holds for every genealogy the sampler proposes.
  auto integral = 0.0;
  auto k = lineages_at(changes, std::nextafter(min_t, -std::numeric_limits<double>::infinity()));
  auto prev = min_t;
  for (const auto& change : changes) {
    if (change.t <= min_t) { continue; }
    if (change.t >= max_t) { break; }
    integral += k * pop_model.intensity_integral(prev, change.t);
    prev = change.t;
    k += change.delta_k;
  }
  integral += k * pop_model.intensity_integral(prev, max_t);

  auto delta = (adding_lineages ? -1.0 : +1.0) * integral;
  delta -= std::log(pop_model.pop_at_time(new_t) / pop_model.pop_at_time(old_t));
  return delta;
}

auto choose_coalescent_grid_t_step(
    const Coalescent_event_times& events,
    const Pop_model& pop_model,
    double t_ref,
    double tolerance_nats,
    int min_cells,
    int max_cells)
    -> Coalescent_grid_excess {

  if (not (tolerance_nats > 0.0)) {
    throw std::invalid_argument(absl::StrFormat(
        "Coalescent grid tolerance must be positive, not %g", tolerance_nats));
  }
  if (not (0 < min_cells && min_cells <= max_cells)) {
    throw std::invalid_argument(absl::StrFormat(
        "Coalescent grid cell counts must satisfy 0 < min_cells <= max_cells, but min_cells=%d and max_cells=%d",
        min_cells, max_cells));
  }

  auto changes = merged_lineage_changes(events);
  CHECK(not changes.empty());
  auto span = t_ref - changes.front().t;
  if (not (span > 0.0)) {
    // Degenerate genealogy (every node at the same time): any positive cell width is exact
    return Coalescent_grid_excess{.t_step = 1.0, .num_cells = 1};
  }

  auto best = Coalescent_grid_excess{};
  for (auto cells = min_cells; ; cells = std::min(2 * cells, max_cells)) {
    best = calc_coalescent_grid_excess(events, pop_model, t_ref, span / cells);
    // Refining past a degenerate population size would only chase a number that is not a
    // measurement; stop and let the caller report it.
    if (std::abs(best.excess) <= tolerance_nats || best.num_degenerate_cells > 0
        || cells == max_cells) {
      break;
    }
  }
  return best;
}

}  // namespace delphy
