#ifndef DELPHY_HUMAN_DATES_H_
#define DELPHY_HUMAN_DATES_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace delphy {

// Sampling-date conventions accepted by the human uniparental input path.  Each metadata row names exactly one
// convention; there is no default and no implicit conversion between conventions.
//
//   iso           Calendar date YYYY-MM-DD, month YYYY-MM or year YYYY (CE).  A month or a year denotes the whole
//                 period, as for the date suffixes of pathogen sequence IDs.
//   ce            Integer calendar year CE (1, 2, ...).
//   bce           Integer calendar year BCE (1 BCE is the year before 1 CE; there is no year zero).
//   bp            Integer calendar-year age before a declared baseline year (bp_reference_year, CE).
//   cal_bp        Integer calibrated radiocarbon age, defined relative to 1950 CE.
//   c14_uncal_bp  Uncalibrated radiocarbon age.  Recognised only so that it can be rejected: it is not a calendar
//                 age and Delphy does not calibrate.
//
// Year-valued conventions (ce, bce, bp, cal_bp) denote the instant 00:00 on 1 January of the corresponding calendar
// year in the proleptic Gregorian calendar, using astronomical year numbering internally (1 BCE == year 0).
enum class Date_convention { iso, ce, bce, bp, cal_bp, c14_uncal_bp };

auto to_string(Date_convention convention) -> std::string_view;

// Throws std::invalid_argument, listing the accepted names, for anything else.
auto parse_date_convention(std::string_view name) -> Date_convention;

// Radiocarbon convention: cal BP counts years before 1 January 1950 CE.
inline constexpr int64_t k_cal_bp_reference_year = 1950;

struct Sample_date_input {
  Date_convention convention = Date_convention::iso;
  std::string from;  // Older bound (or the only value), written in the units of `convention`
  std::string to;    // Younger bound; empty means "no separate upper bound"
  std::optional<int64_t> bp_reference_year;  // Baseline year CE; required for `bp`, optional (1950) for `cal_bp`
};

// Closed interval of possible sampling times, in days since 2020-01-01
struct Time_bounds {
  double t_min;
  double t_max;
};

// Tip date bounds are stored as float32 in Tip_desc / Phylo_node.  Integer-day bounds are exact up to 2^24 days and
// are rounded by at most 1 day up to 2^25 days (~92,000 years from 2020); bounds beyond that are rejected.
inline constexpr double k_max_abs_tip_time_days = 33'554'432.0;  // 2^25

// |t - double(float(t))|
auto float32_rounding_error(double t) -> double;

// Days since 2020-01-01 of 00:00 on 1 January of the given astronomical year (1 BCE == 0, 2 BCE == -1, ...).
auto astronomical_year_to_day(int64_t astronomical_year) -> double;

// Converts a metadata date to Delphy's internal time axis.  Throws std::invalid_argument with a message that says how
// to fix the input when the value is malformed, uses an unsupported or uncalibrated convention, is out of order, or
// cannot be stored with sufficient precision.
auto convert_sample_date(const Sample_date_input& input) -> Time_bounds;

}  // namespace delphy

#endif // DELPHY_HUMAN_DATES_H_
