#include "human_dates.h"

#include <cmath>
#include <stdexcept>

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/time/civil_time.h"

#include "dates.h"

namespace delphy {

namespace {

// Largest |year| accepted before conversion; keeps all day counts far from int64 overflow.
constexpr auto k_max_abs_year = int64_t{10'000'000};

auto k_epoch_day() -> absl::CivilDay { return absl::CivilDay(2020, 1, 1); }

auto parse_integer(std::string_view what, std::string_view text) -> int64_t {
  auto value = int64_t{};
  if (text.empty() || not absl::SimpleAtoi(text, &value)) {
    throw std::invalid_argument(absl::StrFormat(
        "%s '%s' is not an integer.  Year-valued conventions need whole years (for example 3200); "
        "use the 'iso' convention for calendar dates.",
        what, text));
  }
  if (value > k_max_abs_year || value < -k_max_abs_year) {
    throw std::invalid_argument(absl::StrFormat("%s '%s' is out of range (|value| must be at most %d).",
                                                what, text, k_max_abs_year));
  }
  return value;
}

auto check_float32_storable(double t, std::string_view what) -> void {
  if (std::abs(t) > k_max_abs_tip_time_days) {
    throw std::invalid_argument(absl::StrFormat(
        "%s corresponds to %.0f days since 2020-01-01.  Tip date bounds are stored as float32, which holds whole "
        "days to within 1 day only up to %.0f days (about 92,000 years) from 2020; older dates are not supported "
        "by this version.",
        what, t, k_max_abs_tip_time_days));
  }
}

struct Iso_period {
  double first;
  double last;
};

auto parse_iso_period(std::string_view what, std::string_view text) -> Iso_period {
  auto all_digits_and_dashes = [&] {
    for (auto c : text) {
      if (not (std::isdigit(static_cast<unsigned char>(c)) || c == '-')) { return false; }
    }
    return true;
  }();
  if (not all_digits_and_dashes || (text.size() != 4 && text.size() != 7 && text.size() != 10)) {
    throw std::invalid_argument(absl::StrFormat(
        "%s '%s' is not an ISO date.  Use YYYY-MM-DD, YYYY-MM or YYYY with a 4-digit year from 0001 to 9999; "
        "use the 'ce', 'bce', 'bp' or 'cal_bp' conventions for ancient samples.",
        what, text));
  }
  if (text.substr(0, 4) == "0000") {
    throw std::invalid_argument(absl::StrFormat(
        "%s '%s' uses year 0000.  There is no year zero in the calendar conventions used here; "
        "use convention 'bce' with value 1 for the year before 1 CE.",
        what, text));
  }
  try {
    if (text.size() == 10) {
      auto t = parse_iso_date(text);
      return {t, t};
    } else if (text.size() == 7) {
      auto [first, next] = parse_iso_month(text);
      return {first, next};
    } else {
      auto [first, next] = parse_iso_year(text);
      return {first, next};
    }
  } catch (const std::runtime_error&) {
    throw std::invalid_argument(absl::StrFormat("%s '%s' is not a valid calendar date.", what, text));
  }
}

// Converts a single year-valued number to an astronomical year
auto to_astronomical_year(Date_convention convention, int64_t value, std::optional<int64_t> bp_reference_year,
                          std::string_view what) -> int64_t {
  switch (convention) {
    case Date_convention::ce:
      if (value < 1) {
        throw std::invalid_argument(absl::StrFormat(
            "%s %d is not a valid CE year.  CE years start at 1; use convention 'bce' for earlier years.",
            what, value));
      }
      return value;
    case Date_convention::bce:
      if (value < 1) {
        throw std::invalid_argument(absl::StrFormat(
            "%s %d is not a valid BCE year.  BCE years start at 1 (the year before 1 CE); there is no year 0.",
            what, value));
      }
      return 1 - value;
    case Date_convention::bp:
    case Date_convention::cal_bp:
      if (value < 0) {
        throw std::invalid_argument(absl::StrFormat(
            "%s %d is negative.  BP ages count years before the baseline year and must be >= 0.", what, value));
      }
      return *bp_reference_year - value;
    default:
      break;
  }
  throw std::logic_error("to_astronomical_year(): not a year-valued convention");
}

}  // namespace

auto to_string(Date_convention convention) -> std::string_view {
  switch (convention) {
    case Date_convention::iso: return "iso";
    case Date_convention::ce: return "ce";
    case Date_convention::bce: return "bce";
    case Date_convention::bp: return "bp";
    case Date_convention::cal_bp: return "cal_bp";
    case Date_convention::c14_uncal_bp: return "c14_uncal_bp";
  }
  return "unknown";
}

auto parse_date_convention(std::string_view name) -> Date_convention {
  for (auto c : {Date_convention::iso, Date_convention::ce, Date_convention::bce, Date_convention::bp,
                 Date_convention::cal_bp, Date_convention::c14_uncal_bp}) {
    if (name == to_string(c)) { return c; }
  }
  throw std::invalid_argument(absl::StrFormat(
      "Unknown date convention '%s'.  Accepted values: iso, ce, bce, bp, cal_bp, c14_uncal_bp "
      "(lower case; 'c14_uncal_bp' is recognised only to be rejected).",
      name));
}

auto float32_rounding_error(double t) -> double {
  return std::abs(t - static_cast<double>(static_cast<float>(t)));
}

auto astronomical_year_to_day(int64_t astronomical_year) -> double {
  return static_cast<double>(absl::CivilDay(astronomical_year, 1, 1) - k_epoch_day());
}

auto convert_sample_date(const Sample_date_input& input) -> Time_bounds {
  if (input.from.empty()) {
    throw std::invalid_argument("date_from is empty.  Give the sampling date, or leave date_convention empty "
                                "to use the uniform sampling date.");
  }

  auto result = Time_bounds{};

  if (input.convention == Date_convention::c14_uncal_bp) {
    throw std::invalid_argument(
        "Convention 'c14_uncal_bp' is an uncalibrated radiocarbon age, not a calendar age, and is rejected.  "
        "Calibrate it externally (for example with OxCal or IntCal) and supply the calibrated range with "
        "convention 'cal_bp', date_from = older bound, date_to = younger bound.");
  }

  if (input.convention == Date_convention::iso) {
    if (input.bp_reference_year.has_value()) {
      throw std::invalid_argument("bp_reference_year must be empty for convention 'iso'.");
    }
    auto from = parse_iso_period("date_from", input.from);
    result.t_min = from.first;
    result.t_max = from.last;
    if (not input.to.empty()) {
      result.t_max = parse_iso_period("date_to", input.to).last;
    }
  } else {
    auto reference_year = input.bp_reference_year;
    if (input.convention == Date_convention::bp) {
      if (not reference_year.has_value()) {
        throw std::invalid_argument(
            "Convention 'bp' requires bp_reference_year (the baseline year CE, for example 1950).  "
            "If the ages are calibrated radiocarbon ages, use convention 'cal_bp' instead.");
      }
      if (*reference_year < 1) {
        throw std::invalid_argument(absl::StrFormat(
            "bp_reference_year %d is not a CE year (must be >= 1).", *reference_year));
      }
    } else if (input.convention == Date_convention::cal_bp) {
      if (reference_year.has_value() && *reference_year != k_cal_bp_reference_year) {
        throw std::invalid_argument(absl::StrFormat(
            "Convention 'cal_bp' is defined relative to %d CE, but bp_reference_year is %d.  "
            "Use convention 'bp' to declare a different baseline year.",
            k_cal_bp_reference_year, *reference_year));
      }
      reference_year = k_cal_bp_reference_year;
    } else if (reference_year.has_value()) {
      throw std::invalid_argument(absl::StrFormat(
          "bp_reference_year must be empty for convention '%s'.", to_string(input.convention)));
    }

    auto from_year = to_astronomical_year(
        input.convention, parse_integer("date_from", input.from), reference_year, "date_from");
    result.t_min = astronomical_year_to_day(from_year);
    result.t_max = result.t_min;
    if (not input.to.empty()) {
      auto to_year = to_astronomical_year(
          input.convention, parse_integer("date_to", input.to), reference_year, "date_to");
      result.t_max = astronomical_year_to_day(to_year);
    }
  }

  if (result.t_min > result.t_max) {
    throw std::invalid_argument(absl::StrFormat(
        "date_from (%s) is more recent than date_to (%s).  date_from must be the older bound; for 'bp' and 'cal_bp' "
        "the older bound is the LARGER number.",
        input.from, input.to));
  }

  check_float32_storable(result.t_min, "date_from");
  check_float32_storable(result.t_max, input.to.empty() ? "date_from" : "date_to");
  return result;
}

}  // namespace delphy
