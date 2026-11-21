#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dates.h"
#include "human_dates.h"

namespace delphy {

namespace {

auto convert(Date_convention c, std::string from, std::string to = "", std::optional<int64_t> ref = {})
    -> Time_bounds {
  return convert_sample_date({.convention = c, .from = std::move(from), .to = std::move(to), .bp_reference_year = ref});
}

auto expect_rejected(Date_convention c, std::string from, std::string to, std::optional<int64_t> ref,
                     const std::string& message_fragment) -> void {
  try {
    convert(c, from, to, ref);
    ADD_FAILURE() << "expected rejection for " << to_string(c) << " '" << from << "'";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr(message_fragment));
  }
}

}  // namespace

TEST(Human_dates_test, parse_convention_names) {
  EXPECT_EQ(parse_date_convention("iso"), Date_convention::iso);
  EXPECT_EQ(parse_date_convention("ce"), Date_convention::ce);
  EXPECT_EQ(parse_date_convention("bce"), Date_convention::bce);
  EXPECT_EQ(parse_date_convention("bp"), Date_convention::bp);
  EXPECT_EQ(parse_date_convention("cal_bp"), Date_convention::cal_bp);
  EXPECT_EQ(parse_date_convention("c14_uncal_bp"), Date_convention::c14_uncal_bp);
  EXPECT_THROW(parse_date_convention("BP"), std::invalid_argument);
  EXPECT_THROW(parse_date_convention("AD"), std::invalid_argument);
  EXPECT_THROW(parse_date_convention(""), std::invalid_argument);
}

TEST(Human_dates_test, iso_matches_existing_parsers) {
  auto day = convert(Date_convention::iso, "2015-06-01");
  EXPECT_EQ(day.t_min, parse_iso_date("2015-06-01"));
  EXPECT_EQ(day.t_max, parse_iso_date("2015-06-01"));

  auto month = convert(Date_convention::iso, "2015-06");
  EXPECT_EQ(month.t_min, parse_iso_date("2015-06-01"));
  EXPECT_EQ(month.t_max, parse_iso_date("2015-07-01"));

  auto year = convert(Date_convention::iso, "2015");
  EXPECT_EQ(year.t_min, parse_iso_date("2015-01-01"));
  EXPECT_EQ(year.t_max, parse_iso_date("2016-01-01"));

  auto range = convert(Date_convention::iso, "2015-06-01", "2015-08-31");
  EXPECT_EQ(range.t_min, parse_iso_date("2015-06-01"));
  EXPECT_EQ(range.t_max, parse_iso_date("2015-08-31"));

  auto month_to_month = convert(Date_convention::iso, "2015-06", "2015-08");
  EXPECT_EQ(month_to_month.t_min, parse_iso_date("2015-06-01"));
  EXPECT_EQ(month_to_month.t_max, parse_iso_date("2015-09-01"));
}

TEST(Human_dates_test, iso_rejections) {
  expect_rejected(Date_convention::iso, "2015-13-01", "", {}, "not a valid calendar date");
  expect_rejected(Date_convention::iso, "2015-02-30", "", {}, "not a valid calendar date");
  expect_rejected(Date_convention::iso, "15-06-01", "", {}, "not an ISO date");
  expect_rejected(Date_convention::iso, "2015/06/01", "", {}, "not an ISO date");
  expect_rejected(Date_convention::iso, "0000-01-01", "", {}, "year 0000");
  expect_rejected(Date_convention::iso, "-0100-01-01", "", {}, "not an ISO date");
  expect_rejected(Date_convention::iso, "2015-06-01", "2015-05-01", {}, "more recent than date_to");
  expect_rejected(Date_convention::iso, "2015", "", 1950, "bp_reference_year must be empty");
  expect_rejected(Date_convention::iso, "", "", {}, "date_from is empty");
}

TEST(Human_dates_test, calendar_year_anchors) {
  EXPECT_EQ(convert(Date_convention::ce, "2020").t_min, 0.0);
  EXPECT_EQ(convert(Date_convention::ce, "2019").t_min, -365.0);
  EXPECT_EQ(convert(Date_convention::ce, "2021").t_min, 366.0);
  EXPECT_EQ(convert(Date_convention::ce, "2015").t_min, parse_iso_date("2015-01-01"));
  EXPECT_EQ(convert(Date_convention::ce, "1000").t_min, parse_iso_date("1000-01-01"));
}

TEST(Human_dates_test, no_year_zero_between_bce_and_ce) {
  // 1 BCE is astronomical year 0, a leap year in the proleptic Gregorian calendar (divisible by 400)
  auto ce1 = convert(Date_convention::ce, "1").t_min;
  auto bce1 = convert(Date_convention::bce, "1").t_min;
  auto bce2 = convert(Date_convention::bce, "2").t_min;
  EXPECT_EQ(ce1 - bce1, 366.0);
  EXPECT_EQ(bce1 - bce2, 365.0);  // astronomical year -1 is not a leap year

  expect_rejected(Date_convention::ce, "0", "", {}, "CE years start at 1");
  expect_rejected(Date_convention::ce, "-5", "", {}, "CE years start at 1");
  expect_rejected(Date_convention::bce, "0", "", {}, "there is no year 0");
  expect_rejected(Date_convention::bce, "-5", "", {}, "there is no year 0");
}

TEST(Human_dates_test, bce_range_orders_older_first) {
  auto r = convert(Date_convention::bce, "3000", "2500");
  EXPECT_EQ(r.t_min, astronomical_year_to_day(1 - 3000));
  EXPECT_EQ(r.t_max, astronomical_year_to_day(1 - 2500));
  EXPECT_LT(r.t_min, r.t_max);
  expect_rejected(Date_convention::bce, "2500", "3000", {}, "date_from must be the older bound");
}

TEST(Human_dates_test, bp_requires_and_uses_declared_baseline) {
  expect_rejected(Date_convention::bp, "3000", "", {}, "requires bp_reference_year");
  expect_rejected(Date_convention::bp, "3000", "", 0, "not a CE year");

  // 0 BP with baseline 1950 is 1 January 1950
  EXPECT_EQ(convert(Date_convention::bp, "0", "", 1950).t_min, parse_iso_date("1950-01-01"));
  // The baseline is applied, not assumed: the same number with another baseline is a different time
  auto b1950 = convert(Date_convention::bp, "3000", "", 1950).t_min;
  auto b2000 = convert(Date_convention::bp, "3000", "", 2000).t_min;
  EXPECT_EQ(b1950, astronomical_year_to_day(1950 - 3000));
  EXPECT_EQ(b2000, astronomical_year_to_day(2000 - 3000));
  EXPECT_NE(b1950, b2000);

  // BP range: the older bound is the larger number
  auto r = convert(Date_convention::bp, "3200", "3100", 1950);
  EXPECT_LT(r.t_min, r.t_max);
  expect_rejected(Date_convention::bp, "3100", "3200", 1950, "LARGER number");
  expect_rejected(Date_convention::bp, "-1", "", 1950, "negative");
}

TEST(Human_dates_test, cal_bp_is_relative_to_1950) {
  auto cal = convert(Date_convention::cal_bp, "3200");
  EXPECT_EQ(cal.t_min, astronomical_year_to_day(1950 - 3200));
  EXPECT_EQ(convert(Date_convention::cal_bp, "3200", "", 1950).t_min, cal.t_min);
  EXPECT_EQ(convert(Date_convention::cal_bp, "3200").t_min, convert(Date_convention::bp, "3200", "", 1950).t_min);
  expect_rejected(Date_convention::cal_bp, "3200", "", 2000, "defined relative to 1950");
}

TEST(Human_dates_test, uncalibrated_radiocarbon_is_rejected_not_converted) {
  expect_rejected(Date_convention::c14_uncal_bp, "3200", "", {}, "uncalibrated radiocarbon");
  expect_rejected(Date_convention::c14_uncal_bp, "3200", "3100", 1950, "Calibrate it externally");
}

TEST(Human_dates_test, malformed_year_values) {
  expect_rejected(Date_convention::cal_bp, "3200.5", "", {}, "not an integer");
  expect_rejected(Date_convention::cal_bp, "abc", "", {}, "not an integer");
  expect_rejected(Date_convention::bce, "99999999999", "", {}, "out of range");
  expect_rejected(Date_convention::ce, "1", "2", 1950, "bp_reference_year must be empty");
}

TEST(Human_dates_test, float32_rounding_error_values) {
  EXPECT_EQ(float32_rounding_error(0.0), 0.0);
  EXPECT_EQ(float32_rounding_error(1500.0), 0.0);
  EXPECT_EQ(float32_rounding_error(-16'777'216.0), 0.0);       // 2^24 is exactly representable
  EXPECT_EQ(float32_rounding_error(16'777'217.0), 1.0);        // first odd integer lost
  EXPECT_EQ(float32_rounding_error(-3'650'000.25), 0.0);       // quarter-day resolution at ~10,000 years
  EXPECT_NEAR(float32_rounding_error(-3'650'000.1), 0.1, 1e-9);   // rounds to the nearest 0.25
}

TEST(Human_dates_test, float32_limit_is_enforced) {
  // Integer-day bounds are exact up to 2^24 days (~46,000 years) and within 1 day up to 2^25 (~92,000 years)
  auto old_enough = convert(Date_convention::cal_bp, "45000");
  EXPECT_EQ(float32_rounding_error(old_enough.t_min), 0.0);
  for (auto years_bp : {50000, 60000, 90000}) {
    SCOPED_TRACE(years_bp);
    for (auto offset = 0; offset != 4; ++offset) {  // the rule must not depend on the particular day
      auto b = convert(Date_convention::cal_bp, std::to_string(years_bp + offset));
      EXPECT_LE(float32_rounding_error(b.t_min), 1.0);
    }
  }
  expect_rejected(Date_convention::cal_bp, "92000", "", {}, "float32");
  expect_rejected(Date_convention::cal_bp, "100000", "", {}, "float32");
  expect_rejected(Date_convention::cal_bp, "100001", "", {}, "float32");
  expect_rejected(Date_convention::bce, "100000", "", {}, "float32");
  expect_rejected(Date_convention::cal_bp, "95000", "90000", {}, "float32");
}

TEST(Human_dates_test, deep_time_round_trips_through_existing_output_functions) {
  for (auto year : {int64_t{-48'000}, int64_t{-10'000}, int64_t{-1250}, int64_t{0}, int64_t{1}, int64_t{1850}}) {
    SCOPED_TRACE(year);
    auto t = astronomical_year_to_day(year);
    auto iso = to_iso_date(t);
    EXPECT_THAT(iso, testing::EndsWith("-01-01"));
    EXPECT_NEAR(to_linear_year(t), static_cast<double>(year), 1e-9);
    // One day later is still in the same year
    EXPECT_NEAR(to_linear_year(t + 1.0), static_cast<double>(year) + 1.0 / (year % 4 == 0 ? 366.0 : 365.0), 0.01);
  }
}

TEST(Human_dates_test, uniform_year_span_day_counts_are_consistent) {
  // 400 Gregorian years always contain 146097 days, anywhere on the time axis
  for (auto start : {int64_t{-40'000}, int64_t{-1200}, int64_t{1600}}) {
    EXPECT_EQ(astronomical_year_to_day(start + 400) - astronomical_year_to_day(start), 146097.0);
  }
}

}  // namespace delphy
