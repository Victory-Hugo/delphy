#ifndef DELPHY_VERSION_H_
#define DELPHY_VERSION_H_

#include <string>

namespace delphy {

extern const std::string k_delphy_version_string;
extern const int k_delphy_build_number;
extern const std::string k_delphy_commit_string;

// Delphy-Extended release label and the upstream Delphy commit it extends.  Independent of the
// upstream version above, which is what output files record.
extern const std::string k_delphy_extended_version_string;
extern const std::string k_delphy_extended_upstream_commit;

}  // namespace delphy

#endif // DELPHY_VERSION_H_
