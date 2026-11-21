#include "gtest/gtest.h"

#include <sstream>
#include <string>
#include <vector>

#include "newick.h"

namespace delphy {

namespace {

auto parse(const std::string& s) -> Newick_tree {
  auto is = std::istringstream{s};
  auto parser = Newick_parser{is};
  return parser.parse_tree();
}

auto tip_names(const Newick_tree& tree) -> std::vector<std::string> {
  auto result = std::vector<std::string>{};
  for (auto node = Node_index{0}; node != tree.size(); ++node) {
    if (tree.at(node).is_tip()) { result.push_back(tree.at(node).name); }
  }
  return result;
}

auto serialize_node(const Newick_tree& tree, Node_index node) -> std::string {
  auto result = std::string{};
  const auto& n = tree.at(node);
  if (n.is_inner_node()) {
    result += "(";
    for (auto i = 0; i != std::ssize(n.children); ++i) {
      if (i > 0) { result += ","; }
      result += serialize_node(tree, n.children[i]);
    }
    result += ")";
  }
  if (not n.name.empty()) { result += newick_label(n.name); }
  if (n.has_branch_length) {
    auto ss = std::ostringstream{};
    ss.precision(17);
    ss << n.branch_length;
    result += ":" + ss.str();
  }
  return result;
}

auto serialize(const Newick_tree& tree) -> std::string { return serialize_node(tree, tree.root) + ";"; }

auto expect_same_tree(const Newick_tree& a, const Newick_tree& b) -> void {
  ASSERT_EQ(a.size(), b.size());
  EXPECT_EQ(a.root, b.root);
  for (auto node = Node_index{0}; node != a.size(); ++node) {
    EXPECT_EQ(a.at(node).name, b.at(node).name);
    EXPECT_EQ(a.at(node).parent, b.at(node).parent);
    EXPECT_EQ(a.at(node).children, b.at(node).children);
    EXPECT_EQ(a.at(node).has_branch_length, b.at(node).has_branch_length);
    EXPECT_EQ(a.at(node).branch_length, b.at(node).branch_length);
  }
}

const auto k_tricky_names = std::vector<std::string>{
    "123", "007", "-5", "1e5", "2A", "12_34-56", "3.14x",
    "A@B", "semi;colon", "dq\"uote", "it's", "''", "'", "a''b", "tilde~bang!q?",
    "with space", " leading", "trailing ", "(paren)", "a,b", "x:y", "[bracket]", "tab\there",
    "Ünïcødé-样本-🧬", "plain_name", "hCoV-19/USA/MA-1/2020|EPI_ISL_1|2020-03-01",
    std::string(5000, 'Z') + "'" + std::string(5000, ';')};

}  // namespace

TEST(Newick_test, quoted_label_is_consumed_before_branch_length) {
  auto tree = parse("('A B':1.5,'C':2)'root';");
  EXPECT_EQ(tip_names(tree), (std::vector<std::string>{"A B", "C"}));
  EXPECT_EQ(tree.at(tree.root).name, "root");
  EXPECT_DOUBLE_EQ(tree.at(tree.at(tree.root).children[0]).branch_length, 1.5);
  EXPECT_DOUBLE_EQ(tree.at(tree.at(tree.root).children[1]).branch_length, 2.0);
}

TEST(Newick_test, numeric_and_digit_leading_labels) {
  auto tree = parse("((123:1,007:2):0.5,(2A:1,1e5:1e-1):0.25,-5:3);");
  EXPECT_EQ(tip_names(tree), (std::vector<std::string>{"123", "007", "2A", "1e5", "-5"}));
  const auto& c = tree.at(tree.root).children;
  EXPECT_DOUBLE_EQ(tree.at(c[1]).branch_length, 0.25);
  EXPECT_DOUBLE_EQ(tree.at(tree.at(c[1]).children[1]).branch_length, 0.1);
}

TEST(Newick_test, semicolon_inside_quotes_does_not_end_the_tree) {
  auto tree = parse("('a;b':1,'c;':2);");
  EXPECT_EQ(tip_names(tree), (std::vector<std::string>{"a;b", "c;"}));
}

TEST(Newick_test, quote_escaping_and_empty_quoted_label) {
  auto tree = parse("('it''s':1,'''':1,'':1,'a''''b':1);");
  EXPECT_EQ(tip_names(tree), (std::vector<std::string>{"it's", "'", "", "a''b"}));
}

TEST(Newick_test, newick_label_quotes_only_when_needed) {
  EXPECT_EQ(newick_label("ABC_1"), "ABC_1");
  EXPECT_EQ(newick_label("123"), "123");
  EXPECT_EQ(newick_label("A@B~!?\""), "A@B~!?\"");
  EXPECT_EQ(newick_label(""), "''");
  EXPECT_EQ(newick_label("a b"), "'a b'");
  EXPECT_EQ(newick_label("it's"), "'it''s'");
  EXPECT_EQ(newick_label("a;b"), "'a;b'");
}

TEST(Newick_test, labels_round_trip) {
  for (const auto& name : k_tricky_names) {
    EXPECT_EQ(parse_newick_label(newick_label(name)), name) << name;
  }
  EXPECT_EQ(parse_newick_label(newick_label("")), "");
}

TEST(Newick_test, parse_serialize_parse_preserves_ids_topology_and_lengths) {
  auto text = std::string{"("};
  for (auto i = 0; i != std::ssize(k_tricky_names); ++i) {
    if (i > 0) { text += ","; }
    text += "(" + newick_label(k_tricky_names[i]) + ":" + std::to_string(0.125 * (i + 1)) + ",'x" +
            std::to_string(i) + "':1e-3)";
  }
  text += ")'root node';";
  auto first = parse(text);
  auto names = tip_names(first);
  for (auto i = 0; i != std::ssize(k_tricky_names); ++i) { EXPECT_EQ(names[2 * i], k_tricky_names[i]); }
  auto second = parse(serialize(first));
  expect_same_tree(first, second);
  EXPECT_EQ(serialize(second), serialize(first));
}

TEST(Newick_test, plain_trees_are_unchanged) {
  auto text = std::string{"((A:1,B:2)[&x=1]:0.5,C_d/e|2020-01-01:3);"};
  auto tree = parse(text);
  EXPECT_EQ(tip_names(tree), (std::vector<std::string>{"A", "B", "C_d/e|2020-01-01"}));
  EXPECT_EQ(serialize(tree), "((A:1,B:2):0.5,C_d/e|2020-01-01:3);");
}

TEST(Newick_test, malformed_input_throws) {
  for (auto bad : {"('unterminated:1,B:1);", "(A:1,B:1)", "(A:1,B:1", "(A:1,B:x);",
                   "(A:1,'B''':1'C');", "(A B:1,C:1);", "(A:1,B:1)];", "(A:1,B:1.);", ""}) {
    EXPECT_THROW(parse(bad), std::runtime_error) << bad;
  }
  for (auto bad : {"'abc", "a b", "'a'b", "", "a;"}) {
    EXPECT_THROW(parse_newick_label(bad), std::runtime_error) << bad;
  }
}

TEST(Newick_test, unquoted_label_at_end_of_input_terminates) {
  EXPECT_THROW(parse("(A,B)C"), std::runtime_error);
  EXPECT_EQ(parse_newick_label("abc"), "abc");
  EXPECT_EQ(parse_newick_label("42"), "42");
}

}  // namespace delphy
