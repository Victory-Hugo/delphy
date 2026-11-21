#ifndef DELPHY_HISTORY_GIBBS_H_
#define DELPHY_HISTORY_GIBBS_H_

#include <array>
#include <span>
#include <vector>

#include "absl/random/bit_gen_ref.h"

#include "ctmc_bridge.h"
#include "evo_model.h"
#include "phylo_tree.h"
#include "sequence_overlay.h"

namespace delphy {

// Gibbs updates of the explicit mutational history of one site at a time.
//
// The existing mutational-history proposal replaces the history of every site of a path at once,
// drawn from a Jukes-Cantor process at the average rate, and corrects the whole proposal with one
// Metropolis-Hastings decision.  Its importance weight is a product over sites, so its acceptance
// falls with the total number of substitution events and the history freezes at high divergence
// (doc/human_uniparental/38_substitution_model_report.md (development record, not in this repository), section 5).
//
// The updates here work on one site at a time instead.  For a single site the full conditional of the
// history, given everything else in the augmented posterior, is a continuous-time Markov bridge under
// that site's own rate matrix, which `Ctmc_bridge_sampler` draws exactly.  Each update is therefore a
// Gibbs step whose acceptance probability is one, and nothing about it degrades as the sequence gets
// longer or more divergent.
//
// The augmented posterior that these updates must leave invariant is the one Delphy already computes:
// the branch genetic prior of `calc_branch_log_G` is the path measure of the site's CTMC, the root
// prior of `calc_log_root_prior` is the product of the stationary frequencies over the root sequence,
// and no other term of the posterior -- the coalescent prior, the calibrations, the parameter priors
// -- depends on where the substitutions sit inside a branch or on the state of an internal node.

// Per-partition bridge samplers for the current parameters.  Rebuild it whenever the evolutionary
// model changes; construction only forms the uniformized matrices and is cheap.
class Site_history_sampler {
 public:
  explicit Site_history_sampler(const Global_evo_model& evo);

  auto evo() const -> const Global_evo_model& { return evo_; }

  // The factor multiplying the partition's rate matrix at site `l`, i.e. mu_l nu_l.
  auto rate_scale_of_site(Site_index l) const -> double;

  // exp(Q_l T), where Q_l is the effective rate matrix of site `l`.
  auto calc_transition_probs(Site_index l, double T) const -> Seq_matrix<double>;

  // Appends to `out` the substitution events of an exact draw from the conditional history of site
  // `l` across a branch of duration `T` that starts in state `a` and ends in state `b`.  Event times
  // are offsets into the branch, in (0, T).
  auto sample_bridge(
      Site_index l,
      Real_seq_letter a,
      Real_seq_letter b,
      double T,
      absl::BitGenRef bitgen,
      std::vector<Ctmc_bridge_event>& out)
      -> void;

 private:
  // Held by value: the samplers outlive any particular `Global_evo_model` object, and a `Subrun` that
  // owns one can be moved inside a vector.
  Global_evo_model evo_;
  std::vector<Ctmc_bridge_sampler> per_partition_;
};

// The contribution of one site to the branch genetic prior of `calc_branch_log_G`: the log density of
// the site's path across [t_P, t_X], which starts in `start_state` and makes the substitutions in
// `site_muts` (which must be sorted, confined to site `l` and consistent with `start_state`).
auto calc_site_branch_log_G(
    double t_P,
    double t_X,
    Real_seq_letter start_state,
    const Global_evo_model& evo,
    Site_index l,
    std::span<const Mutation> site_muts)
    -> double;

// Result of one Gibbs update, for the move's bookkeeping and for the efficiency diagnostics the task
// asks for: how many substitution events the update destroyed and created, and what it changed.
struct History_gibbs_result {
  // False when the chosen site carries no history at the chosen place, because it is missing there.
  // The update is then the identity, which is the correct Gibbs step for an empty block.
  bool performed = false;
  bool state_changed = false;
  int num_old_events = 0;
  int num_new_events = 0;
  double delta_log_G = 0.0;
  // Change in `lambda_i` at the node whose state was resampled; zero for every other node.
  double delta_lambda_at_node = 0.0;
};

// Counters for the efficiency of the Gibbs updates.  A Gibbs step has no acceptance rate to report,
// so what is counted instead is how much history actually moved per update.
struct History_gibbs_stats {
  int64_t branch_updates_proposed = 0;
  int64_t branch_updates_performed = 0;
  int64_t node_updates_proposed = 0;
  int64_t node_updates_performed = 0;
  int64_t node_states_changed = 0;
  int64_t site_refreshes = 0;
  int64_t events_removed = 0;
  int64_t events_created = 0;

  auto operator+=(const History_gibbs_stats& that) -> History_gibbs_stats& {
    branch_updates_proposed += that.branch_updates_proposed;
    branch_updates_performed += that.branch_updates_performed;
    node_updates_proposed += that.node_updates_proposed;
    node_updates_performed += that.node_updates_performed;
    node_states_changed += that.node_states_changed;
    site_refreshes += that.site_refreshes;
    events_removed += that.events_removed;
    events_created += that.events_created;
    return *this;
  }
};

// C1: redraws the history of site `l` on the branch ending at `X`, holding the state of the site at
// both ends of the branch, and everything else in the tree, fixed.  This is the update that can
// create a hidden pair of substitutions on a branch whose endpoints agree, delete a redundant pair,
// produce back substitutions and multiple hits, and move substitution times.
//
// `X` must not be the root.  `state_at_parent` must be the state of site `l` at the parent of `X`.
auto gibbs_resample_branch_site_history(
    Phylo_tree& tree,
    Branch_index X,
    Site_index l,
    Real_seq_letter state_at_parent,
    Site_history_sampler& sampler,
    absl::BitGenRef bitgen)
    -> History_gibbs_result;

// C2: redraws the state of site `l` at the inner node `V` from its full conditional, and then the
// conditional histories of all the branches incident on `V`.  The state at the far end of each
// incident branch is held fixed, as is everything outside those branches.
//
// At the root of `tree` the conditional uses the root prior of the model in place of a parent branch,
// and the new root state is recorded as a mutation above the root, which is where Delphy already
// keeps the difference between the reference sequence and the root sequence.
//
// `state_at_parent` must be the state of site `l` at the parent of `V`, and is ignored when `V` is the
// root.  The caller supplies it so that a move visiting several sites of one node can read them all
// out of one sequence view instead of walking to the root once per site; a debug build checks it.
auto gibbs_resample_node_site_state(
    Phylo_tree& tree,
    Node_index V,
    Site_index l,
    Real_seq_letter state_at_parent,
    Site_history_sampler& sampler,
    absl::BitGenRef bitgen)
    -> History_gibbs_result;

// C3: redraws the whole history of site `l` over the whole tree.  Felsenstein pruning for the single
// site gives the marginal likelihood of the states at every internal node; the root state is drawn
// from the root prior times the root partial likelihood, the internal states are drawn from the root
// down, and the history of every branch is then an exact conditional bridge.
//
// This is the update that mixes the root state and every internal state at once.  It costs a pass
// over the tree for the one site; no probability matrix is kept for more than one branch at a time.
struct Site_refresh_result {
  int num_old_events = 0;
  int num_new_events = 0;
  int num_states_changed = 0;
  bool root_state_changed = false;
};

auto gibbs_refresh_site_history(
    Phylo_tree& tree,
    Site_index l,
    Site_history_sampler& sampler,
    absl::BitGenRef bitgen)
    -> Site_refresh_result;

// Local marginalization, for a node-time move
// ===========================================
//
// Once the history of a site can move, the binding constraint on the node times is that they are
// hard-bounded by the mutation times on the incident branches: `Subrun::inner_node_displace_move`
// takes `t_min` from the latest mutation above the node and `t_max` from the earliest mutation below
// it, so a node can cross at most one mutation gap per step.  That gap is about one over the total
// mutation rate at the node, which shrinks as the sequence gets longer or more divergent, so the node
// time diffuses over the posterior in steps that shrink with the data: at 2,000 sites and 0.72
// substitutions per site the gap is about 1.4 years against a posterior some 670 years wide.
//
// The way out is to integrate the histories of the incident branches out, move the node time without
// any reference to them, and then draw them back from their exact conditional.  With `F(t)` the local
// marginal likelihood below, the Metropolis-Hastings ratio of such a move is
//
//   [ p_coal(t') F(t') q(t' -> t) ] / [ p_coal(t) F(t) q(t -> t') ],
//
// because the proposal density of the new history, rho(H' | t') / F(t'), cancels against the
// history's own factor in the target: nothing about either history survives in the ratio.  The move
// is therefore exact, and its step is limited only by the times of the surrounding nodes.

// The states just outside the branches incident on a node: the sequence at the parent and at each
// child.  None of them depends on where the node sits in time, so a move that evaluates the local
// marginal likelihood at two candidate times and then redraws the histories reads them out once and
// shares them.  Each one costs a walk to the root, which is otherwise the dominant cost of the move.
struct Incident_sequences {
  Sequence_overlay parent;
  std::array<Sequence_overlay, 2> child;
};

auto read_incident_sequences(const Phylo_tree& tree, Node_index V) -> Incident_sequences;

// log F(t_V): the marginal likelihood of the branches incident on `V`, with the state of every site
// at `V` and the whole history of those branches integrated out, evaluated as if `V` sat at time
// `t_V`, which must lie strictly between the time of the parent of `V` and the earliest of its
// children's, so that every incident branch has a positive duration.  The states at the far ends of
// the incident branches come from `outside` and are held fixed.  Sites missing at `V` contribute
// nothing, and a child below which the site is missing contributes a factor of one.
auto calc_local_marginal_log_likelihood(
    const Phylo_tree& tree,
    Node_index V,
    double t_V,
    const Incident_sequences& outside,
    Site_history_sampler& sampler)
    -> double;

// The refresh replaces the whole content of three branches, so it reports only what moved; the caller
// recomputes `lambda_i` at the node and the three branches' contribution to log G, which is cheaper
// than threading them through here.
struct Node_refresh_result {
  int num_old_events = 0;
  int num_new_events = 0;
  int num_states_changed = 0;
};

// Redraws the state of **every** site at `V` from its full conditional at the node's current time,
// and then the conditional history of every branch incident on `V`.  This is what the node-time move
// uses to put the histories back after moving the node, and it is C2 applied to the whole sequence at
// once.  The old histories need not be legal at the node's current time, so this must be called right
// after the node has been moved.
auto gibbs_refresh_node_histories(
    Phylo_tree& tree,
    Node_index V,
    const Incident_sequences& outside,
    Site_history_sampler& sampler,
    absl::BitGenRef bitgen)
    -> Node_refresh_result;

}  // namespace delphy

#endif // DELPHY_HISTORY_GIBBS_H_
