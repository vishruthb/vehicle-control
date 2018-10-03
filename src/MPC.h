#ifndef MPC_H
#define MPC_H

#include "Eigen-3.3/Eigen/Core"
#include "Model.h"
#include <string>
#include <vector>

using Eigen::VectorXd;
using std::string;
using std::vector;
using Dvector = CPPAD_TESTVECTOR(double);

/*!\brief Struct used to hold bounds for state constraints. Specifically, x_low_
<= x(t) <= x_up_, and u_low_ <= u(t) <= u_low_ for all t, where constraints are
enforced element wise.

Struct used to hold bounds for state constraints. Specifically, x_low_ <= x(t)
<= x_up_, and u_low_ <= u(t) <= u_low_ for all t, where constraints are enforced
element wise.  Derive from this class to add additional constraint bounds, etc.,
if you need to pass in more information to the MPC solver. */

struct Bounds {
  vector<double> x_up_;  ///> upper bound on state
  vector<double> x_low_; ///>lower bound on state
  vector<double> u_up_;  ///>upper bound on input
  vector<double> u_low_; ///>lower bound on input
};

/*!\brief Base MPC class.  Whereas the model that you pass in takes care of
 * specifying the dynamics and the cost function, the MPC object sets up the
 * bounds on variables (as specified in Bounds object), and on constraints
 * (enforces x(0) = x0, x(t+1) - f(x(t),u(t)) = 0), and also warmstarts your
 * solver with a suitably shifted version of the last run's solution.  */
class MPC {
protected:
  /// horizon is N_ -1
  size_t N_;
  /// Model object specifying dynamics, cost function, and
  /// additional constraints beyond box constraints in state and
  /// input
  Model &model_;
  size_t nvars_;        ///> number of optimizaiton variables
  size_t nconstraints_; ///> number of constraints (excluding box constraints)
  /// Bounds struct specifying input/state box constraints and
  /// whatever else a user chooses to add
  Bounds bounds_;
  vector<int> starts_; ///> bookkeeping: this is set using the starts_ vector of
                       ///> Model to ensure consistency

  vector<double> warmstart_; ///> remember last run's solution for warm starting

  virtual void SetupWarmStart(Dvector &vars, const VectorXd &state);
  virtual void SetupVarBounds(Dvector &vars_lowerbound,
                              Dvector &vars_upperbound);
  virtual void SetupConstraintBounds(Dvector &constraints_lowerbound,
                                     Dvector &constraints_upperbound,
                                     const VectorXd &state);

  virtual void ProcessSolution(vector<double> &result,
                               const vector<double> &sol, int shift = 0);
  virtual void SetupOptions(string &options);

public:
  MPC(size_t N, Model &model, size_t nvars, size_t nconstraints, Bounds bounds)
      : N_{N}, model_(model), nvars_(nvars), nconstraints_(nconstraints),
        bounds_(bounds), warmstart_(0) {
    starts_ = model_.starts();
    model_.set_trajectory(warmstart_);
  }

  /// return optimization horizon
  size_t get_horizon() { return N_; }
  /// set optimization horizon to N_ = n (actual horizon is n-1)
  void set_horizon(int n) { N_ = n; }

  /// update model -- this also updates our start variable and clears warmstart_
  void set_model(Model &new_model) {
    model_ = new_model;
    starts_ = new_model.starts();
    warmstart_.clear();
  }

  virtual ~MPC() {}

  /// Solve the model given an initial state and a reference trajectory.
  /// Return the first actuations.
  virtual vector<double> Solve(const VectorXd &state, const VectorXd &ref);
};

///\brief Almost identical to standard MPC, just needs to update model's
/// trajectory to be used in linearization step
/// don't need this anymore!
class SeqLinMPC : public MPC {

public:
  SeqLinMPC(size_t N, Model &model, size_t nvars, size_t nconstraints,
            Bounds bounds)
      : MPC(N, model, nvars, nconstraints, bounds) {
    // the model will use the previous solution as a trajectory to linearize
    // around
    // model_.set_trajectory(warmstart_);
  }

  /* virtual vector<double> Solve(const VectorXd &state,
                               const VectorXd &ref) override {
    // we use state[1] of warmstart_ to store x0 to be used in seq.
    // linearization.  This is ugly. but we'll leave it as is for now.
    if (warmstart_.size()) {
      //   for (int i = 0; i < model_.nx(); i++)
      // warmstart_[starts_[i + 1]] = state[i];
      model_.set_trajectory(warmstart_);
    }
    return MPC::Solve(state, ref);
    }*/

  // model_.set_trajectory(warmstart_);
  //}
};

#endif /* MPC_H */
