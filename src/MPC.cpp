#include "MPC.h"
#include "Eigen-3.3/Eigen/Core"
#include "Model.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <vector>

using CppAD::AD;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void MPC::SetupWarmStart(Dvector &vars, const VectorXd &state) {
  // warm start using previous solution, if it exists
  if (warmstart_.size()) {
    // warm start x[t] = x_old[t+1] u[t] = u_old[t+1] for t<N-2
    for (int t = 0; t < N_ - 2; t++) {
      for (int i = 0; i < starts_.size(); i++) {
        vars[starts_[i] + t] = warmstart_[starts_[i] + t + 1];
      }
    }
    // warm start x[N-2] = x[N-1] = x_old[N-1] and x[N-1], and u[N-2] =
    // u_old[N-2]
    for (int i = 0; i < starts_.size(); i++) {
      if (i < model_.nx()) {
        vars[starts_[i] + N_ - 2] = warmstart_[starts_[i] + N_ - 1];
        vars[starts_[i] + N_ - 1] = warmstart_[starts_[i] + N_ - 1];
      } else {
        vars[starts_[i] + N_ - 2] = warmstart_[starts_[i] + N_ - 2];
      }
    }
  } else { // just start with everything equal to zero except x[0] = x0;
    for (int i = 0; i < model_.nx(); i++) {
      vars[starts_[i]] = state[i];
    }
    for (int i = model_.nx(); i < nvars_; i++) {
      vars[i] = 0;
    }
  }
}

void MPC::SetupVarBounds(Dvector &vars_lowerbound, Dvector &vars_upperbound) {
  for (int t = 0; t < N_ - 1; t++) {
    for (int i = 0; i < model_.nx(); i++) {
      vars_lowerbound[starts_[i] + t] = bounds_.x_low_[i];
      vars_upperbound[starts_[i] + t] = bounds_.x_up_[i];
    }
    for (int i = 0; i < model_.nu(); i++) {
      vars_lowerbound[starts_[model_.nx() + i] + t] = bounds_.u_low_[i];
      vars_upperbound[starts_[model_.nx() + i] + t] = bounds_.u_up_[i];
    }
  }

  for (int i = 0; i < model_.nx(); i++) {
    vars_lowerbound[starts_[i] + N_ - 1] = bounds_.x_low_[i];
    vars_upperbound[starts_[i] + N_ - 1] = bounds_.x_up_[i];
  }
}

void MPC::SetupConstraintBounds(Dvector &constraints_lowerbound,
                                Dvector &constraints_upperbound,
                                const VectorXd &state) {
  for (int i = 0; i < nconstraints_; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  for (int i = 0; i < model_.nx(); i++) {
    constraints_lowerbound[starts_[i]] = state[i];
    constraints_upperbound[starts_[i]] = state[i];
  }
}

void MPC::ProcessSolution(vector<double> &result, Dvector sol, int shift) {
  // push back first actutation values
  for (int i = model_.nx(); i < starts_.size(); i++) {
    result.push_back(sol[starts_[i] + shift]);
  }

  // push back predicted trajectory to plot in simulator
  int xstart = model_.xstart();
  int ystart = model_.ystart();
  for (int t = 1; t < N_; t++) {
    size_t x_idx =
        std::min(static_cast<size_t>(xstart + shift + t), xstart + N_ - 1);
    size_t y_idx =
        std::min(static_cast<size_t>(ystart + shift + t), ystart + N_ - 1);
    result.push_back(sol[x_idx]);
    result.push_back(sol[y_idx]);
  }
}

void MPC::SetupOptions(std::string &options) {
  // options for IPOPT solver

  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";
}

vector<double> MPC::Solve(const VectorXd &state, const VectorXd &coeffs) {

  // vector containing starting points of the different stacked variables
  // x(0:N) and u(0:N-1) in vars (this is obtained from the model to ensure
  // consistency)

  // vector of decision variables
  Dvector vars(nvars_);

  SetupWarmStart(vars, state);

  // setup upper and lower box constraints on state and input
  Dvector vars_lowerbound(nvars_);
  Dvector vars_upperbound(nvars_);

  SetupVarBounds(vars_lowerbound, vars_upperbound);

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(nconstraints_);
  Dvector constraints_upperbound(nconstraints_);

  SetupConstraintBounds(constraints_lowerbound, constraints_upperbound, state);

  std::string options;
  SetupOptions(options);

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, Model>(options, vars, vars_lowerbound,
                                      vars_upperbound, constraints_lowerbound,
                                      constraints_upperbound, model_, solution);

  // Check some of the solution values
  bool ok = true;
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  vector<double> result;
  if (ok) {
    ProcessSolution(result, solution.x);
    warmstart_ = solution.x;
  } else {
    // if we get an infeasible solution, use last feasible one shifted by one
    // TODO: this no longer makes sense after if we get mroe than one infeasible
    // solution in a row
    assert(warmstart_.size() && "First iteration infeasible");
    ProcessSolution(result, warmstart_, 1);
  }

  return result;
}
