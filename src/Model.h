#ifndef MODEL_H
#define MODEL_H

#include "Eigen-3.3/Eigen/Core"
#include <cmath>
#include <cppad/cppad.hpp>
#include <vector>

using CppAD::AD;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::max;
using std::vector;

using ADVec = CPPAD_TESTVECTOR(AD<double>);

/*!\brief Base Model Class.  Takes care of all of the low-level details needed
 * to implement the cost function, dynamics.  Derive from this class and
 * override the pure virtual functions Cost(t,x(t),u(t),u(t+1)),
 * TerminalCost(x(N)), and DynamicsF(t,x(t),u(t));  */
class Model {

protected:
  int N_;     ///> optimization horizon is N_-1
  int nx_;    ///> state dimension
  int nu_;    ///> input dimension
  int delay_; ///> input delay: x(t+1) = f(x(t),u(t-delay_))
  vector<int>
      starts_; ///> bookkeeping vector: starts_[i] says where in optimization
               ///> vector vars component i (either a scalar state or input)
  vector<double> trajectory_;

  /// Pure virtual function: should return per-stage cost
  virtual AD<double> Cost(int t, const ADVec &xt, const ADVec &ut,
                          const ADVec &utp1) = 0;

  /// Pure virtual function: should return terminal cost
  virtual AD<double> TerminalCost(const ADVec &xN) = 0;

  /// Pure virtual function: should return x(t+1) = f(x(t),u(t))
  virtual ADVec DynamicsF(int t, const ADVec &xt, const ADVec &ut) = 0;

  /// Helper function to extract x(t) from vars
  template <typename Vector> Vector x_t(int t, const Vector &vars) {
    Vector xt(nx_);
    for (int i = 0; i < nx_; ++i)
      xt[i] = vars[starts_[i] + t];

    return xt;
  }

  /// Helper function to extract u(t) from vars
  template <typename Vector> Vector u_t(int t, const Vector &vars) {
    Vector ut(nu_);
    for (int i = 0; i < nu_; ++i)
      ut[i] = vars[starts_[nx_ + i] + t];

    return ut;
  }

public:
  // need to have this typedef exactly as is for this to work with IpOPT;
  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  /// Establishes opitmization horizon (N), state-dim (nx), input-dim (nu),
  /// input delay (delay), and populates bookkeeping vector starts_
  Model(int N, int nx, int nu, int delay, const vector<double> &trajectory = {})
      : N_(N), nx_(nx), nu_(nu), delay_(delay), starts_{0},
        trajectory_(trajectory) {
    for (int i = 1; i <= nx_; ++i)
      starts_.push_back(starts_[i - 1] + N);

    for (int i = 0; i < nu_ - 1; ++i)
      starts_.push_back(starts_[nx_ + i] + N - 1);
  }

  /// Called by MPC to set Model's trajectory_ to MPC's warmstart_, where the
  /// latter term is teh solution to the previous MPC run
  void set_trajectory(vector<double> &trajectory) { trajectory_ = trajectory; }

  bool initialized() { return (trajectory_.size() > 0); }

  /// returns reference to starts vector
  vector<int> &starts() { return starts_; }

  /// returns state dim nx
  int nx() { return nx_; }

  /// returns input dim nu
  int nu() { return nu_; }

  /// returns horizon N
  int N() { return N_; }

  /// Needed to plot MPC planned trajectory in simulator: should return
  /// starts_[X], where X specifies the the position of X in the the state
  /// vector.  e.g. if state[0] = x, then xstart shoudl return starts_[0];
  virtual int xstart() = 0;

  /// Needed to plot MPC planned trajectory in simulator: should return
  /// starts_[Y], where Y specifies the the position of X in the the state
  /// vector.  e.g. if state[1] = y, then xstart shoudl return starts_[1];
  virtual int ystart() = 0;

  /// Functor operator that is called by IpOPT.  This implements the low level
  /// details
  void operator()(ADVec &fg, const ADVec &vars) {

    // start by setting the cost
    fg[0] = 0;
    for (int t = 0; t < N_ - 2; ++t) {
      fg[0] += Cost(t, x_t(t, vars), u_t(t, vars), u_t(t + 1, vars));
    }
    fg[0] +=
        Cost(N_ - 2, x_t(N_ - 2, vars), u_t(N_ - 2, vars), u_t(N_ - 2, vars));
    fg[0] += TerminalCost(x_t(N_ - 1, vars));

    // set the constraint used to enforce x(0) = x0
    ADVec x0 = x_t(0, vars);
    for (int i = 0; i < nx_; ++i)
      fg[1 + starts_[i]] = vars[starts_[i]];

    // set the constraints used to enforce dynamics;
    // delay_ specifies
    for (int t = 0; t < N_ - 1; ++t) {
      ADVec Fxtut = DynamicsF(t, x_t(t, vars), u_t(max(0, t - delay_), vars));
      ADVec xtp1 = x_t(t + 1, vars);

      // enforce things elementwise
      for (int i = 0; i < nx_; ++i)
        fg[2 + starts_[i] + t] = xtp1[i] - Fxtut[i];
    }
  }
  virtual ~Model() {}
};

/*! \brief Implementation of simple bike model used for direct NLP.  The cost
 * function parameters were taken from someone's solution online just to debug
 * things, will tune them more substantially later*/
class BikeModel : public Model {
protected:
  double dt_;       ///>sampling time dt
  double vref_;     ///>reference velocity
  VectorXd coeffs_; ///>coefficients
  /*!\enum BikeModel::State positions of different state components in state,
   * i.e. if state(t) =
   * [x(t); y(t); psi(t); cte(t); epsi(t)] then state[X](t) = x(t)*/
  enum State { X, Y, PSI, V, CTE, EPSI };

  ///\enum positions of different input components in input u(t), i.e. ut(t) =
  ///[delta(t); a(t)].  Also specifies starts (offset by nx_)
  enum Input { DELTA, A };

  const double Lf_ = 2.67; ///>model parameter roughly specifying turning radius

  virtual AD<double> Cost(int t, const ADVec &xt, const ADVec &ut,
                          const ADVec &utp1) override {
    AD<double> cost(0);

    cost += 3000 * CppAD::pow(xt[CTE], 2);
    cost += 3000 * CppAD::pow(xt[EPSI], 2);
    cost += CppAD::pow(xt[V] - vref_, 2);

    cost += 5 * CppAD::pow(ut[DELTA], 2);
    cost += 5 * CppAD::pow(ut[A], 2);

    cost += 250 * CppAD::pow(ut[DELTA] * xt[V], 2);

    cost += 200 * CppAD::pow(utp1[DELTA] - ut[DELTA], 2);
    cost += 10 * CppAD::pow(utp1[A] - ut[A], 2);

    return cost;
  }

  virtual AD<double> TerminalCost(const ADVec &xN) override {
    AD<double> cost(0);

    cost += 3000 * CppAD::pow(xN[CTE], 2);
    cost += 3000 * CppAD::pow(xN[EPSI], 2);
    cost += CppAD::pow(xN[V] - vref_, 2);

    return cost;
  }
  virtual ADVec DynamicsF(int t, const ADVec &xt, const ADVec &ut) override {

    AD<double> x = xt[X];
    AD<double> y = xt[Y];
    AD<double> psi = xt[PSI];
    AD<double> v = xt[V];
    AD<double> cte = xt[CTE];
    AD<double> epsi = xt[EPSI];
    AD<double> a = ut[A];
    AD<double> delta = ut[DELTA];

    AD<double> f = coeffs_[0] + coeffs_[1] * x + coeffs_[2] * CppAD::pow(x, 2) +
                   coeffs_[3] * CppAD::pow(x, 3);

    AD<double> psides = CppAD::atan(coeffs_[1] + 2 * coeffs_[2] * x +
                                    3 * coeffs_[3] * CppAD::pow(x, 2));

    ADVec fxtut(nx());
    fxtut[X] = x + v * CppAD::cos(psi) * dt_;
    fxtut[Y] = y + v * CppAD::sin(psi) * dt_;
    fxtut[PSI] = psi - v / Lf_ * delta * dt_;
    fxtut[V] = v + a * dt_;
    fxtut[CTE] = (f - y) + v * CppAD::sin(epsi) * dt_;
    fxtut[EPSI] = (psi - psides) - v / Lf_ * delta * dt_;

    return fxtut;
  }

public:
  BikeModel(int N, int nx, int nu, int delay, double dt, double vref,
            const VectorXd &coeffs = {}, const vector<double> &trajectory = {})
      : Model(N, nx, nu, delay, trajectory), dt_(dt), vref_(vref),
        coeffs_(coeffs) {}

  /// call this before every mpc solve to update reference trajectory, which is
  /// specified in terms of polynomial = sum_{i=0}^2 coeffs[i] * pow(x,i)
  void set_coeffs(const VectorXd &coeffs) { coeffs_ = coeffs; }

  virtual int xstart() override { return starts_[X]; }
  virtual int ystart() override { return starts_[Y]; }

  virtual ~BikeModel() {}
};

#endif
