#ifndef SEQ_LIN_BIKE_MODEL_H
#define SEQ_LIN_BIKE_MODEL_H

#include "Model.h"

using std::cos;
using std::sin;

class SeqLinBikeModel : public BikeModel {
private:
  virtual AD<double> Cost(int t, const ADVec &xt, const ADVec &ut,
                          const ADVec &utp1) override {
    if (trajectory_.size() == 0)
      return BikeModel::Cost(t, xt, ut, utp1);

    AD<double> cost(0);

    cost += 3000 * CppAD::pow(xt[CTE], 2);
    cost += 3000 * CppAD::pow(xt[EPSI], 2);
    cost += CppAD::pow(xt[V] - vref_, 2);

    cost += 5 * CppAD::pow(ut[DELTA], 2);
    cost += 5 * CppAD::pow(ut[A], 2);

    double v0 = x_t(t + 1, trajectory_)[V];
    double d0 = u_t(t + 1, trajectory_)[DELTA];
    AD<double> dv = xt[V] - v0;
    AD<double> ddel = ut[DELTA] - d0;

    // take quadratic approximation to BikeModel's non-convex term
    cost += 250 * (2 * v0 * v0 * d0 * ddel + v0 * v0 * CppAD::pow(ddel, 2) +
                   2 * v0 * d0 * d0 * dv + 2 * v0 * d0 * dv * ddel +
                   d0 * d0 * CppAD::pow(dv, 2));

    cost += 200 * CppAD::pow(utp1[DELTA] - ut[DELTA], 2);
    cost += 10 * CppAD::pow(utp1[A] - ut[A], 2);

    return cost;
  }
  /*

  // ditto
  virtual AD<double> TerminalCost(const ADVec &xN) override {
    return BikeModel::TerminalCost(xN);
    }*/

  /// compute jacobian d/dx f(x,u) evaluated at linearization point (x0,u0)
  MatrixXd ComputeA(const vector<double> &x0, const vector<double> &u0) {
    double x = x0[X];
//     double y = x0[Y];
    double psi = x0[PSI];
    double v = x0[V];
//     double cte = x0[CTE];
    double epsi = x0[EPSI];
//     double a = u0[A];
    double delta = u0[DELTA];

    double df = coeffs_[1] + 2*coeffs_[2]*x + 3*coeffs_[3]*x*x;
    double d2f = 2*coeffs_[2] + 6*coeffs_[3]*x;

    Eigen::Matrix<double, 6, 6> A;

    // compute jacobian wrt to x and evaluate at xt, ut
    A << 1, 0, (-v * sin(psi) * dt_), (cos(psi) * dt_), 0, 0,  //
        0, 1, (v * cos(psi) * dt_), (sin(psi) * dt_), 0, 0,    //
        0, 0, 1, (-1 / Lf_ * delta * dt_), 0, 0,               //
        0, 0, 0, 1, 0, 0,                                      //
        df, -1, 0, (sin(epsi) * dt_), 0, (v * cos(epsi) * dt_), //
        -d2f/(df*df+1), 0, 1, (-1 / Lf_ * delta * dt_), 0, 0;

    return A;
  }

  /// compute jacobian d/du f(x,u) evaluated at linearization point (x0,u0)
  MatrixXd ComputeB(const vector<double> &x0, const vector<double> &u0) {
    MatrixXd B(6, 2);
    // compute jacobian wrt to u and evaluate at x0, u0
    B << 0, 0,             //
        0, 0,              //
        -x0[V] / Lf_ * dt_, 0, //
        0, dt_,            //
        0, 0,              //
        -x0[V] / Lf_ * dt_, 0;

    return B;
  }
  /// If we're on our first run, use the nonlinear model.  Otherwise, linearize
  /// around stored trajectory (which is previous MPC run's solution, but with
  /// the actual state x0 swapped in
  virtual ADVec DynamicsF(int t, const ADVec &xt, const ADVec &ut) override {

    // if this is our first run, use nonlinear solver to initialize
    if (trajectory_.size() == 0) {
      return BikeModel::DynamicsF(t, xt, ut);
    }

    // get one step shfited x0,u0 to linearize about
    vector<double> x0 = x_t(t + 1, trajectory_);
    vector<double> u0 = u_t(t + 1, trajectory_);

    MatrixXd mA = ComputeA(x0, u0);
    MatrixXd mB = ComputeB(x0, u0);

    AD<double> x = xt[X];

    auto Dot = [](const VectorXd &a, const ADVec &x) {
      AD<double> ax = 0;
      for (int i = 0; i < a.size(); i++)
        ax += a[i] * x[i];
      return ax;
    };

    ADVec fxtut(nx());
    double xx = x0[X];
    double y = x0[Y];
    double psi = x0[PSI];
    double v = x0[V];
//     double cte = x0[CTE];
    double epsi = x0[EPSI];
    double a = u0[A];
    double delta = u0[DELTA];

    double f = coeffs_[0] + coeffs_[1] * xx + coeffs_[2] * CppAD::pow(xx, 2) +
                   coeffs_[3] * CppAD::pow(xx, 3);

    double psides = CppAD::atan(coeffs_[1] + 2 * coeffs_[2] * xx +
                                    3 * coeffs_[3] * CppAD::pow(xx, 2));

    // first add term contributed from linearization point
    fxtut[X] = xx + v * cos(psi) * dt_;
    fxtut[Y] = y + v * sin(psi) * dt_;
    fxtut[PSI] = psi - v / Lf_ * delta * dt_;
    fxtut[V] = v + a * dt_;
    fxtut[CTE] = (f - y) + v * sin(epsi) * dt_;
    fxtut[EPSI] = (psi - psides) - v / Lf_ * delta * dt_;

    // now add term from linear deviations

    auto diff = [](const ADVec &v1, const vector<double> &v2) {
      ADVec del(v1.size());
      for (int i = 0; i < v1.size(); i++)
        del[i] = v1[i] - v2[i];

      return del;
    };

    ADVec dx = diff(xt, x0);
    ADVec du = diff(ut, u0);

    fxtut[X] += Dot(mA.row(X), dx) + Dot(mB.row(X), du);
    fxtut[Y] += Dot(mA.row(Y), dx) + Dot(mB.row(Y), du);
    fxtut[PSI] += Dot(mA.row(PSI), dx) + Dot(mB.row(PSI), du);
    fxtut[V] += Dot(mA.row(V), dx) + Dot(mB.row(V), du);
    fxtut[CTE] += Dot(mA.row(CTE), dx) + Dot(mB.row(CTE), du);
    fxtut[EPSI] += Dot(mA.row(EPSI), dx) + Dot(mB.row(EPSI), du);

    return fxtut;
  }

public:
  SeqLinBikeModel(int N, int nx, int nu, int delay, double dt, double vref,
                  const VectorXd &coeffs = {},
                  const vector<double> &trajectory = {})
      : BikeModel(N, nx, nu, delay, dt, vref, coeffs, trajectory) {}

  virtual ~SeqLinBikeModel() {}
};

#endif
