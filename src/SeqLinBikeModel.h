#ifndef SEQ_LIN_BIKE_MODEL_H
#define SEQ_LIN_BIKE_MODEL_H

#include "Model.h"

using std::cos;
using std::sin;

class SeqLinBikeModel : public BikeModel {
private:
  // For now ust use nonlinear BikeModel's cost, uncomment and modify below if
  // you want to deviate from the NL model's cost
  /* virtual AD<double> Cost(int t, const ADVec &xt, const ADVec &ut,
                          const ADVec &utp1) override {
    return BikeModel::Cost(t, xt, ut, utp1);
  }

  // ditto
  virtual AD<double> TerminalCost(const ADVec &xN) override {
    return BikeModel::TerminalCost(xN);
    }*/

  /// compute jacobian d/dx f(x,u) evaluated at linearization point (x0,u0)
  MatrixXd ComputeA(const vector<double> &x0, const vector<double> &u0) {
    double x = x0[X];
    double y = x0[Y];
    double psi = x0[PSI];
    double v = x0[V];
    double cte = x0[CTE];
    double epsi = x0[EPSI];
    double a = u0[A];
    double delta = u0[DELTA];

    Eigen::Matrix<double, 6, 6> A;

    // compute jacobian wrt to x and evaluate at xt, ut
    A << 1, 0, (-v * sin(psi) * dt_), (cos(psi) * dt_), 0, 0,  //
        0, 1, (v * cos(psi) * dt_), (sin(psi) * dt_), 0, 0,    //
        0, 0, 1, (-1 / Lf_ * delta * dt_), 0, 0,               //
        0, 0, 0, 1, 0, 0,                                      //
        0, -1, 0, (sin(epsi) * dt_), 0, (v * cos(epsi) * dt_), //
        0, 0, 1, (-1 / Lf_ * delta * dt_), 0, 0;

    return A;
  }

  /// compute jacobian d/du f(x,u) evaluated at linearization point (x0,u0)
  MatrixXd ComputeB(const vector<double> &x0, const vector<double> &u0) {
    double x = x0[X];
    double y = x0[Y];
    double psi = x0[PSI];
    double v = x0[V];
    double cte = x0[CTE];
    double epsi = x0[EPSI];
    double a = u0[A];
    double delta = u0[DELTA];

    MatrixXd B(6, 2);
    // compute jacobian wrt to u and evaluate at x0, u0
    B << 0, 0,             //
        0, 0,              //
        -v / Lf_ * dt_, 0, //
        0, dt_,            //
        0, 0,              //
        -v / Lf_ * dt_, 0;

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

    // if t=0, we use the measured state x(0) = x0 to initialize,
    // otherwise, we use shifted trjaectory
    vector<double> x0 = x_t(t + 1, trajectory_);
    vector<double> u0 = u_t(t + 1, trajectory_);

    MatrixXd mA = ComputeA(x0, u0);
    MatrixXd mB = ComputeB(x0, u0);

    AD<double> x = xt[X];
    AD<double> f = coeffs_[0] + coeffs_[1] * x + coeffs_[2] * CppAD::pow(x, 2) +
                   coeffs_[3] * CppAD::pow(x, 3);

    AD<double> psides = CppAD::atan(coeffs_[1] + 2 * coeffs_[2] * x +
                                    3 * coeffs_[3] * CppAD::pow(x, 2));

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
    double cte = x0[CTE];
    double epsi = x0[EPSI];
    double a = u0[A];
    double delta = u0[DELTA];

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
