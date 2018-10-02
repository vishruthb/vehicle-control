#ifndef SEQ_LIN_BIKE_MODEL_H
#define SEQ_LIN_BIKE_MODEL_H

#include "Model.h"

using std::cos;
using std::sin;

class SeqLinBikeModel : public BikeModel {
private:
  vector<double> trajectory_;

  // for now just use nonlinear BikeModel's cost
  virtual AD<double> Cost(int t, const ADVec &xt, const ADVec &ut,
                          const ADVec &utp1) override {
    return BikeModel::Cost(t, xt, ut, utp1);
  }

  // ditto
  virtual AD<double> TerminalCost(const ADVec &xN) override {
    return BikeModel::TerminalCost(xN);
  }

  MatrixXd ComputeA(const vector<double> &x0, const vector<double> &u0) {
    AD<double> x = x0[X];
    AD<double> y = x0[Y];
    AD<double> psi = x0[PSI];
    AD<double> v = x0[V];
    AD<double> cte = x0[CTE];
    AD<double> epsi = x0[EPSI];
    AD<double> a = u0[A];
    AD<double> delta = u0[DELTA];

    MatrixXd A;
    // compute jacobian wrt to x and evaluate at xt, ut
    A << 1, 0, -v * sin(psi) * dt_, cos(psi) * dt_, 0, 0,  //
        0, 1, v * cos(psi) * dt_, sin(psi) * dt_, 0, 0,    //
        0, 0, 1, -1 / Lf_ * delta * dt_, 0, 0,             //
        0, 0, 0, 1, 0, 0,                                  //
        0, -1, 0, sin(epsi) * dt_, 0, v * cos(epsi) * dt_, //
        0, 0, 1, -1 / Lf_ * delta * dt_, 0, 0;

    return A;
  }

  MatrixXd ComputeB(const vector<double> &x0, const vector<double> &u0) {
    AD<double> x = x0[X];
    AD<double> y = x0[Y];
    AD<double> psi = x0[PSI];
    AD<double> v = x0[V];
    AD<double> cte = x0[CTE];
    AD<double> epsi = x0[EPSI];
    AD<double> a = u0[A];
    AD<double> delta = u0[DELTA];

    MatrixXd B;
    // compute jacobian wrt to u and evaluate at x0, u0
    B << 0, 0,             //
        0, 0,              //
        0, -v / Lf_ * dt_, //
        dt_, 0,            //
        0, 0,              //
        0, -v / Lf_ * dt_;

    return B;
  }
  // here's where the magic happens
  virtual ADVec DynamicsF(int t, const ADVec &xt, const ADVec &ut) override {
    vector<double> x0 = x_t(t, trajectory_);
    vector<double> u0 = u_t(t, trajectory_);

    MatrixXd A = ComputeA(x0, u0);
    MatrixXd B = ComputeB(x0, u0);

    AD<double> f = coeffs_[0] + coeffs_[1] * x + coeffs_[2] * CppAD::pow(x, 2) +
                   coeffs_[3] * CppAD::pow(x, 3);

    AD<double> psides = CppAD::atan(coeffs_[1] + 2 * coeffs_[2] * x +
                                    3 * coeffs_[3] * CppAD::pow(x, 2));

    auto Dot = [] AD<double>(const VectorXd &a, const ADVec &x) {
      AD<double> ax = 0;
      for (int i = 0; i < a.size(); i++)
        ax += a[i] * x[i];
      return ax;
    };

    ADVec fxtut(nx());
    fxtut[X] = x0[X] + Dot(A.row(X), xt) + Dot(B.row(X), ut);
    fxtut[Y] = x0[Y] + Dot(A.row(Y), xt) + Dot(B.row(Y), ut);
    fxtut[PSI] = x0[PSI] + Dot(A.row(PSI), xt) + Dot(B.row(PSI), ut);
    fxtut[V] = x0[V] + Dot(A.row(V), xt) + Dot(B.row(V), ut);
    fxtut[CTE] = x0[CTE] + f + Dot(A.row(CTE), xt) + Dot(B.row(CTE), ut);
    fxtut[EPSI] =
        x0[EPSI] - psides + Dot(A.row(EPSI), xt) + Dot(B.row(EPSI), ut);

    return fxtut;
  }

public:
  SeqLinBikeModel(int N, int nx, int nu, int delay, double dt, double vref,
                  const VectorXd &coeffs = {},
                  const vector<double> &trajectory = {})
      : BikeModel(N, nx, nu, delay, dt, vref, coeffs), trajectory_(trajectory) {
  }

  void set_trajectory(const vector<double> &trajectory) {
    trajectory_ = trajectory;
  }

  virtual ~SeqLinBikeModel() {}
};

#endif
