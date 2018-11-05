#ifndef DYN_BIKE_MODEL_H 
#define DYN_BIKE_MODEL_H

#include "Model.h"



class DynBikeModel : public Model {
protected:
  double dt_;
  double vref_;
  double Iz_;
  VectorXd coeffs_;

  enum State { X, Y, PSI, VX, VY, DPSI };
  enum Input { DELTA, A };

  const double Lf_ = 1.6;
  const double Lr_ = 1.27;
  const double M_ = 1.080; // body + 4 wheels
  const double Ca_ = 0.8 * 9.81;

  virtual AD<double> Cost(int t, const ADVec &xt, const ADVec &ut,
                          const ADVec &utp1) override {
    AD<double> cost(0);

    AD<double> x = xt[X];
    AD<double> f = coeffs_[0] + coeffs_[1] * x + coeffs_[2] * CppAD::pow(x, 2) +
                   coeffs_[3] * CppAD::pow(x, 3);
    AD<double> psides = CppAD::atan(coeffs_[1] + 2 * coeffs_[2] * x +
                                    3 * coeffs_[3] * CppAD::pow(x, 2));

    cost += 3000 * CppAD::pow(f - xt[Y], 2);
    cost += 3000 * CppAD::pow(xt[PSI] - psides, 2);
    cost += CppAD::pow(xt[VX] - vref_, 2);

    cost += 5 * CppAD::pow(ut[DELTA], 2);
    cost += 5 * CppAD::pow(ut[A], 2);

//     cost += 200 * CppAD::pow(ut[DELTA] * xt[V], 2);
    cost += 300 * CppAD::pow(xt[VY], 2); // penalize longitudinal velocity
//     cost += 200 * CppAD::pow(xt[DPSI], 2); // penalize yaw rate

    cost += 200 * CppAD::pow(utp1[DELTA] - ut[DELTA], 2);
    cost += 10 * CppAD::pow(utp1[A] - ut[A], 2);

    return cost;
  }

  virtual AD<double> TerminalCost(const ADVec &xN) override {
    AD<double> cost(0);
    AD<double> x = xN[X];
    AD<double> f = coeffs_[0] + coeffs_[1] * x + coeffs_[2] * CppAD::pow(x, 2) +
                   coeffs_[3] * CppAD::pow(x, 3);
    AD<double> psides = CppAD::atan(coeffs_[1] + 2 * coeffs_[2] * x +
                                    3 * coeffs_[3] * CppAD::pow(x, 2));

    cost += 3000 * CppAD::pow(f - xN[Y], 2);
    cost += 3000 * CppAD::pow(psides - xN[PSI], 2);
    cost += CppAD::pow(xN[VX] - vref_, 2);

    return cost;
  }

  virtual ADVec DynamicsF(int t, const ADVec &xt, const ADVec &ut) override {

    AD<double> x = xt[X];
    AD<double> y = xt[Y];
    AD<double> psi = xt[PSI];
    AD<double> vx = xt[VX];
    AD<double> vy = xt[VY];
    AD<double> dpsi = xt[DPSI]; // wrong direction?

    AD<double> a = ut[A];
    AD<double> delta = ut[DELTA]; // wrong direction?

    ADVec fxtut(nx());

    // slip angles of front and rear wheels
    AD<double> alpha_f = -atan2(vy + Lf_ * dpsi, vx);
    if (isnan(alpha_f)) { alpha_f = 0; }
    alpha_f += delta;

    AD<double> alpha_r= -atan2(vy - Lf_ * dpsi, vx);
    if (isnan(alpha_r)) { alpha_r = 0; }

    // cornering (lateral) forces for front and rear wheels
    AD<double> F_yf = -2 * Ca_ * alpha_f;
    AD<double> F_yr = -2 * Ca_ * alpha_r;

    AD<double> dvx = a - F_yf * CppAD::sin(delta) / M_ + vy * dpsi;
    AD<double> dvy = (F_yf * CppAD::cos(delta) + F_yr) / M_ - vx * dpsi;
    AD<double> ddpsi = (Lf_ * F_yf * CppAD::cos(delta) - Lr_ * F_yr) / Iz_;
    AD<double> dx = vx * CppAD::cos(psi) - vy * CppAD::sin(psi);
    AD<double> dy = vx * CppAD::sin(psi) + vy * CppAD::cos(psi);

    fxtut[X] = x + dx * dt_;
    fxtut[Y] = y + dy * dt_;
    fxtut[PSI] = psi + dpsi * dt_;
    fxtut[VX] = vx + dvx * dt_;
    fxtut[VY] = vy + dvy * dt_;
    fxtut[DPSI] = dpsi + ddpsi * dt_;

    // compute the simple kinematic bike model predictions
    AD<double> v, dx0, dy0, dpsi0, dvx0, dvy0, ddpsi0;
    v = CppAD::sqrt(CppAD::pow(vx, 2) + CppAD::pow(vy, 2));
    dx0 = v * CppAD::cos(psi);
    dy0 = v * CppAD::sin(psi);
    dpsi0 = -v / Lf_ * delta;
    dvx0 = a * CppAD::cos(psi) - v * CppAD::sin(psi) * dpsi0;
    dvy0 = a * CppAD::sin(psi) + v * CppAD::cos(psi) * dpsi0;
    ddpsi0 = -a / Lf_ * delta;

    std::cout << "fxtut: " << fxtut << std::endl;

    std::cout << "\nv: " << v << ", a: " << a << ", delta: " << delta << std::endl;
    std::cout << "x: " << x << ", dx: " << dx << ", dx0: " << dx0 << std::endl;
    std::cout << "y: " << y << ", dy: " << dy << ", dy0: " << dy0 << std::endl;
    std::cout << "dpsi: " << dpsi << ", dpsi0: " << dpsi0 << std::endl;
    std::cout << "vx: " << vx << ", dvx: " << dvx << ", dvx0: " << dvx0 << std::endl;
    std::cout << "vy: " << vy << ", dvy: " << dvy << ", dvy0: " << dvy0 << std::endl;
    std::cout << "ddpsi: " << ddpsi << ", ddpsi0: " << ddpsi0 << std::endl;

    std::cout << "\nalpha_f: " << alpha_f<< "\talpha_r: " << alpha_r<< std::endl;
    std::cout << "F_yf: " << F_yf << "\tF_yr: " << F_yr << std::endl;

    return fxtut;
  }

public:
  DynBikeModel(int N, int nx, int nu, int delay,
              double dt, double vref, double inertia = 0.05,
              const VectorXd &coeffs = {},
              const vector<double> &trajectory = {})
      : Model(N, nx, nu, delay, trajectory),
        dt_(dt), vref_(vref), Iz_(inertia), coeffs_(coeffs) {}

  void set_coeffs(const VectorXd &coeffs) { coeffs_ = coeffs; }
  void set_inertia(double inertia) { Iz_ = inertia; }
  double inertia() { return Iz_; }

  virtual int xstart() override { return starts_[X]; }
  virtual int ystart() override {return starts_[Y]; }

  virtual ~DynBikeModel() {}
};

#endif
