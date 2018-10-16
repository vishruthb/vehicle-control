#ifndef DYN_BIKE_MODEL_H 
#define DYN_BIKE_MODEL_H

#include "Model.h"



class DynBikeModel : public Model {
protected:
  double dt_;
  double vref_;
  double inertia_;
  VectorXd coeffs_;

  enum State { X, Y, PSI, VX, VY, DPSI };
  enum Input { DELTA, A };

  const double Lf_ = 2.67;
  const double Ca_ = 2e4;
  const double M_ = 1000 + 80; // body + 4 wheels

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
    AD<double> s_cf = -atan((vy + Lf_ * dpsi) / vx);
    if (isnan(s_cf)) { s_cf = 0; }
    s_cf += delta;
    AD<double> s_cr = -atan((vy - Lf_ * dpsi) / vx);
    if (isnan(s_cr)) { s_cr = 0; }

    // cornering (lateral) forces for front and rear wheels
    AD<double> F_cr = Ca_ * s_cr;
    AD<double> F_cf = Ca_ * s_cf;

    AD<double> dvx = a + dpsi * vy;
    AD<double> dvy = (F_cf + F_cr) * CppAD::cos(delta) * 2 / M_ - vx * dpsi;
    AD<double> ddpsi = 2 * Lf_ * (F_cf - F_cr) / inertia_;

    fxtut[X] = x + vx * dt_;
    fxtut[Y] = y + vy * dt_;
    fxtut[PSI] = psi + dpsi * dt_;
    fxtut[VX] = vx + dvx * dt_;
    fxtut[VY] = vy + dvy * dt_;
    fxtut[DPSI] = dpsi + ddpsi * dt_;

    // compute the simple kinematic bike model predictions
//     AD<double> v, dx0, dy0, dpsi0, dvx0, dvy0, ddpsi0;
//     v = CppAD::sqrt(CppAD::pow(vx, 2) + CppAD::pow(vy, 2));
//     dx0 = v * CppAD::cos(psi);
//     dy0 = v * CppAD::sin(psi);
//     dpsi0 = -v / Lf_ * delta;
//     dvx0 = a * CppAD::cos(psi) - v * CppAD::sin(psi) * dpsi0;
//     dvy0 = a * CppAD::sin(psi) + v * CppAD::cos(psi) * dpsi0;
//     ddpsi0 = -a / Lf_ * delta;
// 
//     fxtut[X] = x + dx0 * dt_;
//     fxtut[Y] = y + dy0 * dt_;
//     fxtut[PSI] = psi + dpsi0 * dt_;
//     fxtut[VX] = vx + dvx0 * dt_;
//     fxtut[VY] = vy + dvy0 * dt_;
//     fxtut[DPSI] = dpsi + ddpsi0 * dt_;

//     std::cout << "fxtut: " << fxtut << std::endl;
//     std::cout << "\nv: " << v << ", a: " << a << ", delta: " << delta << std::endl;
//     std::cout << "x: " << x << ", vx: " << vx << ", dx0: " << dx0 << std::endl;
//     std::cout << "y: " << y << ", vy: " << vy << ", dy0: " << dy0 << std::endl;
//     std::cout << "psi: " << psi << ", dpsi: " << dpsi << ", dpsi0: " << dpsi0 << std::endl;
//     std::cout << "dvx: " << dvx << ", dvx0: " << dvx0 << std::endl;
//     std::cout << "dvy: " << dvy << ", dvy0: " << dvy0 << std::endl;
//     std::cout << "dpsi: " << dpsi << ", dpsi0: " << dpsi0 << std::endl;
//     std::cout << "\ns_cf: " << s_cf << "\ts_cr: " << s_cr << std::endl;
//     std::cout << "F_cf: " << F_cf << "\tF_cr: " << F_cr << std::endl;

    return fxtut;
  }

public:
  DynBikeModel(int N, int nx, int nu, int delay,
              double dt, double vref, double inertia = 0,
              const VectorXd &coeffs = {},
              const vector<double> &trajectory = {})
      : Model(N, nx, nu, delay, trajectory),
        dt_(dt), vref_(vref), inertia_(inertia), coeffs_(coeffs) {}

  void set_coeffs(const VectorXd &coeffs) { coeffs_ = coeffs; }
  void set_inertia(double inertia) { if (inertia_ == 0) { inertia_ = inertia; } }
  double inertia() { return inertia_; }

  virtual int xstart() override { return starts_[X]; }
  virtual int ystart() override {return starts_[Y]; }

  virtual ~DynBikeModel() {}
};

#endif
