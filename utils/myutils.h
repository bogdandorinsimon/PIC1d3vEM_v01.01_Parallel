#ifndef MYUTILS_H
#define MYUTILS_H

void bfield(VecDoub &by, VecDoub &bz, VecDoub ey, VecDoub ez, int m, double c);
void efield(VecDoub &ex, VecDoub &ey, VecDoub &ez, VecDoub by, VecDoub bz, VecDoub jxe, VecDoub jye, VecDoub jze, VecDoub jxi, VecDoub jyi, VecDoub jzi, int m, double c);
void mover(VecDoub &x, VecDoub &vx, VecDoub &vy, VecDoub &vz, VecDoub ex, VecDoub ey, VecDoub ez, VecDoub by, VecDoub bz, \
           double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double c, int np, int m);
void current(VecDoub &jxe_s, VecDoub &jye_s, VecDoub &jze_s, VecDoub &jxi_s, VecDoub &jyi_s, VecDoub &jzi_s, VecDoub x, \
             VecDoub vx, VecDoub vy, VecDoub vz, double qse, double qsi, int np, int m);
void energy(double &etaK, double &etaE, double &etaB, VecDoub vx, VecDoub vy, VecDoub vz, VecDoub ex, VecDoub ey, VecDoub ez, VecDoub by, VecDoub bz, \
			VecDoub jxe, VecDoub jye, VecDoub jze, VecDoub jxi, VecDoub jyi, VecDoub jzi, double ex0, double ey0, double ez0, double bx0, double by0, double bz0, \
			double qme, double qmi, double qse, double qsi, double c, int np, int m);                     

#endif
