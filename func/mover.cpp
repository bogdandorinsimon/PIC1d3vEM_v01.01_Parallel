// This function MOVE all the particles over one time-step

#include <cmath>
#include "../utils/nr3.h"

void mover(VecDoub &x, VecDoub &vx, VecDoub &vy, VecDoub &vz, VecDoub ex, VecDoub ey, VecDoub ez, VecDoub by, VecDoub bz, \
           double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double c, int np, int m)
{
	int k, i, j;
	double ex_p, ey_p, ez_p, by_p, bz_p, qm, gamma_p, eps_x, eps_y, eps_z, beta_x, beta_y, beta_z, ux1, uy1, uz1, wx, wy, wz, \
	       ux2, uy2, uz2, ux, uy, uz;
	
	for (k=0; k<2*np; k++)
	{
		// particle 'k' is located between i and i+1 and between j+1/2 and j+3/2
		// see diagram on page 22 of Notebook 5
		i=x[k];
		j=x[k]-0.5;
		
		// compute the electric and magnetic fields in the actual position of each particle
		// see Notebook 5, page 45 for details
		// relocation from half-integer to full-integer grid points is needed for Ex: interpolarion from full-integer grid-points
		ex_p=(i+1-x[k])*0.5*(ex[i-1]+ex[i])+(x[k]-i)*0.5*(ex[i]+ex[i+1]);
		
		// no relocation is needed for Ey and Ez: interpolation from half-integer grid-points
		ey_p=(j+1.5-x[k])*ey[j]+(x[k]-j-0.5)*ey[j+1];
		ez_p=(j+1.5-x[k])*ez[j]+(x[k]-j-0.5)*ez[j+1];
		
		// relocation from full- to half-integer grid points is needed for By and Bz: interpolation from half-integer grid-points
		by_p=(j+1.5-x[k])*0.5*(by[j]+by[j+1])+(x[k]-j-0.5)*0.5*(by[j+1]+by[j+2]);
		bz_p=(j+1.5-x[k])*0.5*(bz[j]+bz[j+1])+(x[k]-j-0.5)*0.5*(bz[j+1]+bz[j+2]);
		
		// check if particle 'k' is electron or ion
		if (k<np)
		{
			// electron
			qm=qme;
		}
		else
		{
			// ion
			qm=qmi;
		}
		
		// factor proportional to E-field
		// see Notebook 5, pg. 26 for details
		// note that the external electric field is added to the internal one at this step
		eps_x=qm*0.5*(ex_p+ex0);
		eps_y=qm*0.5*(ey_p+ey0);
		eps_z=qm*0.5*(ez_p+ez0);
		
		// compute the relativistic factor gamma
		// see page 26 on Notebook 5 for more details
		gamma_p=1.0/sqrt(1.0-(pow(vx[k],2.0)+pow(vy[k],2.0)+pow(vz[k],2.0))/pow(c,2.0));
		
		// compute u1 as shown in Notebook 5, page 26
		ux1=gamma_p*vx[k]+eps_x;
		uy1=gamma_p*vy[k]+eps_y;
		uz1=gamma_p*vz[k]+eps_z;
		
		// compute the new relativistic factor
		gamma_p=sqrt(1.0+(pow(ux1,2.0)+pow(uy1,2.0)+pow(uz1,2.0))/pow(c,2.0));
			
		// factor proportional to B-field
		// see Notebook 5, pg. 26 for details
		// note that the external magnetic field is added to the internal one at this step	
		beta_x=(qm*0.5/gamma_p)*(bx0/c);
		beta_y=(qm*0.5/gamma_p)*((by_p+by0)/c);
		beta_z=(qm*0.5/gamma_p)*((bz_p+bz0)/c);
		
		// intermediate quantity: Notebook 5, page 25
		wx=ux1+uy1*beta_z-uz1*beta_y;
		wy=uy1+uz1*beta_x-ux1*beta_z;
		wz=uz1+ux1*beta_y-uy1*beta_x;
		
		// compute u2 as shown in Notebook 5, page 26
		ux2=ux1+(2/(1+pow(beta_x,2.0)+pow(beta_y,2.0)+pow(beta_z,2.0)))*(wy*beta_z-wz*beta_y);
		uy2=uy1+(2/(1+pow(beta_x,2.0)+pow(beta_y,2.0)+pow(beta_z,2.0)))*(wz*beta_x-wx*beta_z);
		uz2=uz1+(2/(1+pow(beta_x,2.0)+pow(beta_y,2.0)+pow(beta_z,2.0)))*(wx*beta_y-wy*beta_x);
	
		// compute u as shown in Notebook 5, page 26
		ux=ux2+eps_x;
		uy=uy2+eps_y;
		uz=uz2+eps_z;
		
		// compute the new relativistic factor
		gamma_p=sqrt(1.0+(pow(ux,2.0)+pow(uy,2.0)+pow(uz,2.0))/pow(c,2.0));		
		
		// compute the new velocity after one time-step: page 26, Notebook 5
		vx[k]=ux/gamma_p;
		vy[k]=uy/gamma_p;
		vz[k]=uz/gamma_p;
		
		// move the particle over one time-step: page 25 on Notebook 5
		x[k]=x[k]+vx[k];
	}
}
