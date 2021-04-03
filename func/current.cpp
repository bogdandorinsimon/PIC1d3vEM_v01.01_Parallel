// This function computes the CURRENT DENSITY at a given time for all grid cells in the simulation domain

#include "../utils/nr3.h"

void current(VecDoub &jxe_s, VecDoub &jye_s, VecDoub &jze_s, VecDoub &jxi_s, VecDoub &jyi_s, VecDoub &jzi_s, VecDoub x, \
             VecDoub vx, VecDoub vy, VecDoub vz, double qse, double qsi, int np, int m)
{
	int i, j, k;
	double xp1, xp2, xp;
	VecDoub jxe(m), jye(m), jze(m), jxi(m), jyi(m), jzi(m);
		
	// initialize current vectors with zeros
	for (i=0; i<m; i++)
	{
		jxe[i]=0.;
		jye[i]=0.;
		jze[i]=0.;
		
		jxi[i]=0.;
		jyi[i]=0.;
		jzi[i]=0.;
		
		jxe_s[i]=0.;
		jye_s[i]=0.;
		jze_s[i]=0.;
		
		jxi_s[i]=0.;
		jyi_s[i]=0.;
		jzi_s[i]=0.;		
	}

	// compute current density for electrons
	for (k=0; k<np; k++)
	{
		// particle 'k' is moving from xp1 to xp2 over one time-step
		// see Notebook 5, page 20 for details
		xp1=x[k]-vx[k];
		xp2=x[k];
	
		// particle 'k' is located between i and i+1 at time "1"
		// see diagram on page 22 of Notebook 5
		i=xp1;

		// compute Jx component
		if (xp2<i)
			jxe[i-1]=jxe[i-1]+qse*(xp2-i);
		else
			jxe[i-1]=jxe[i-1]+0.;	
			
		if (xp2<i)
			jxe[i]=jxe[i]+qse*(i-xp1);
		else if (xp2>(i+1))
			jxe[i]=jxe[i]+qse*(i+1-xp1);
		else
			jxe[i]=jxe[i]+qse*(xp2-xp1);
			
		if (xp2>(i+1))
			jxe[i+1]=jxe[i+1]+qse*(xp2-i-1);
		else
			jxe[i+1]=jxe[i+1]+0.;
			
		// particle 'k' is located at time "1.5" in xp and between j+1/2 and j+3/2
		// see diagram on page 22 of Notebook 5
		xp=(xp1+xp2)/2;
		j=xp-0.5;
		
		// compute Jy and Jz components
		// see Notebook 5, page 23 for details
		jye[j]=jye[j]+(j+1.5-xp)*qse*vy[k];
		jze[j]=jze[j]+(j+1.5-xp)*qse*vz[k];
		
		jye[j+1]=jye[j+1]+(xp-j-0.5)*qse*vy[k];
		jze[j+1]=jze[j+1]+(xp-j-0.5)*qse*vz[k];			
	}				
		
	// compute current density for ions
	for (k=np; k<2*np; k++)
	{
		// particle 'k' is moving from xp1 to xp2 over one time-step
		// see Notebook 5, page 20 for details
		xp1=x[k]-vx[k];
		xp2=x[k];
	
		// particle 'k' is located between i and i+1 at time "1"
		// see diagram on page 22 of Notebook 5
		i=xp1;

		// compute Jx component
		if (xp2<i)
			jxi[i-1]=jxi[i-1]+qsi*(xp2-i);
		else
			jxi[i-1]=jxi[i-1]+0.;
			
		if (xp2<i)
			jxi[i]=jxi[i]+qsi*(i-xp1);
		else if (xp2>(i+1))
			jxi[i]=jxi[i]+qsi*(i+1-xp1);
		else
			jxi[i]=jxi[i]+qsi*(xp2-xp1);
			
		if (xp2>(i+1))
			jxi[i+1]=jxi[i+1]+qsi*(xp2-i-1);
		else
			jxi[i+1]=jxi[i+1]+0.;
			
		// particle 'k' is located at time "1.5" in xp and between j+1/2 and j+3/2
		// see diagram on page 22 of Notebook 5
		xp=(xp1+xp2)/2;
		j=xp-0.5;
		
		// compute Jy and Jz components
		// see Notebook 5, page 23 for details
		jyi[j]=jyi[j]+(j+1.5-xp)*qsi*vy[k];
		jzi[j]=jzi[j]+(j+1.5-xp)*qsi*vz[k];
		
		jyi[j+1]=jyi[j+1]+(xp-j-0.5)*qsi*vy[k];
		jzi[j+1]=jzi[j+1]+(xp-j-0.5)*qsi*vz[k];			
	}
		
	// apply periodic boundary conditions
	jxe[1]=jxe[1]+jxe[m-4];
	jye[1]=jye[1]+jye[m-4];
	jze[1]=jze[1]+jze[m-4];
	jxi[1]=jxi[1]+jxi[m-4];
	jyi[1]=jyi[1]+jyi[m-4];
	jzi[1]=jzi[1]+jzi[m-4];
	
	jxe[m-4]=jxe[1];
	jye[m-4]=jye[1];
	jze[m-4]=jze[1];
	jxi[m-4]=jxi[1];
	jyi[m-4]=jyi[1];
	jzi[m-4]=jzi[1];
	
	jxe[m-3]=jxe[m-3]+jxe[2];
	jye[m-3]=jye[m-3]+jye[2];
	jze[m-3]=jze[m-3]+jze[2];
	jxi[m-3]=jxi[m-3]+jxi[2];
	jyi[m-3]=jyi[m-3]+jyi[2];
	jzi[m-3]=jzi[m-3]+jzi[2];
	
	jxe[2]=jxe[m-3];
	jye[2]=jye[m-3];
	jze[2]=jze[m-3];
	jxi[2]=jxi[m-3];
	jyi[2]=jyi[m-3];
	jzi[2]=jzi[m-3];
		
	// apply smoothing procedure
	// see Notebook 5, page 43
	for (i=2; i<=m-4; i++)
	{
		// smoothing for electrons
		jxe_s[i]=0.25*jxe[i-1]+0.5*jxe[i]+0.25*jxe[i+1];
		jye_s[i]=0.25*jye[i-1]+0.5*jye[i]+0.25*jye[i+1];
		jze_s[i]=0.25*jze[i-1]+0.5*jze[i]+0.25*jze[i+1];

		// smoothing for ions		
		jxi_s[i]=0.25*jxi[i-1]+0.5*jxi[i]+0.25*jxi[i+1];
		jyi_s[i]=0.25*jyi[i-1]+0.5*jyi[i]+0.25*jyi[i+1];
		jzi_s[i]=0.25*jzi[i-1]+0.5*jzi[i]+0.25*jzi[i+1];		
	}
	
	// apply again periodicity	
	jxe_s[1]=jxe_s[m-4];
	jye_s[1]=jye_s[m-4];
	jze_s[1]=jze_s[m-4];
	jxi_s[1]=jxi_s[m-4];
	jyi_s[1]=jyi_s[m-4];
	jzi_s[1]=jzi_s[m-4];	
		
	jxe_s[m-3]=jxe_s[2];	
	jye_s[m-3]=jye_s[2];		
	jze_s[m-3]=jze_s[2];		
	jxi_s[m-3]=jxi_s[2];	
	jyi_s[m-3]=jyi_s[2];		
	jzi_s[m-3]=jzi_s[2];
}
