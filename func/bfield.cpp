// This function computes the MAGNETIC FIELD at a given time for all grid cells in the simulation domain

#include "../utils/nr3.h"

void bfield(VecDoub &by, VecDoub &bz, VecDoub ey, VecDoub ez, int m, double c)
{
	int i;
	
	// compute magnetic field from Faraday's law
	// components of the Bfield are computed at a given time by knowing their values at a previous time-step
	// see Notebook 5, page 15 for more details
	// advance the magnetic field only for a half time-step: delta_t=0.5
	for (i=3; i<=m-3; i++)
	{
		by[i]=by[i]+0.5*c*(ez[i]-ez[i-1]);
		bz[i]=bz[i]-0.5*c*(ey[i]-ey[i-1]);
	}
	
	// apply periodic boundary conditions for the magnetic field
	// see Notebook 5, page 30 for more details
	by[2]=by[m-3];
	bz[2]=bz[m-3];
	
	// apply periodicity to the guard cells too
	by[1]=by[m-4];
	bz[1]=bz[m-4];
	
	by[0]=by[m-5];
	bz[0]=bz[m-5];
	
	by[m-2]=by[3];
	bz[m-2]=bz[3];
	
	by[m-1]=by[4];
	bz[m-1]=bz[4];
}
