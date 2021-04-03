// This function computes the ELECTRIC FIELD at a given time for all grid cells in the simulation domain

#include "../utils/nr3.h"

void efield(VecDoub &ex, VecDoub &ey, VecDoub &ez, VecDoub by, VecDoub bz, VecDoub jxe, VecDoub jye, VecDoub jze, \
            VecDoub jxi, VecDoub jyi, VecDoub jzi, int m, double c)
{
	int i;
	
	// compute electric field from Ampere's law
	// components of the Efield are computed at a given time by knowing their values at a previous time-step
	// see Notebook 5, page 15 for more details
	for (i=2; i<m-3; i++)
	{
		ex[i]=ex[i]-(jxe[i]+jxi[i]);
		ey[i]=ey[i]-(jye[i]+jyi[i])-c*(bz[i+1]-bz[i]);
		ez[i]=ez[i]-(jze[i]+jzi[i])+c*(by[i+1]-by[i]);
	}
	
	// apply periodic boundary conditions for the electric field
	// see Notebook 5, page 29 for more details
	ex[m-3]=ex[2];
	ey[m-3]=ey[2];
	ez[m-3]=ez[2];
	
	// apply periodicity to the guard cells too
	ex[1]=ex[m-4];
	ey[1]=ey[m-4];
	ez[1]=ez[m-4];
	
	ex[0]=ex[m-5];
	ey[0]=ey[m-5];
	ez[0]=ez[m-5];
	
	ex[m-2]=ex[3];
	ey[m-2]=ey[3];
	ez[m-2]=ez[3];
	
	ex[m-1]=ex[4];
	ey[m-1]=ey[4];
	ez[m-1]=ez[4];
}
