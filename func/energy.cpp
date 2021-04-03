// Aceasta functie se foloseste pentru calcularea energiei in timpul rularii simularii. Se calculeaza densitatea de energie cinetica,
// a campului electric si a campului magnetic la fiecare iteratie temporala. Pentru detalii vezi Notebook 5, pag. 89.

#include "../utils/nr3.h"

void energy(double &etaK, double &etaE, double &etaB, VecDoub vx, VecDoub vy, VecDoub vz, VecDoub ex, VecDoub ey, VecDoub ez, VecDoub by, VecDoub bz, \
			VecDoub jxe, VecDoub jye, VecDoub jze, VecDoub jxi, VecDoub jyi, VecDoub jzi, double ex0, double ey0, double ez0, double bx0, double by0, double bz0, \
			double qme, double qmi, double qse, double qsi, double c, int np, int m)
{
	int p, i;
	double aux_K, aux_E, aux_B, gamma_p;
	
	// DENSITATEA DE ENERGIE CINETICA
	// vezi Notebook 5, pag. 96-98
	aux_K=0.0;
	
	// electrons
	for (p=0; p<np; p++)
	{
		gamma_p=1.0/sqrt(1.0-(pow(vx[p],2.0)+pow(vy[p],2.0)+pow(vz[p],2.0))/pow(c,2.0));
		aux_K=aux_K+(gamma_p-1)*(qse/qme);
	}
	
	// ions
	for (p=np; p<2*np; p++)
	{
		gamma_p=1.0/sqrt(1.0-(pow(vx[p],2.0)+pow(vy[p],2.0)+pow(vz[p],2.0))/pow(c,2.0));
		aux_K=aux_K+(gamma_p-1)*(qsi/qmi);
	}	
	
	// densitatea de energie cinetica a intregului sistem simulat la iteratia curenta
	etaK=(pow(c,2.0)/(m-5))*aux_K;
	
	// --------------------------------------------------------------------------------------------------------------------------------
	
	// DENSITATEA DE ENERGIE A CAMPULUI ELECTRIC
	// vezi Notebook 5, pag. 98
	aux_E=0.0;
	
	// half-advance of the E-field backwards in time from the current iteration "n+1" to iteration "n+1/2" or from 
	// the current iteration "n" to iteration "n-1/2" (it is the same thing)
	for (i=2; i<m-3; i++)
	{
		ex[i]=ex[i]+0.5*(jxe[i]+jxi[i]);
		ey[i]=ey[i]+0.5*(jye[i]+jyi[i]+c*(bz[i+1]-bz[i]));
		ez[i]=ez[i]+0.5*(jze[i]+jzi[i]-c*(by[i+1]-by[i]));
	}	
	
	// apply periodic boundary conditions for the electric field
	// see Notebook 5, page 29 for more details
	ex[m-3]=ex[2];
	ey[m-3]=ey[2];
	ez[m-3]=ez[2];
	
	// sumarea contributiilor individuale ale fiecarei celule
	// se adauga si contributia externa; in versiunea curenta campul extern este uniform
	for (i=2; i<=m-3; i++)
		aux_E=aux_E+pow(ex[i]+ex0,2.0)+pow(ey[i]+ey0,2.0)+pow(ez[i]+ez0,2.0);
	
	// densitatea de energie electrica a intregului sistem simulat la iteratia curenta	
	etaE=aux_E/(2*(m-5));
	
	// --------------------------------------------------------------------------------------------------------------------------------
	
	// DENSITATEA DE ENERGIE A CAMPULUI MAGNETIC
	// vezi Notebook 5, pag. 99
	aux_B=0.0;
	
	// sumarea contributiilor individuale ale fiecarei celule
	// se adauga si contributia externa; in versiunea curenta campul extern este uniform
	for (i=2; i<=m-3; i++)
		aux_B=aux_B+pow(bx0,2.0)+pow(by[i]+by0,2.0)+pow(bz[i]+bz0,2.0);
		
	// densitatea de energie magnetica a intregului sistem simulat la iteratia curenta	
	etaB=aux_B/(2*(m-5));
}            
                 