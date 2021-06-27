// COD simulare PIC1d3vEM
// 1d3v electromagnetic explicit relativistic particle-in-cell code (x, vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz)
// VERSION 01.01
// Details are given in Notebook 5 and in code documentation

#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <ctime>
#include "utils/nr3.h"
#include "utils/myutils.h"
#include "utils/rngs.h"
#include "utils/rvgs.h"
#include "parallel_func/parallel.cuh"

using namespace std;

int main(int argc, char **argv)
{
	char inputfilename[1000], nt_string[100], formatSpec[10], fname_prtl[1000], aux_prtl[1000], fname_flds[1000], aux_flds[1000], \
	     aux_crnt[1000], fname_crnt[1000];
	int i, chpos, m, nt, np, sind_p, sind_v, niter, m_check, np_check, niter_check, n, save_data_step, parallel;
	double qme, qmi, qse, qsi, VTe0, VTi0, Uxe0, Uye0, Uze0, Uxi0, Uyi0, Uzi0, ex0, ey0, ez0, bx0, by0, bz0, c, etaK, etaE, etaB;
	double tempvar[24];
	FILE *f_flds, *f_prtl, *f_crnt, *f_enrg, *xvalues_file;
	clock_t timeCPU, timeCPU0, timeCPU_START, timeCPU_END;

	VecDoub x(1), vx(1), vy(1), vz(1), ex(1), ey(1), ez(1), by(1), bz(1), jxe(1), jye(1), jze(1), jxi(1), jyi(1), jzi(1), \
	        etaK_ALL(1), etaE_ALL(1), etaB_ALL(1);
	
	cudaError_t cudaStatus;

	// measure the time when simulation starts
	timeCPU_START=clock();
	
	// Initialization messages
	cout << "********************************************\n";
	cout << "**** RUNNING simulation code PIC1d3vEM! ****\n";
	cout << "********************************************\n";
	
	cout << "**** Input file: <" << argv[1] << ">\n";
	cout << "**** Save data step: " << argv[2] << "\n";
	cout << "**** Serial/parallel implementation: " << argv[3] << "\n";

	parallel = strcmp(argv[3], "P") == 0;
	if (!parallel & strcmp(argv[3], "S") != 0) {
		cout << "********* ERROR! - Choose S or P which stands for Serial/Parallel implementations ... now exiting to system!\n\n";
		exit(EXIT_FAILURE);
	}

	if (parallel) {
		xvalues_file = fopen("files/parallel_xvalues.x", "wb");
	}
	else {
		xvalues_file = fopen("files/serial_xvalues.x", "wb");
	}

	cout << "**** Is this a NEW run? [Y/N]: " << argv[4] << "\n";
	if (strcmp(argv[4],"N")==0)
	{
		if (argc!=7)
		{
			cout << "********* ERROR! - Specify the fields and particles files ... now exiting to system!\n\n";	
			exit(EXIT_FAILURE);			
		}
		cout << "********* This is NOT a new run!\n";
		cout << "********* Starting simulation from the previous time-step:\n";
		cout << "********* fields file: <" << argv[5] << "> and particles file: <" << argv[6] << ">\n";
	}
	
	if (strcmp(argv[4],"Y")!=0 & strcmp(argv[4],"N")!=0)
	{
		cout << "********* ERROR! - Choose Y or N ... now exiting to system!\n\n";	
		exit(EXIT_FAILURE);
	}
	
	// temporal step used to save output data during the simulation runtime
	save_data_step=strtol(argv[2],NULL,10);
	
	//--------------------------------------------------------------------------------------------------------------------------------
	// READ the content of the INPUT file
	
	// read the name of the input file
	sprintf(inputfilename,"%s",argv[1]);
	FILE *finput=fopen(inputfilename,"r");
	
	// search for '#' character into the input file
	do
    {
    	chpos=fgetc(finput);
    }
    while (chpos != '#');
    
    // read the content on the right of character "=" and after the character "#"
    i=0;
    do
    {
    	chpos=fgetc(finput);
        if (chpos == '=')
        {
        	fscanf(finput,"%lf",&tempvar[i]);
            i++;
        }
    }
    while (chpos != EOF);
	
	fclose(finput);
	
	//--------------------------------------------------------------------------------------------------------------------------------
	
	// Input parameters taken from input file
	qme=tempvar[0];     // charge-to-mass ratio for electrons: q/m
	qmi=tempvar[1];     // q/m for ions
	qse=tempvar[2];     // superparticle charge for electrons: qs
	qsi=tempvar[3];     // qs for ions
	VTe0=tempvar[4];    // thermal velocity for electrons at t=0: VT0
	VTi0=tempvar[5];    // VT0 for ions
	Uxe0=tempvar[6];    // initial bulk velocity for electrons: U0, x-component
	Uye0=tempvar[7];    // y-component
	Uze0=tempvar[8];    // z-component
	Uxi0=tempvar[9];    // U0 for ions, x-component
	Uyi0=tempvar[10];   // y-component
	Uzi0=tempvar[11];   // z-component
	ex0=tempvar[12];    // external electric field: e0, x-component
	ey0=tempvar[13];    // y-component
	ez0=tempvar[14];    // z-component
	bx0=tempvar[15];    // external magnetic field: b0=c*B0, x-component
	by0=tempvar[16];    // y-component
	bz0=tempvar[17];    // z-component
	m=tempvar[18];      // no. of grid-cells along the x-axis: simulation domain is defined from 2 to m-3
	nt=tempvar[19];     // no . of time-steps
	np=tempvar[20];     // no. of particles per species (np electrons and np ions)
	c=tempvar[21];      // speed of light in vacuum
	sind_p=tempvar[22]; // stream index for random number generator - for positions initialization (0, 1, ..., 255)
	sind_v=tempvar[23]; // stream index for random number generator - for velocities initialization (0, 1, ..., 255)
	
	// Printing the input parameters
	cout << "**** INPUT parameters:\n";
	printf("********* qme= %+.4e\n",qme);
	printf("********* qmi= %+.4e\n",qmi);
	printf("********* qse= %+.4e\n",qse);
	printf("********* qsi= %+.4e\n",qsi);
	printf("********* VTe= %+.4e\n",VTe0);	
	printf("********* VTi= %+.4e\n",VTi0);	
	printf("********* Uxe= %+.4e\n",Uxe0);	
	printf("********* Uye= %+.4e\n",Uye0);	
	printf("********* Uze= %+.4e\n",Uze0);	
	printf("********* Uxi= %+.4e\n",Uxi0);	
	printf("********* Uyi= %+.4e\n",Uyi0);	
	printf("********* Uzi= %+.4e\n",Uzi0);		
	printf("********* ex0= %+.4e\n",ex0);
	printf("********* ey0= %+.4e\n",ey0);
	printf("********* ez0= %+.4e\n",ez0);
	printf("********* bx0= %+.4e\n",bx0);
	printf("********* by0= %+.4e\n",by0);
	printf("********* bz0= %+.4e\n",bz0);	
	printf("********* m = %d\n",m);
	printf("********* nt= %d\n",nt);	
	printf("********* np= %d\n",np);
	printf("********* c = %f\n",c);
	printf("********* sp= %d\n",sind_p);
	printf("********* sv= %d\n",sind_v);
	
	// Create some auxiliary variables that are needed to write the output files at various time-steps
	sprintf(nt_string,"%d",nt);
    sprintf(formatSpec,"%%0%dd",strlen(nt_string));	
    sprintf(aux_prtl,"files/prtl_%s.d",formatSpec);
    sprintf(aux_flds,"files/flds_%s.d",formatSpec);
    sprintf(aux_crnt,"files/crnt_%s.d",formatSpec);
	
	//--------------------------------------------------------------------------------------------------------------------------------
	// INITIALIZATION section
	
	if (strcmp(argv[4],"Y")==0)
	{
		// NEW SIMULATION
	
		cout << "**** Starting INITIALIZATION procedure\n";
	
		// Initialize POSITIONS - electrons and ions uniformly distributed in the interval [2,m-3]
		// set 'x'-size to have '2*np' elements: FIRST 'np eletrons' and SECOND 'np ions'
		// see Notebook 5, page 32 for details
		x.resize(2*np);
	
		// set the stream of the random number generator - for positions
		SelectStream(sind_p);
	
		// generate positions of electrons and ions
		for (i=0; i<np; i++)
		{
			x[i]=2+(m-5)*Random();
			x[i+np]=x[i];
		}
	
		// Initialize VELOCITIES - electrons and ions normally distributed in the velocity space
		// set 'vx, vy, vz'-size to have '2*np' elements: FIRST 'np eletrons' and SECOND 'np ions'
		// see Notebook 5, page 33 for details	
		vx.resize(2*np);
		vy.resize(2*np);
		vz.resize(2*np);
	
		// set the stream of the random number generator - for velocities
		SelectStream(sind_v);	
	
		// generate the velocities of the electrons and ions
		for (i=0; i<np; i++)
		{
			vx[i]=Normal(Uxe0,VTe0/sqrt(2));
			vy[i]=Normal(Uye0,VTe0/sqrt(2));
			vz[i]=Normal(Uze0,VTe0/sqrt(2));
		
			vx[i+np]=Normal(Uxi0,VTi0/sqrt(2));
			vy[i+np]=Normal(Uyi0,VTi0/sqrt(2));
			vz[i+np]=Normal(Uzi0,VTi0/sqrt(2));		
		}

		// Initialize FIELDS - electromagnetic field is set to zero initially (this is the self-consistent one)
		// for details see Notebook 5, pag. 35
		ex.assign(m,0.0);
		ey.assign(m,0.0);
		ez.assign(m,0.0);
		by.assign(m,0.0);
		bz.assign(m,0.0);
	
		cout << "**** INITIALIZATION procedure COMPLETED!\n";
		
		// Writing the initialization data to files		
        // build the names of the output files corresponding to t=0
        // fields file and particles file at t=0
        sprintf(fname_prtl,aux_prtl,0);
        sprintf(fname_flds,aux_flds,0);        
        
        // iteratia curenta corespunzatoare lui t=0
        niter=0;
        
        // write particles file
        // see Notebook 5, pag. 37 for details
        f_prtl=fopen(fname_prtl,"wb");
        	fwrite(&niter,sizeof(int),1,f_prtl);
        	fwrite(&np,sizeof(int),1,f_prtl);
        	fwrite(&x[0],sizeof(double),2*np,f_prtl);
        	fwrite(&vx[0],sizeof(double),2*np,f_prtl);
        	fwrite(&vy[0],sizeof(double),2*np,f_prtl);
        	fwrite(&vz[0],sizeof(double),2*np,f_prtl);      
        fclose(f_prtl);
        
        // write fields file
        // see Notebook 5, pag. 37 for details
        f_flds=fopen(fname_flds,"wb");
            fwrite(&niter,sizeof(int),1,f_flds);
        	fwrite(&m,sizeof(int),1,f_flds);
        	fwrite(&ex[0],sizeof(double),m,f_flds);
        	fwrite(&ey[0],sizeof(double),m,f_flds);
        	fwrite(&ez[0],sizeof(double),m,f_flds);
        	fwrite(&by[0],sizeof(double),m,f_flds);
        	fwrite(&bz[0],sizeof(double),m,f_flds);
        fclose(f_flds);     
	}
	else if (strcmp(argv[4],"N")==0)
	{
		// STARTING from a PREVIOUS simulation
	
		cout << "**** Reading the output files from the previous saved time-step\n";
		cout << "**** Simulation INITIALIZED based on the output files from the previous time-step\n";
		
		// set the size of the field vectors to have 'm' elements
		ex.resize(m);
		ey.resize(m);
		ez.resize(m);
		by.resize(m);
		bz.resize(m);
	
		// read the fields file	
		f_flds=fopen(argv[5],"r");
			fread(&niter,sizeof(int),1,f_flds);
			fread(&m_check,sizeof(int),1,f_flds);
			fread(&ex[0],sizeof(double),m,f_flds);
        	fread(&ey[0],sizeof(double),m,f_flds);
        	fread(&ez[0],sizeof(double),m,f_flds);
        	fread(&by[0],sizeof(double),m,f_flds);
        	fread(&bz[0],sizeof(double),m,f_flds);
		fclose(f_flds);
		
		// set the size of the particle vectors to have '2*np' elements
		x.resize(2*np);
		vx.resize(2*np);
		vy.resize(2*np);
		vz.resize(2*np);
		
		// read the particles file
		f_prtl=fopen(argv[6],"r");
        	fread(&niter_check,sizeof(int),1,f_prtl);
        	fread(&np_check,sizeof(int),1,f_prtl);
        	fread(&x[0],sizeof(double),2*np,f_prtl);
        	fread(&vx[0],sizeof(double),2*np,f_prtl);
        	fread(&vy[0],sizeof(double),2*np,f_prtl);
        	fread(&vz[0],sizeof(double),2*np,f_prtl); 			
		fclose(f_prtl);
		
		// check that the output files are provided at the same time-step
		if (niter_check==niter)
		{
			cout << "**** Starting simulation from time-step: " << niter << "\n";
		}
		else
		{
			cout << "********* ERROR! - Files NOT provided at the SAME time-step ... now exiting to system!\n\n";	
			exit(EXIT_FAILURE);
		}
				
		cout << "**** INITIALIZATION procedure COMPLETED!\n";
	}
	
	//--------------------------------------------------------------------------------------------------------------------------------
	// ITERATIVE section of the simulation code (temporal iterations)
	
	// set the size of the current vectors to have 'm' elements
	jxe.resize(m);
	jye.resize(m);
	jze.resize(m);	
	jxi.resize(m);
	jyi.resize(m);
	jzi.resize(m);
	
	// set the size of the energy vectors to have 'nt+1' elements
	etaK_ALL.assign(nt+1,0.0);
	etaE_ALL.assign(nt+1,0.0);
	etaB_ALL.assign(nt+1,0.0);
	
	// compute the initial current density and the initial energy
	if (strcmp(argv[4],"Y")==0)
	{
		// current density computation
		current(jxe, jye, jze, jxi, jyi, jzi, x, vx, vy, vz, qse, qsi, np, m);
		
		// calcularea densitatii de energie
		energy(etaK, etaE, etaB, vx, vy, vz, ex, ey, ez, by, bz, jxe, jye, jze, jxi, jyi, jzi, ex0, ey0, ez0, bx0, by0, bz0, qme, qmi, qse, qsi, c, np, m);
		etaK_ALL[0]=etaK;
		etaE_ALL[0]=etaE;
		etaB_ALL[0]=etaB;		
		
		// write current file
        // see Notebook 5, pag. 54 for details
        sprintf(fname_crnt,aux_crnt,0);
        f_crnt=fopen(fname_crnt,"wb");
            fwrite(&niter,sizeof(int),1,f_crnt);
        	fwrite(&m,sizeof(int),1,f_crnt);
        	fwrite(&jxe[0],sizeof(double),m,f_crnt);
        	fwrite(&jye[0],sizeof(double),m,f_crnt);
        	fwrite(&jze[0],sizeof(double),m,f_crnt);
        	fwrite(&jxi[0],sizeof(double),m,f_crnt);
        	fwrite(&jyi[0],sizeof(double),m,f_crnt);
        	fwrite(&jzi[0],sizeof(double),m,f_crnt);
        fclose(f_crnt);
	}	
	
	// starting from 'niter+1' iteration (based on input from 'niter' iteration)
	// last iteration is 'nt'
	// see Notebook 5, page 42 for flow chart
	cout << "**** Starting ITERATIVE procedure (temporal iterations)\n";
	cout << "********* Simulation IN PROGRESS ...\n";
	cout << "*********\n";
	
	// starting time of iterative process
	timeCPU0=clock();
	timeCPU=timeCPU0;
	for (n=niter+1; n<=nt; n++)
	{
		if (parallel) {
			cudaStatus = bfieldWithCuda(&by[0], &bz[0], &ey[0], &ez[0], m, c);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "bfieldWithCuda failed: %s\n", cudaGetErrorString(cudaStatus));
				return EXIT_FAILURE;
			}

			
			cudaStatus = moverWithCuda(&x[0], &vx[0], &vy[0], &vz[0], &ex[0], &ey[0], &ez[0], &by[0], &bz[0], ex0, ey0, ez0, bx0, by0, bz0, qme, qmi, c, np, m);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "moverWithCuda failed: %s\n", cudaGetErrorString(cudaStatus));
				return EXIT_FAILURE;
			}

			cudaStatus = bfieldWithCuda(&by[0], &bz[0], &ey[0], &ez[0], m, c);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "bfielWithCuda failed: %s\n", cudaGetErrorString(cudaStatus));
				return EXIT_FAILURE;
			}

			cudaStatus = currentWithCuda(&jxe[0], &jye[0], &jze[0], &jxi[0], &jyi[0], &jzi[0], &x[0], &vx[0], &vy[0], &vz[0], qse, qsi, np, m);
			if(cudaStatus != cudaSuccess) {
				fprintf(stderr, "currentWithCuda failed: %s\n", cudaGetErrorString(cudaStatus));
				return EXIT_FAILURE;
			}

			cudaStatus = efieldWithCuda(&ex[0], &ey[0], &ez[0], &by[0], &bz[0], &jxe[0], &jye[0], &jze[0], &jxi[0], &jyi[0], &jzi[0], m, c);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "efieldWithCuda failed: %s\n", cudaGetErrorString(cudaStatus));
				return EXIT_FAILURE;
			}
		} else {
			// half-advance of the magnetic field
			bfield(by, bz, ey, ez, m, c);

			// push particles (positions and velocities) over one time-step
			mover(x, vx, vy, vz, ex, ey, ez, by, bz, ex0, ey0, ez0, bx0, by0, bz0, qme, qmi, c, np, m);

			// half-advance of the magnetic field
			bfield(by, bz, ey, ez, m, c);

			// current density computation
			current(jxe, jye, jze, jxi, jyi, jzi, x, vx, vy, vz, qse, qsi, np, m);
		
			// full-advance of the electric field
			efield(ex, ey, ez, by, bz, jxe, jye, jze, jxi, jyi, jzi, m, c);
		}	

		// apply periodicity for particles
		for (i=0; i<2*np; i++)
		{
			if (x[i]<2)
				x[i]=x[i]+m-5;
			else if (x[i]>=(m-3))
				x[i]=x[i]-(m-5);	
		}
		
		// calcularea densitatii de energie la iteratia curenta: cinetica, electrica, magnetica
		// vezi Notebook 5, pag. 89
		energy(etaK, etaE, etaB, vx, vy, vz, ex, ey, ez, by, bz, jxe, jye, jze, jxi, jyi, jzi, ex0, ey0, ez0, bx0, by0, bz0, qme, qmi, qse, qsi, c, np, m);
		
		// salvarea energiei la fiecare iteratie temporala
		etaK_ALL[n]=etaK;
		etaE_ALL[n]=etaE;
		etaB_ALL[n]=etaB;
		
		// save simulation data to output files
		// data are saved for n = save_data_step, 2*save_data_step, ...
		if (n % save_data_step == 0)
		{
			// save data for testing purposes
			fwrite(&x[0], sizeof(double), 2 * np, xvalues_file);

			// build the names of the output files corresponding to iteration "n"
        	sprintf(fname_prtl,aux_prtl,n);
        	sprintf(fname_flds,aux_flds,n);
        	sprintf(fname_crnt,aux_crnt,n);
        	
			// write particles file
        	// see Notebook 5, pag. 37 for details
        	f_prtl=fopen(fname_prtl,"wb");
        		fwrite(&n,sizeof(int),1,f_prtl);
        		fwrite(&np,sizeof(int),1,f_prtl);
        		fwrite(&x[0],sizeof(double),2*np,f_prtl);
        		fwrite(&vx[0],sizeof(double),2*np,f_prtl);
        		fwrite(&vy[0],sizeof(double),2*np,f_prtl);
        		fwrite(&vz[0],sizeof(double),2*np,f_prtl);      
        	fclose(f_prtl);
        
         	// write fields file
        	// see Notebook 5, pag. 37 for details
        	f_flds=fopen(fname_flds,"wb");
            	fwrite(&n,sizeof(int),1,f_flds);
        		fwrite(&m,sizeof(int),1,f_flds);
        		fwrite(&ex[0],sizeof(double),m,f_flds);
        		fwrite(&ey[0],sizeof(double),m,f_flds);
        		fwrite(&ez[0],sizeof(double),m,f_flds);
        		fwrite(&by[0],sizeof(double),m,f_flds);
        		fwrite(&bz[0],sizeof(double),m,f_flds);
        	fclose(f_flds);  
			
			// write current file
        	// see Notebook 5, pag. 54 for details
        	f_crnt=fopen(fname_crnt,"wb");
            	fwrite(&n,sizeof(int),1,f_crnt);
        		fwrite(&m,sizeof(int),1,f_crnt);
        		fwrite(&jxe[0],sizeof(double),m,f_crnt);
        		fwrite(&jye[0],sizeof(double),m,f_crnt);
        		fwrite(&jze[0],sizeof(double),m,f_crnt);
        		fwrite(&jxi[0],sizeof(double),m,f_crnt);
        		fwrite(&jyi[0],sizeof(double),m,f_crnt);
        		fwrite(&jzi[0],sizeof(double),m,f_crnt);
        	fclose(f_crnt);
        	
        	// write energy file
        	// see Notebook 5, pag. 100 for details
        	f_enrg=fopen("files/energy.d","wb");
        		fwrite(&etaK_ALL[0],sizeof(double),nt+1,f_enrg);
        		fwrite(&etaE_ALL[0],sizeof(double),nt+1,f_enrg);
        		fwrite(&etaB_ALL[0],sizeof(double),nt+1,f_enrg);
        	fclose(f_enrg);
		}	
		
		// messages about the simulation progress
		if (int(0.05*nt)>=1)
		{
			if (n % int(0.05*nt) == 0)
			{
				cout << "********* " << n-niter << " of " << nt-niter << " iterations completed [" << 100*(float(n-niter))/(float(nt-niter)) << "%]\n";
				cout << "*********    in " << (float(clock()-timeCPU0))/CLOCKS_PER_SEC << " sec. or " << \
				        ((float(clock()-timeCPU0))/CLOCKS_PER_SEC)/3600 << " hours or " << \
				        (((float(clock()-timeCPU0))/CLOCKS_PER_SEC)/3600)/24 << " days\n";
				cout << "********* it took " << (float(clock()-timeCPU))/CLOCKS_PER_SEC << " sec. or " << \
						((float(clock()-timeCPU))/CLOCKS_PER_SEC)/3600 << " hours or " << \
						(((float(clock()-timeCPU))/CLOCKS_PER_SEC)/3600)/24 << " days\n" << \
						"*********    for the last " << int(0.05*nt) << " iterations\n";
				cout << "********* TTC is " << (nt-n)*((float(clock()-timeCPU))/CLOCKS_PER_SEC)/(int(0.05*nt)) << " sec. or " << \
						((nt-n)*((float(clock()-timeCPU))/CLOCKS_PER_SEC)/(int(0.05*nt)))/3600 << " hours or " << \
						(((nt-n)*((float(clock()-timeCPU))/CLOCKS_PER_SEC)/(int(0.05*nt)))/3600)/24 << " days\n";
				cout << "*********\n";		
				timeCPU=clock();
			}
		}
	}
	
	cout << "**** ITERATIVE procedure COMPLETED!\n";
	
	//--------------------------------------------------------------------------------------------------------------------------------
	
	// measure the time when simulation stops
	timeCPU_END=clock();
	
	// printing total simulation time
	cout << "**** Total COMPLETION TIME is " << (float(timeCPU_END-timeCPU_START))/CLOCKS_PER_SEC << " sec. or " << \
			((float(timeCPU_END-timeCPU_START))/CLOCKS_PER_SEC)/3600 << " hours or " << \
			(((float(timeCPU_END-timeCPU_START))/CLOCKS_PER_SEC)/3600)/24 << " days\n";
	
	// CLOSING MESSAGE
	cout << "*********************************************************\n";
	cout << "**** SIMULATION COMPLETED SUCCESSFULLY - END OF RUN! ****\n";
	cout << "*********************************************************\n\n";

	fclose(xvalues_file);
	const unsigned NUMBER_OF_PARTICLES = 2 * np;
	double *p_xvalues = (double*)malloc(NUMBER_OF_PARTICLES * sizeof(double));
	double *s_xvalues = (double*)malloc(NUMBER_OF_PARTICLES * sizeof(double));

	FILE* p_xvalues_file = fopen("files/parallel_xvalues.x", "rb");
	FILE* s_xvalues_file = fopen("files/serial_xvalues.x", "rb");

	fread(p_xvalues, sizeof(double), 2 * np, p_xvalues_file);
	fread(s_xvalues, sizeof(double), 2 * np, s_xvalues_file);

	fclose(p_xvalues_file);
	fclose(s_xvalues_file);

	int count_diff = 0;
	double err = 0.0;
	double err2 = 0.0;

	for (int i = 0; i < 2 * np; i++) {
		if (p_xvalues[i] != s_xvalues[i]) {
			count_diff++; 
		}

		err = err + fabs(p_xvalues[i] - s_xvalues[i]);
	}
	printf("\n # of Different values: %i . Err: %.17g", count_diff, err);

	getchar();
}
