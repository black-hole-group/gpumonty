/*
   Using monte carlo method, estimate spectrum of an appropriately
   scaled axisymmetric GRMHD simulation as a function of
   latitudinal viewing angle.

   Input simulation data is assumed to be in dump format provided by
   HARM code.  Location of input file is, at present, hard coded
   (see init_sim_data.c).

   Nph super-photons are generated in total and then allowed
   to propagate.  They are weighted according to the emissivity.
   The photons are pushed by the geodesic equation.
   Their weight decays according to the local absorption coefficient.
   The photons also scatter with probability related to the local
   scattering opacity.

   The electrons are assumed to have a thermal distribution
   function, and to be at the same temperature as the protons.
 */

#include "decs.h"
#include "harm_model.h"
#include "gpu_utils.h"
#include <time.h>

#define NPHOTONS_BIAS 23 // Expected photons to be produces equals NPHOTONS_BIAS * Ns
#define MIN2(fst, snd) ((fst) <= (snd) ? (fst) : (snd))

// Declare external global variables
struct of_geom **geom;
double Ladv, dMact, bias_norm;
double gam;
double dMsim;
double M_unit;
double T_unit;
double RHO_unit;
double U_unit;
double startx[NDIM], stopx[NDIM], dx[NDIM];
double B_unit;
double max_tau_scatt;
double Ne_unit;
double Thetae_unit;
int N1, N2;
__device__ double d_startx[NDIM], d_stopx[NDIM], d_dx[NDIM];
__device__ double d_B_unit;
__device__ double d_max_tau_scatt;
__device__ double d_Ne_unit;
__device__ double d_Thetae_unit;
__device__ int d_N1, d_N2;
__device__ unsigned long long d_N_superph_made;


void terminate (const char *msg, int code) {
	fprintf(stderr, "%s\n", msg);
	exit(code);
}

unsigned long long generate_photons (unsigned long long Ns, struct of_photon **phs) {
	unsigned long long phs_max = Ns * NPHOTONS_BIAS;
	unsigned long long ph_count = 0;
	unsigned long long n2gen;
	double dnmax;

	*phs = (struct of_photon *) malloc(phs_max * sizeof(struct of_photon));

	for (int zi = 0; zi < N1; zi++) {
		for (int zj = 0; zj < N2; zj++) {
			init_zone(zi, zj, &n2gen, &dnmax, Ns);
			if (ph_count + n2gen >= phs_max) {
				phs_max = MIN2(ph_count + n2gen * (N1 - zi) * (N2 - zj),
							   2 * (ph_count + n2gen));
				*phs = (struct of_photon *) realloc(*phs, phs_max *
													sizeof(struct of_photon));
			}
			for (unsigned long long gen = 0; gen < n2gen; gen++) {
				//small trick: !gen equals (gen == 0 ? 1 : 0)
				sample_zone_photon(zi, zj, dnmax, &((*phs)[ph_count]), !gen);
				ph_count++;
			}
		}
	}
	//trim excedent memory
	*phs = (struct of_photon *) realloc(*phs, ph_count*sizeof(struct of_photon));
	return ph_count;
}

void check_env_vars(int *NUM_BLOCKS, int *BLOCK_SIZE) {
	char *num_blocks_str = getenv("NUM_BLOCKS");
	char *block_size_str = getenv("BLOCK_SIZE");

	if (num_blocks_str && (sscanf(num_blocks_str, "%d", NUM_BLOCKS) != 1))
		terminate("Invalid argument for NUM_BLOCKS environment variable.",
				  EINVAL);

	if (block_size_str && (sscanf(block_size_str, "%d", BLOCK_SIZE) != 1))
		terminate("Invalid argument for BLOCK_SIZE environment variable.",
				  EINVAL);
}

void check_args (int argc, char *argv[], unsigned long long *Ns, unsigned long *seed) {
	if (argc < 4 || argc > 5) terminate("usage: grmonty Ns infilename M_unit [seed]\nWhere seed >= 1", 0);
	if (argc == 5) {
		sscanf(argv[4], "%lu", seed);
		if (*seed < 1) terminate("error: seed must be >= 1\nusage: grmonty Ns infilename M_unit [seed]", 0);
	}
	else *seed = 139 + time(NULL); /* Arbitrarily picked initial seed */
	sscanf(argv[1], "%llu", Ns);
}

int main(int argc, char *argv[]) {
	unsigned long long Ns, N_superph_made;
	struct of_photon *phs, *d_phs;
	curandState_t *d_curandstates;
	unsigned long seed;
	time_t currtime, starttime;

	// Initializations and initial assignments
	check_args(argc, argv, &Ns, &seed);
	harm_rng_init(seed);
	init_spectrum();
	init_model(argv); /* initialize model data, auxiliary variables */
	starttime = time(NULL);


	fprintf(stderr, "Generating photons...\n\n");
	fflush(stderr);
	N_superph_made = generate_photons(Ns, &phs);


	int BLOCK_SIZE = 256;
	int NUM_BLOCKS = 20;
	check_env_vars(&NUM_BLOCKS, &BLOCK_SIZE);
	fprintf(stderr, "Kenels-config: %d BLOCKS of %d THREADS.\n\n", NUM_BLOCKS, BLOCK_SIZE);
	fflush(stderr);

	fprintf(stderr, "Copying photons to GPU and initializing RNG...\n\n");
	fflush(stderr);
	// Copy phs and curandstates to GPU
	CUDASAFE(cudaMemcpyToSymbol(d_N_superph_made, &N_superph_made, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
	CUDASAFE(cudaMalloc(&d_curandstates, BLOCK_SIZE * NUM_BLOCKS * sizeof(curandState_t)));
	gpu_rng_init<<<BLOCK_SIZE, NUM_BLOCKS>>>(d_curandstates, seed);
	CUDAERRCHECK();
	CUDASAFE(cudaMalloc(&d_phs, N_superph_made * sizeof(struct of_photon)));
	CUDASAFE(cudaMemcpy(d_phs, phs, N_superph_made * sizeof(struct of_photon),
						cudaMemcpyHostToDevice));

	fprintf(stderr, "Entering main loop...\n\n");
	fflush(stderr);
	int batchsize = BLOCK_SIZE * NUM_BLOCKS;
	for(int offset = 0; offset + batchsize < N_superph_made; offset += batchsize) {
		track_super_photon<<<BLOCK_SIZE, NUM_BLOCKS>>>(d_curandstates, d_phs, offset);
		CUDAERRCHECK();
		CUDASAFE(cudaDeviceSynchronize());
	}

	currtime = time(NULL);
	fprintf(stderr, "Final time %g, rate %g ph/s\n",
	(double) (currtime - starttime),
	(double) N_superph_made / (currtime - starttime));

	copy_spect_from_gpu();
	report_spectrum(N_superph_made);

	free(phs);

	return 0;

}
