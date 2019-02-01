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
#include <omp.h>

#define NPHOTONS_BIAS 23 // Expected photons to be produces equals NPHOTONS_BIAS * Ns
#define MIN2(fst, snd) ((fst) <= (snd) ? (fst) : (snd))

// Declare external global variables
int BLOCK_SIZE = 512;
int NUM_BLOCKS = 30;
int N_CPU_THS = 8;
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

void terminate (const char *msg, int code) {
	fprintf(stderr, "%s\n", msg);
	fflush(stderr);
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

void check_env_vars() {
	char *num_blocks_str = getenv("NUM_BLOCKS");
	char *block_size_str = getenv("BLOCK_SIZE");
	char *n_cpu_ths_str = getenv("N_CPU_THS");

	if (num_blocks_str && (sscanf(num_blocks_str, "%d", NUM_BLOCKS) != 1 ||
		NUM_BLOCKS <= 0))
		terminate("Invalid argument for NUM_BLOCKS environment variable.",
				  EINVAL);

	if (block_size_str && (sscanf(block_size_str, "%d", BLOCK_SIZE) != 1 ||
		BLOCK_SIZE <= 0))
		terminate("Invalid argument for BLOCK_SIZE environment variable.",
				  EINVAL);

	if (n_cpu_ths_str && (sscanf(n_cpu_ths_str, "%d", N_CPU_THS) != 1 ||
		N_CPU_THS <= 0))
		terminate("Invalid argument for N_CPU_THS environment variable.",
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

void core_photon_tracking(struct of_photon *phs, unsigned long long N_superph_made) {
	cudaStream_t batch_cpy_stream;
	CUDASAFE(cudaStreamCreate(&batch_cpy_stream));

	struct of_photon *d_batch_1, *d_batch_2;
	CUDASAFE(cudaMalloc(&d_batch_1, N_GPU_THS * sizeof(struct of_photon)));
	CUDASAFE(cudaMalloc(&d_batch_2, N_GPU_THS * sizeof(struct of_photon)));

	unsigned int N_1 = (unsigned int) MIN2(N_GPU_THS, N_superph_made);
	unsigned int N_2;
	CUDASAFE(cudaMemcpy(d_batch_1, phs, N_1 * sizeof(struct of_photon),
						cudaMemcpyHostToDevice));
	track_super_photon_batch<<<BLOCK_SIZE, NUM_BLOCKS>>>(d_batch_1, N_1);
	CUDAERRCHECK();

	unsigned long long offset_1 = 0, offset_2 = N_1;

	while (offset_2 < N_superph_made) {

		N_2 = (unsigned int) MIN2(N_GPU_THS, N_superph_made - offset_2);
		CUDASAFE(cudaMemcpyAsync(d_batch_2, phs + offset_2, N_2 *
								 sizeof(struct of_photon),
								 cudaMemcpyHostToDevice, batch_cpy_stream));
		CUDASAFE(cudaDeviceSynchronize()); // wait previous kernel launch and asyncmemcpy

		track_super_photon_batch<<<BLOCK_SIZE, NUM_BLOCKS>>>(d_batch_2, N_2);

		CUDASAFE(cudaMemcpyAsync(phs + offset_1, d_batch_1, N_1 *
								 sizeof(struct of_photon),
								 cudaMemcpyDeviceToHost, batch_cpy_stream));
		CUDASAFE(cudaStreamSynchronize(batch_cpy_stream));

		// Add batch_1 contribution to spect
		// #pragma omp parallel for schedule(static)
		for (unsigned int i = 0; i < N_1; ++i) {
			struct of_photon *ph = &phs[offset_1 + i];
			if (ph->tracking_status == TRACKING_STATUS_COMPLETE &&
				record_criterion(ph)) record_super_photon(ph);
		}

		// Calculate remainings of batch_1

		// Flip batch 1 and 2
		struct of_photon *d_batch_aux = d_batch_1;
		d_batch_1 = d_batch_2;
		d_batch_2 = d_batch_aux;

		offset_1 = offset_2;
		offset_2 = offset_2 + N_2;

		N_1 = N_2;
	}

	// Wait last kernel launch and bring the batch back
	CUDASAFE(cudaMemcpy(phs + offset_1, d_batch_1, N_1 *
						sizeof(struct of_photon), cudaMemcpyDeviceToHost));

	// Add batch_1 contribution to spect
	// #pragma omp parallel for schedule(static)
	for (unsigned int i = 0; i < N_1; ++i) {
		struct of_photon *ph = &phs[offset_1 + i];
		if (record_criterion(ph)) record_super_photon(ph);
	}

	// Calculate remainings of batch_1
}

int main(int argc, char *argv[]) {
	unsigned long long Ns, N_superph_made;
	struct of_photon *phs;
	unsigned long seed;
	time_t currtime, starttime;

	// Initializations and initial assignments
	check_args(argc, argv, &Ns, &seed);
	rng_init(seed);
	init_spectrum();
	init_model(argv); /* initialize model data, auxiliary variables */
	starttime = time(NULL);

	fprintf(stderr, "Generating photons...\n\n");
	fflush(stderr);
	N_superph_made = generate_photons(Ns, &phs);

	check_env_vars();
	omp_set_num_threads(N_CPU_THS);
	fprintf(stderr, "Config:\n");
	fprintf(stderr, "  CUDA: %d blocks of %d threads\n", NUM_BLOCKS, BLOCK_SIZE);
	fprintf(stderr, "  OMP:  %d threads\n\n", N_CPU_THS);
	fflush(stderr);

	if (N_superph_made == 0)
		terminate("0 photons were generated. Aborting...", 1);
	fprintf(stderr, "Staring photon tracking...\n\n");
	fflush(stderr);

	// Track with GPU
	core_photon_tracking(phs, N_superph_made);

	// Track with CPU
	// #pragma omp parallel for schedule(guided)
	// for (unsigned long long i = 0; i < N_superph_made; ++i)
	// 	track_super_photon(&phs[i]);
	// for (unsigned long long i = 0; i < N_superph_made; ++i)
	// 	if (phs[i].tracking_status == TRACKING_STATUS_COMPLETE &&
	// 		record_criterion(&phs[i])) record_super_photon(&phs[i]);

	currtime = time(NULL);
	fprintf(stderr, "Final time %g, rate %g ph/s\n",
	(double) (currtime - starttime),
	(double) N_superph_made / (currtime - starttime));

	report_spectrum(N_superph_made);

	free(phs);

	return 0;
}
