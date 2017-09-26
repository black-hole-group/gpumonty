

def mpimonty():
	"""
	Pseudocode for grmonty supporting MPI.
	"""

	# initialize random number generator in parallel
	init_monty_rand(139 * myid + time(NULL));	/* Arbitrarily picked initial seed */

	# reads data from GRMHD simulation
	init_model(argv);

	# initializes global number of photons produced, scattered and recorded
	# in the SED, will be updated after all slaves are done 
	N_superph_made = 0;
	N_superph_recorded = 0;
	N_scatt = 0;
	quit_flag = 0;

	# parallel while loop that generates light rays
	while (1):
		if (!quit_flag):
			# generate photon
			make_super_photon(&ph, &quit_flag);
		
		# if critical number of photons were produced, finish
		if (quit_flag):
			break;

		# solve geodesics
		track_super_photon(&ph);

		N_superph_made += 1;

	# Once done with all photons, assemble SED
	omp_reduce_spect();
	report_spectrum((int) N_superph_made);

