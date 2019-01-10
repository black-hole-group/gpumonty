#!/usr/bin/python3

import subprocess
import sys
import signal
import os
import time
import ntpath

if sys.version_info[1] < 7:
    print("Error: This script was made for python >= 3.7, and you're trying " +
          "to run with python" + str(sys.version_info[0]) + "." +
          str(sys.version_info[1]))
    sys.exit(1)

# Defines the path of this file
SCRIPT_BASEDIR = os.path.dirname(os.path.realpath(__file__)) + "/"

################################################################################
#                                Tester Settings
#                           (Change them as you will)
################################################################################

DUMPS = [SCRIPT_BASEDIR + "../data/dump1000"] # Abs. paths for the dump files
SIZES = [(5000, 5)] # No of photons, executation rounds for each No of photons
M_UNIT = 4.e19
NUM_THREADS = None # Use None for default
NUM_BLOCKS = None # Use None for default
BLOCK_SIZE = None # Use None for default
EXEC_TIMEOUT = 300 # timeout in seconds
WARMUP_ROUNDS = 2
################################################################################
#                                Path settings
# (you can change then, but be carefull! Some rm commands are runned upon then!)
################################################################################
PERSIST_BUILD_DIR = False # IMPORTANT! Set True if you wish to persist the
# build dir. If you are building in the same dir of your project files,
# you MUST set this to True, otherwise, your project's directory WILL BE LOST)
ALWAYS_MAKE = True # When set to True, will always make the code (make's -B)
BUILD_DIR = ".time_build" # Name for the temp dir time.py creates to
# build and run grmonty
BUILD_PATH = SCRIPT_BASEDIR + BUILD_DIR # Complete path for the dir
# described above
MAKE_PATH = SCRIPT_BASEDIR + "../" # Path were the makefile is
EXEC_PATH = BUILD_PATH + "/bin/" # Path where, after build, the executable will
# be found
EXEC_NAME = "grmonty" # Name of the executable
################################################################################



################################################################################
######## Print functions

# Terminal colors
CEND = '\33[0m'
CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'
CBLUE = '\33[34m'

def say(msg, end="\n"):
	print(CBLUE + msg + CEND, end=end)
	sys.stdout.flush()

def success(msg, end="\n"):
	print(CGREEN + msg + CEND, end=end)
	sys.stdout.flush()

def warn(msg, end="\n"):
	print(CYELLOW + msg + CEND, end=end)
	sys.stdout.flush()

def complain(msg, end="\n"):
	print(CRED + msg + CEND, end=end)
	sys.stdout.flush()


################################################################################
######## Exit functions

def abort(msg, code=1):
	complain(msg)
	sys.exit(code)

def finish(code=0):
    if not PERSIST_BUILD_DIR: remove_tests_build_dir()
    sys.exit(code)

def set_sigint_handler():
    def sigint_handler(sig, frame):
        finish(1)
    signal.signal(signal.SIGINT, sigint_handler)
    if not PERSIST_BUILD_DIR: remove_tests_build_dir()

################################################################################
######## Janitors: running processes, create and remove dirs

def run(command, timeout=None, cwd=None):
    try:
        completed = subprocess.run(command, encoding="utf-8",
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   timeout=timeout, cwd=cwd)
        completed.check_returncode()
        return completed
    except FileNotFoundError:
        abort("Couldn't execute '" + command[0] + "'. File not found.")
    except subprocess.TimeoutExpired:
        abort("Timeout executing '" + command[0] + "' (" + str(timeout) +
			  " seconds)")

def remove_tests_build_dir(force=True):
    try:
        if force:
            run(["rm", "-rf", BUILD_PATH])
        else:
            run(["rm", "-r", "--interactive=never", BUILD_PATH])
    except subprocess.CalledProcessError as exception:
        abort("Failed to remove tests build dir:\n" + exception.stderr,
              exception.returncode)

def create_tests_build_dir():
    try:
        run(["mkdir", "-p", BUILD_PATH])
        run(["mkdir", "-p", BUILD_PATH + "/bin"])
    except subprocess.CalledProcessError as exception:
        abort("Failed to create tests build dir:\n" + exception.stderr,
              exception.returncode)

################################################################################
######## Helper overall python functions

def last_line(string, n=1):
	return string.split("\n")[-n]

################################################################################
######## Timing functions

def extract_time(stderr):
	times_str = last_line(stderr, 2)
	return [float(t) for t in times_str.split(" ")]

def mean(l):
	return sum(l)/len(l)

################################################################################
######## grmonty making and running

def build():
    print("Making")
    try:
        create_tests_build_dir()
        os.environ["GRMONTY_BASEBUILD"] = BUILD_PATH
        if ALWAYS_MAKE:
            run(["make", "-j8", "-B", "-C", MAKE_PATH])
        else:
            run(["make", "-j8", "-C", MAKE_PATH])
    except subprocess.CalledProcessError as exception:
        abort("Make failed:\n" + exception.stderr, exception.returncode)

def warmup():
	pass

def run_and_get_times():
	if NUM_THREADS: os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
	if NUM_BLOCKS: os.environ["NUM_BLOCKS"] = str(NUM_BLOCKS)
	if BLOCK_SIZE: os.environ["BLOCK_SIZE"] = str(BLOCK_SIZE)
	for dump in DUMPS:
		for size, rounds in SIZES:
			dumpname = ntpath.basename(dump)
			elapsed_times = []
			cpu_times = []
			print("[dump]:" + dumpname + " [N]:" + str(size))
			say("    warmup(%d): " % WARMUP_ROUNDS, end="")
			for iteration in range(WARMUP_ROUNDS):
				try:
					say("*", end="")
					process = run(["/bin/time", "--format=%e %U %S",
								  "./" + EXEC_NAME, str(size), dump,
								  str(M_UNIT)], timeout=EXEC_TIMEOUT,
								  cwd=EXEC_PATH)
				except subprocess.CalledProcessError as exception:
					print()
					abort("Error executing grmonty, at warmup:\n" +
						  exception.stderr, exception.returncode)
			print()
			for iteration in range(rounds):
				try:
					process = run(["/bin/time", "--format=%e %U %S",
								  "./" + EXEC_NAME, str(size), dump,
								  str(M_UNIT)], timeout=EXEC_TIMEOUT,
								  cwd=EXEC_PATH)
					mtime = extract_time(process.stderr)
					say("    elapsed: %.2fs cpu: %.2fcs" % (mtime[0],
						mtime[1] + mtime[2]))
					elapsed_times.append(mtime[0])
					cpu_times.append(mtime[1] + mtime[2])
				except subprocess.CalledProcessError as exception:
					abort("Error executing grmonty:\n" + exception.stderr,
						  exception.returncode)
			success("    mean(%d): elapsed: %.2fs cpu: %.2fcs" % (rounds,
					mean(elapsed_times), mean(cpu_times)))


################################################################################
######## Main

def invalid_options():
    complain("Invalid options")
    print_help()
    sys.exit(1)

def print_help():
    print("Usage: ./time.py [-h|--help]")
    print("  help - displays this help message")

def parse_args():
    if len(sys.argv) > 2: invalid_options()
    if len(sys.argv) == 2:
        opt = sys.argv[1]
        if opt in ("--help", "-h"): print_help (); sys.exit(0)
        else: invalid_options()

def main():
    parse_args()
    set_sigint_handler()
    build()
    run_and_get_times()
    finish()

if __name__ == "__main__":
    main()
