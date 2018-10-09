#!/usr/bin/python3

import subprocess
import re
import sys
import signal
import os
import time
import ntpath
import math
import json
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

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
SIZES = [(50000, 5)] # No of photons, executation rounds for each No of photons
M_UNIT = 4.e19
NUM_THREADS = 8
ATT_DIFF_LIMIT = 1.0 # Absolute percentage threshold to validate output atts
SPECIAL_ATT_DIFF_LIMIT = { # Threshold for specific attributes
    "max_tau_scatt": 15.0, # max_tau_scatt varies a lot
    "N_superph_recorded": 3.0
}
SPEC_DIFF_LIMIT = 200 # Threshold for spectrum difference from reference
EXEC_TIMEOUT = 300 # seconds (timeout for each round execution)
ALWAYS_PRINT_TESTS_INFO = False # True = Print tests info even if they succeed
PRINT_INDIVIDUAL_TEST_INFO = True # Print info for every test round (True) or
# just the mean (False)

################################################################################
#                                Path settings
# (you can change then, but be carefull! Some rm commands are runned upon then!)
################################################################################

PERSIST_TESTS_BUILD_DIR = False # IMPORTANT! Set True if you wish to persist the
# test's build dir. If you are building in the same dir of your project files,
# you MUST set this to True, otherwise, your project's directory WILL BE LOST)
TESTS_BUILD_DIR = ".tests_build" # Name for the temp dir tester.py creates to
# build and run grmonty
BUILD_PATH = SCRIPT_BASEDIR + TESTS_BUILD_DIR # Complete path for the dir
# described above
MAKE_PATH = SCRIPT_BASEDIR + "../" # Path were the makefile is
EXEC_PATH = BUILD_PATH + "/bin/" # Path where, after build, the executable will
# be found
EXEC_NAME = "grmonty" # Name of the executable
SPEC_FILE = "grmonty.spec" # Name of grmonty's output spectrum file
EXTRACTOR_OUTPUT_FILE = SCRIPT_BASEDIR + "references.json" # File where
# extractor will store references
INPUT_REF_FILE = EXTRACTOR_OUTPUT_FILE # File where tester will load references
# from
################################################################################




# Class to define execution mode: whether tester.py will be testing the code or
# extracting a reference output from it
class Mode(Enum):
    Test = 1
    Extract = 2
    def __str__(self):
        if self == Mode.Test: return "Test"
        elif self == Mode.Extract: return "Extract"
        return "None"

# Terminal colors
CEND = '\33[0m'
CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'
CBLUE = '\33[34m'
CREDBG = '\33[41m'
CGREENBG = '\33[42m'
CYELLOWBG = '\33[43m'

# Some internal globals
start_time = None
exec_mode = Mode(Mode.Test) # Execution Mode. Don't change manually, just by CLI

# Reference code outputs (yet to be imported from INPUT_REF_FILE)
ref = None
ref_spec = None


################################################################################
######## Some helpers for overall tester execution

def run(command, timeout=None, cwd=None):
    try:
        completed = subprocess.run(command, encoding="utf-8",
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   timeout=timeout, cwd=cwd)
        completed.check_returncode()
        return completed
    except FileNotFoundError:
        tester_error("Couldn't execute '" + command[0] + "'. File not found.")
    except subprocess.TimeoutExpired:
        tester_error("Timeout executing '" + command[0] + "' (" +
                     str(timeout) + " seconds)")

def dict2str_float_formater(dictionary, float_format):
    string = "{"
    for k, v in dictionary.items():
        string += "'" +str(k) + "': " + float_format % v + ", "
    if len(string) > 1:
        string = string[:-2]
    string += "}"
    return string

def tail(arr, n):
    return arr[len(arr) - n:]

def colored_string(string, color):
    return color + string + CEND

def tester_print(msg, sep=False, color=None, prompt=False):
    string = "[TESTER] " if exec_mode == Mode.Test else "[EXTRACTOR] "
    if sep: string += "--------------------- "
    string += msg
    if color: string = colored_string(string, color)
    if prompt: return input(string)
    else: print(string)
    return None

def tester_error(msg, code=1):
    tester_print("Tester.py failed...\n" + msg, color=CRED)
    finish_tester(code)

def str_range(n, padding=None):
    if n < 0: return ""
    string = ""
    for i in range(n):
        string += str(i) + ", "
    string += "[" + str(n) + "]"
    if padding and padding > n + 1:
        for i in range(n + 1, padding):
            string += ", " + str(i)
    return string

def remove_tests_build_dir(force=True):
    try:
        if force:
            run(["rm", "-rf", BUILD_PATH])
        else:
            run(["rm", "-r", "--interactive=never", BUILD_PATH])
    except subprocess.CalledProcessError as exception:
        tester_error("Failed to remove tests build dir:\n" + exception.stderr,
                     exception.returncode)

def create_tests_build_dir():
    try:
        run(["mkdir", "-p", BUILD_PATH])
        run(["mkdir", "-p", BUILD_PATH + "/bin"])
    except subprocess.CalledProcessError as exception:
        tester_error("Failed to create tests build dir:\n" + exception.stderr,
                     exception.returncode)

def start_tester():
    global start_time
    start_time = time.time()
    def sigint_handler(sig, frame):
        finish_tester(1)
    signal.signal(signal.SIGINT, sigint_handler)
    if not PERSIST_TESTS_BUILD_DIR: remove_tests_build_dir()

def finish_tester(code=0):
    end_time = time.time()
    tester_print("Elapsed time: " + "%.3f" % (end_time - start_time) +
                 " seconds", sep=True)
    if not PERSIST_TESTS_BUILD_DIR: remove_tests_build_dir()
    sys.exit(code)


################################################################################
######## References preparation handling: Importing and checking

def convert_int_keys(d):
    '''
    Takes a dict d and coverts it's int keys that are stored as strings to
    true ints.
    '''
    conv_d = {}
    for key, value in d.items():
        conv_key = None
        try:
            conv_key = int(key)
        except ValueError:
            conv_key = key
        if isinstance(value, dict):
            value = convert_int_keys(value)
        conv_d[conv_key] = value
    return conv_d


def load_references():
    global ref, ref_spec
    try:
        ref_file = open(INPUT_REF_FILE, "r")
        ref_data = json.loads(ref_file.read())
        ref = convert_int_keys(ref_data["ref"])
        ref_spec = convert_int_keys(ref_data["ref_spec"])
        ref_file.close()
    except json.decoder.JSONDecodeError as e:
        tester_error("Could'nt decode references. It seems that the " +
                     "references' file is incorrect" +
                     ((":\n" + str(e)) if e and str(e) else "."))
    except FileNotFoundError:
        tester_error("Could't load references. File \"" + INPUT_REF_FILE +
                     "\" wasn't found.")
    except Exception as e:
        tester_error("Unexpected error when trying to load references" +
                     ((":\n" + str(e)) if e and str(e) else "."))

def check_references():
    try:
        if ref is None or ref_spec is None:
            tester_error("Reference variables are None. Maybe references file" +
                         " wasn't imported?")
    except NameError:
        tester_error("Reference variables were not loaded. Maybe references " +
                     "file wasn't imported?")
    for dump in DUMPS:
        dump = ntpath.basename(dump)
        for size, _ in SIZES:
            if dump not in ref or size not in ref[dump]:
                tester_error("Reference file has no output reference for " +
                             dump + " size " + str(size))
            elif dump not in ref_spec or size not in ref_spec[dump]:
                tester_error("Reference file has no spectrum reference for " +
                             dump + " size " + str(size))

def check_extraction_settings():
    dumpnames = {}
    for dump in DUMPS:
        dname = ntpath.basename(dump)
        if dname in dumpnames:
            tester_error("Your DUMPS list contain dumps with the same name," +
                         " e.g: \"" + dname + "\"")
        dumpnames[dname] = True

def check_ref_file_overwrite():
    if os.path.exists(INPUT_REF_FILE):
        ## From https://gist.github.com/hrouault/1358474
        while True:
            choice = tester_print("WARNING: File \"" + INPUT_REF_FILE +
                                  "\" already exists and\nwill be overwritten" +
                                  ". Continue? [y/N] ",
                                  color=CYELLOW, prompt=True).lower()
            if choice in ('', "n"):
                tester_print("Ok, aborting...")
                finish_tester()
            elif choice == "y":
                print()
                return
            print()

################################################################################
######## Functions to handle grmonty's outputs

def extract_infos(stderr):
    lines = tail(stderr.split("\n"), 5)
    tuples = {}
    for line in lines:
        line = [re.sub("(^ +| +$)", "", w) for w in line.replace(":", "").split(",")]
        for key_value in line:
            if key_value:
                attribute, value = re.split(" +", key_value)
                tuples[attribute] = float(value)
    return tuples

def calc_diffs_from_ref(infos, dump, size):
    diffs = {}
    for att, val in infos.items():
        if val == 0.0:
            if ref[dump][size][att] == 0.0: diffs[att] = 0.0
            elif ref[dump][size][att] > 0: diffs[att] = float("inf")
            else: diffs[att] = -float("inf")
        else: diffs[att] = ((val - ref[dump][size][att]) * 100) / val
    return diffs

def mean_arr_of_dicts(arr):
    m = {}
    if len(arr) == 0: return m
    try:
        for att, _ in arr[0].items():
            if att not in m: m[att] = 0.0
            for ele in arr:
                m[att] += ele[att]
            m[att] /= len(arr)
    except KeyError as e:
        tester_error("Attribute " + str(e) + " couldn't be captured in some " +
                     "execution.")
    return m


################################################################################
######## Functions to handle spectrums

def mean_spec(specs):
    mean_y = []
    x_axis = extract_total_x_axis(specs)
    for x in x_axis:
        y = 0.0
        for spec in specs:
            acc = get_y_in_spec(spec, x)
            y += acc
        y /= len(specs)
        mean_y.append(y)
    return (x_axis, mean_y)

def extract_total_x_axis(specs):
    x_axis = []
    for x, __ in specs:
        x_axis.extend(x)
    x_axis = list(set(x_axis))
    x_axis.sort()
    return x_axis

def get_y_in_spec(spec, x):
    x_axis, y_axis = spec
    prev_i = -1
    for i in range(len(x_axis)):
        if x_axis[i] == x: return y_axis[i]
        elif x_axis[i] > x:
            if prev_i == -1: return y_axis[i]

            elif y_axis[prev_i] <= y_axis[i]:
                return ((x - x_axis[prev_i]) * (y_axis[i] - y_axis[prev_i]) /
                        (x_axis[i] - x_axis[prev_i])) + y_axis[prev_i]

            else: return ((x - x_axis[i]) * (y_axis[prev_i] - y_axis[i]) /
                          (x_axis[prev_i] - x_axis[i])) + y_axis[i]
        prev_i = i
    return y_axis[prev_i]

def load_spectrum():
    me = 9.1e-28
    c = 3.e10
    h = 6.626e-27
    Lsol = 3.83e33
    nbins = 6
    #
    try: data = np.loadtxt(EXEC_PATH + SPEC_FILE)
    except Exception as e: tester_error ("Failed to load spectrum from '" +
                                         SPEC_FILE + "': " +
                                         (str(e) if e and str(e) else ""))
    tdata = np.transpose(data)
    lw = tdata[0,:]	# log of photon energy in electron rest-mass units
    i = np.arange(0,nbins,1)
    nLn = np.log10(tdata[1+i*7,:] + 1.e-30)
    lw = lw + np.log10(me*c*c/h)  # convert to Hz from electron rest-mass energy
    nLn = nLn + np.log10(Lsol)  # convert to erg/s from Lsol
    nLn_mean = np.mean(nLn, axis = 0)
    # return[(lw[j], nLn_mean[j]) for j in range(len(lw))]
    return (lw, nLn_mean)

def compare_to_reference_spectrum(spect, dump, size):
    x_axis = extract_total_x_axis([spect, ref_spec[dump][size]])
    diff = 0.0
    for x in x_axis:
        diff += (get_y_in_spec(spect, x) -
                 get_y_in_spec(ref_spec[dump][size], x)) ** 2
    return diff

def plot_spec_diff(test_spect, dump, size, plot_filename):
    x_axis = extract_total_x_axis([test_spect, ref_spec[dump][size]])
    y_axis = []
    for x in x_axis:
        y1 = get_y_in_spec(ref_spec[dump][size], x)
        y2 = get_y_in_spec(test_spect, x)
        y_axis.append(((y2 - y1) * 100) / y2)
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, y_axis, label="Percentual difference of testing over " +
             "reference")
    plt.title("Spectrum Difference Over Reference: " + dump + " size=" +
              str(size))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(test_spect[0], test_spect[1], "b", label="testing spectrum")
    plt.plot(ref_spec[dump][size][0], ref_spec[dump][size][1], "g",
             label="reference spectrum")
    plt.title("Reference and Testing Spectrums")
    plt.legend()
    plt.figtext(0.75, 0.33, "Sum of Squared Error: ")
    plt.figtext(0.75, 0.31, "%.2f " % compare_to_reference_spectrum(test_spect,
                                                                    dump, size))
    plt.savefig(plot_filename, format="png", dpi=150)


################################################################################
######## Testing functions

def test_failed(test_title, msg, code=1, terminate=True):
    tester_print("TEST FAILED: " + test_title + "\n" +msg + "\n", color=CRED)
    if terminate: finish_tester(code)

def test_succeeded(terminate=False):
    tester_print("Test succeeded", color=CGREEN)
    if terminate: finish_tester()

def extract_succeeded(terminate=False):
    tester_print("Extraction succeeded", color=CGREEN)
    if terminate: finish_tester()

def build():
    tester_print("Making", sep=True)
    try:
        create_tests_build_dir()
        os.environ["GRMONTY_BASEBUILD"] = BUILD_PATH
        print(run(["make", "-B", "-C", MAKE_PATH]).stdout)
    except subprocess.CalledProcessError as exception:
        tester_error("Make failed:\n" + exception.stderr, exception.returncode)

def mk_infos_failed_msg (att, diff, limit, reference, infos_mean, diffs, infos):
    msg = ("Difference for " + att + " is " + "%.3f" % diff + "% (greater, in" +
           " absolute value, than limit=" + str(limit) + "%)\n" +
           "Reference: " + str(reference) + "\nGot:       " + str(infos_mean) +
           "\nDiffs:     " + dict2str_float_formater(diffs, "%.3f%%"))

    if PRINT_INDIVIDUAL_TEST_INFO:
        msg += "\n\nInfos from each execution:\n"
        for info in infos:
            msg += str(info) + "\n"
    return msg

def validate_infos_outputs(dump, size, infos):
    infos_mean = mean_arr_of_dicts(infos)
    diffs = calc_diffs_from_ref(infos_mean, dump, size)
    for att, diff in diffs.items():
        limit = SPECIAL_ATT_DIFF_LIMIT.get(att, ATT_DIFF_LIMIT)
        if math.isnan(diff) or math.isinf(diff) or abs(diff) > abs(limit):
            test_failed("Output infos test",
                        mk_infos_failed_msg(att, diff, limit, ref[dump][size],
                                            infos_mean, diffs, infos),
                        terminate=False)
            return False
    if ALWAYS_PRINT_TESTS_INFO:
        info_str = ("Output infos test\nReference: " + str(ref[dump][size]) +
                    "\nGot:       " + str(infos_mean) + "\nDiffs:     " +
                    dict2str_float_formater(diffs, "%.3f%%"))
        if PRINT_INDIVIDUAL_TEST_INFO:
            info_str += "\n\nInfos from each execution:\n"
            for info in infos:
                info_str += str(info) + "\n"
        tester_print(info_str)
    return True

def validate_spectrum_output(dump, size, spects):
    test_spec = mean_spec(spects)
    spec_diff = compare_to_reference_spectrum(test_spec, dump, size)
    plot_filename = (SCRIPT_BASEDIR + "diff_spect_" + dump + "_" + str(size) +
                     ".png")

    if math.isnan(spec_diff) or math.isinf(spec_diff) or spec_diff > SPEC_DIFF_LIMIT:

        plot_spec_diff(test_spec, dump, size, plot_filename)
        test_failed("Spectrum test", "Difference in spectrum is " +
                    "%.3f" % spec_diff + " (greater, in absolute value, than " +
                    "limit=" + str(SPEC_DIFF_LIMIT) + ")\nSaved testing " +
                    "spectrum difference over reference spectrum in '" +
                    plot_filename + "'", terminate=False)
        return False
    if ALWAYS_PRINT_TESTS_INFO:
        plot_spec_diff(test_spec, dump, size, plot_filename)
        tester_print("Spectrum test\nDifference in spectrum is " +
                     "%.3f" % spec_diff + "\nSaved testing spectrum " +
                     "difference over reference spectrum in '" + plot_filename +
                     "'")
    return True

def validate_test_outputs(dump, size, infos, spects):
    infos_validation = validate_infos_outputs(dump, size, infos)
    spec_validation = validate_spectrum_output(dump, size, spects)
    if not spec_validation or not infos_validation: finish_tester(1)

def run_tests():
    e_ref, e_ref_spec = {}, {} # Used if extracting info
    os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    for dump in DUMPS:
        for size, rounds in SIZES:
            dumpname = ntpath.basename(dump)
            if exec_mode == Mode.Extract:
                e_ref[dumpname] = {}
                e_ref_spec[dumpname] = {}
            tester_print("Running " + dumpname + " N=" + str(size), sep=True)
            infos = []
            spects = []
            for iteration in range(rounds):
                tester_print("Rounds: " + str(iteration+1) + "/" + str(rounds),
                             sep=True)
                try:
                    process = run(["./" + EXEC_NAME, str(size), dump,
                                   str(M_UNIT)], timeout=EXEC_TIMEOUT,
                                  cwd=EXEC_PATH)
                    if process.stdout:
                        print(colored_string(process.stdout, CBLUE))
                    infos.append(extract_infos(process.stderr))
                    spects.append(load_spectrum())
                except subprocess.CalledProcessError as exception:
                    tester_error("Error executing grmonty:\n" +
                                 exception.stderr, exception.returncode)
            if exec_mode == Mode.Extract:
                e_ref[dumpname][size] = mean_arr_of_dicts(infos)
                e_ref_spec[dumpname][size] = mean_spec(spects)
            else:
                validate_test_outputs(ntpath.basename(dump), size, infos,
                                      spects)
    if exec_mode == Mode.Extract:
        try:
            out = open(EXTRACTOR_OUTPUT_FILE, "w+")
            json.dump({"ref": e_ref, "ref_spec": e_ref_spec}, fp=out,
                      indent="  ")
            out.close()
            extract_succeeded()
        except Exception as e:
            tester_error("Error saving references file" +
                         ((":\n" + str(e)) if e and str(e) else "."))
    else:
        test_succeeded()



################################################################################
######## Main

def invalid_options():
    print(CRED + "Invalid options" + CEND)
    print_help()
    sys.exit(1)

def print_help():
    print("Usage: ./tester.py [-e|--extract] [-t|--test] [-h|--help]")
    print("Example: ./tester --extract")
    print("No options run --test by default.")
    print()
    print("  extract - extracts outputs from current grmonty code to use as" +
          " reference for tests")
    print("  test - tests current grmonty code against reference")
    print("  help - displays this help message")

def parse_args():
    global exec_mode
    if len(sys.argv) > 2: invalid_options()
    if len(sys.argv) == 2:
        opt = sys.argv[1]
        if opt in ("--help", "-h"): print_help (); sys.exit(1)
        elif opt in ("--extract", "-e"): exec_mode = Mode(Mode.Extract)
        elif opt in ("--test", "-t"): exec_mode = Mode(Mode.Test)
        else: invalid_options()
    tester_print("MODE: " + str(exec_mode), sep=True)

def main():
    parse_args()
    start_tester()
    if exec_mode == Mode.Test:
        load_references()
        check_references()
    else:
        check_ref_file_overwrite()
        check_extraction_settings()
    build()
    run_tests()
    finish_tester()

if __name__ == "__main__":
    main()
