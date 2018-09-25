#!/usr/bin/python3

import subprocess
import re
import sys
import signal
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import ntpath
import math
from enum import Enum

if sys.version_info[1] < 7:
    print("Error: This script was made for python >= 3.7, and you're trying to run with" +
              " python" + str(sys.version_info[0]) + "." + str(sys.version_info[1]))
    sys.exit(1)

# Defines the path of this file
script_basedir = os.path.dirname(os.path.realpath(__file__)) + "/"

#################################################################
#                    Tester Settings
#                  (Change them as you will)
#################################################################
dumps = [script_basedir + "../data/dump1000"] # Complete paths for the dump files
sizes = [(50000, 5)] # No of photons and executation rounds for each No of photons
M_unit = 4.e19
num_threads = 8
att_diff_limit = 1.0 # Percentage threshold to validate output atributes (jn absolute value)
special_att_diff_limit = {"max_tau_scatt": 8.0, "N_superph_recorded": 2.0} # Threshold for specific attributes
spec_diff_limit = 200 # Threshold for spectrum difference from reference
exec_timeout = 300 # seconds (timeout for each round execution)
always_print_tests_info = False # True = Print tests info even if they succeed
print_individual_test_info = True # Print info for every test round (True) or just the mean (False)
##################################################################
#           Path settings
#     (you can change then, but be carefull! Some rm commands are runned upon then! )
##################################################################
tests_build_dir = ".tests_build" # Name for the temp dir tester.py creates to build and run grmonty
build_path = script_basedir + tests_build_dir # Complete path for the dir described above
make_path = script_basedir + "../" # Path were the makefile is
exec_path = build_path + "/bin/" # Path where, after build, the executable will be found
exec_name = "grmonty" # Name of the executable
spec_file = "spectrum.dat" # Name of grmonty's output spectrum file
extractor_output_file = script_basedir + "references.py" # Name for the file where extractor will store references
##################################################################




# Class to define execution mode: whether tester.py will be testing the code or extracting a
# reference output from it
class Mode(Enum):
    Test = 1
    Extract = 2
    def __str__(self):
        if self == Mode.Test: return "Test"
        elif self == Mode.Extract: return "Extract"
        return "None"

# Terminal colors
CEND      = '\33[0m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'

# Some internal globals
start_time = None
exec_mode = Mode(Mode.Test) # Execution Mode. Do not change this manually. It will be changed by comand line

# Reference code outputs (yet to be imported from references.py)
ref = None
ref_spec = None


#################################################################
######## Some helpers for overall tester execution

def run (command, timeout=None, cwd=None):
    try:
        completed = subprocess.run(command, encoding="utf-8", stderr=subprocess.PIPE,
                                                            stdout=subprocess.PIPE, timeout=timeout, cwd=cwd)
        completed.check_returncode()
        return completed
    except FileNotFoundError:
        tester_error("Couldn't execute '" + command[0] + "'. File not found.")
    except subprocess.TimeoutExpired:
        tester_error("Timeout executing '" + command[0] + "' (" + str(exec_timeout) + " seconds)")

def dict2str_float_formater(dict, format):
    string = "{"
    for k, v in dict.items():
        string += "'" +str(k) + "': " + format % v + ", "
    if (len(string) > 1):
        string = string[:-2]
    string += "}"
    return string

def tail (arr, n):
    return arr[len(arr) - n:]

def tester_print(msg, sep=False, color=None, prompt=False):
    str = "[TESTER] " if exec_mode == Mode.Test else "[EXTRACTOR] "
    if sep: str += "--------------------- "
    str += msg
    if color: str = color + str + CEND
    if prompt: return input(str)
    else: print(str)
    return None

def tester_error(msg, code=1):
    tester_print("Tester.py failed...\n" + msg, color=CRED)
    finish_tester (code)

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
            run(["rm", "-rf", build_path])
        else:
            run(["rm", "-r", "--interactive=never", build_path])
    except subprocess.CalledProcessError as exception:
        tester_error("Failed to remove tests build dir:\n" + exception.stderr, exception.returncode)

def create_tests_build_dir():
    try:
        run(["mkdir", build_path])
        run(["mkdir", build_path + "/bin"])
    except subprocess.CalledProcessError as exception:
        tester_error("Failed to create tests build dir:\n" + exception.stderr, exception.returncode)

def start_tester():
    global start_time
    start_time = time.time()
    def sigint_handler(sig, frame):
            finish_tester(1)
    signal.signal(signal.SIGINT, sigint_handler)
    remove_tests_build_dir()

def finish_tester(code=0):
    end_time = time.time()
    tester_print("Elapsed time: " + "%.3f" % (end_time - start_time) + " seconds", sep=True)
    remove_tests_build_dir()
    sys.exit(code)


#################################################################
######## References preparation handling: Importing and checking

def import_references():
    global ref, ref_spec
    ''' Import Reference code outputs (should be in the folowing structure)
    ref = {
        "dumpXXX": {
            size1: {'luminosity': 939.741, 'dMact': -1.40148e-09, 'efficiency': -0.0452698,
                'L/Ladv': 0.651418, 'max_tau_scatt': 0.000476014, 'N_superph_made': 105433.0,
                'N_superph_recorded': 65778.0},
            size2: {'luminosity': 943.2335999999999, 'dMact': -1.40148e-09, 'efficiency': -0.04543808,
                'L/Ladv': 0.6538394, 'max_tau_scatt': 0.0003997848, 'N_superph_made': 1039350.4,
                'N_superph_recorded': 593041.6},
            ...
        },
        "dumpYYY": { ... }
    }

    ref_spec = {
        "dumpXXX": {
            size1: ([x1, x2, ...], [y1, y2, ...]),
            ...
        },
        "dumpYYY": { ... }
    }
    '''
    try:
        from references import ref, ref_spec
    except (ImportError, ModuleNotFoundError):
        tester_error("Could't import references. Do you have a valid references.py file in tester.py's dir?")

def check_reference():
    try:
        if ref is None or ref_spec is None:
            tester_error("Reference variables are None. Maybe references.py wasn't imported?")
    except NameError:
        tester_error("Reference variables were not loaded. Maybe references.py wasn't imported?")
    for dump in dumps:
        dump = ntpath.basename(dump)
        for size, _ in sizes:
            if dump not in ref or size not in ref[dump]:
                tester_error("No information reference for " + dump + " size " + str(size))
            elif dump not in ref_spec or size not in ref_spec[dump]:
                tester_error("No spectrum reference for " + dump + " size " + str(size))

def check_extraction_settings():
    dumpnames = {}
    for dump in dumps:
        dname = ntpath.basename(dump)
        if dname in dumpnames:
            tester_error ("Your dumps list contain dumps with the same name, e.g: \"" + dname + "\"")
        dumpnames[dname] = True

def display_ref_overwrite_warning():
    ## From https://gist.github.com/hrouault/1358474
    while True:
        choice = tester_print("WARNING: You already have a references.py at tester.py's dir.\n" +
            "This file will be overwritten. Continue? [y/N] ", color=CYELLOW, prompt=True).lower()
        if choice == '' or choice == "n":
            tester_print("Ok, aborting...")
            finish_tester()
        elif choice == "y":
            print()
            return
        print()

#################################################################
######## Functions to handle grmonty's outputs

def extract_infos(stderr):
    lines = tail(stderr.split("\n"), 5)
    tuples = {}
    for line in lines:
        line = [re.sub("(^ +| +$)", "", word) for word in line.replace(":", "").split(",")]
        for tuple in line:
            if tuple:
                attribute, value = re.split(" +", tuple)
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
        tester_error("Attribute " + str(e) + " couldn't be captured in some execution.")
    return m


#################################################################
######## Functions to handle spectrums

def mean_spec(specs):
    mean_y = []
    x_axis = extract_total_x_axis (specs)
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
    for x,__ in specs:
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
                return ((x - x_axis[prev_i]) * (y_axis[i] - y_axis[prev_i]) / (x_axis[i] - x_axis[prev_i])) + y_axis[prev_i]
            else: return ((x - x_axis[i]) * (y_axis[prev_i] - y_axis[i]) / (x_axis[prev_i] - x_axis[i])) + y_axis[i]
        prev_i = i
    return y_axis[prev_i]

def load_spectrum():
    me = 9.1e-28
    c = 3.e10
    h = 6.626e-27
    Lsol = 3.83e33
    nbins = 6
    #
    try: data = np.loadtxt(exec_path + spec_file)
    except Exception as e: tester_error ("Failed to load spectrum from '" + spec_file + "': " + (str(e) if e and str(e) else ""))
    tdata = np.transpose(data)
    lw = tdata[0,:]	# log of photon energy in electron rest-mass units
    i = np.arange(0,nbins,1)
    nLn = np.log10(tdata[1+i*7,:] + 1.e-30)
    lw = lw + np.log10(me*c*c/h)   # convert to Hz from electron rest-mass energy
    nLn = nLn + np.log10(Lsol)  # convert to erg/s from Lsol
    nLn_mean = np.mean(nLn, axis = 0)
    # return[(lw[j], nLn_mean[j]) for j in range(len(lw))]
    return (lw, nLn_mean)

def compare_to_reference_spectrum(spect, dump, size):
    x_axis = extract_total_x_axis([spect, ref_spec[dump][size]])
    diff = 0.0
    for x in x_axis:
        diff += (get_y_in_spec(spect, x) - get_y_in_spec(ref_spec[dump][size], x)) ** 2
    return diff

def plot_spec_diff(test_spect, dump, size, plot_filename):
    x_axis = extract_total_x_axis([test_spect, ref_spec[dump][size]])
    y_axis = []
    for x in x_axis:
        y1 = get_y_in_spec(ref_spec[dump][size], x)
        y2 = get_y_in_spec(test_spect, x)
        y_axis.append(((y2 - y1) * 100) / y2)
    plt.figure(figsize=(12,12))
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, y_axis, label="Percentual difference of testing over reference")
    plt.title("Spectrum Difference Over Reference: " + dump + " size=" + str(size))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(test_spect[0], test_spect[1], "b", label="testing spectrum")
    plt.plot(ref_spec[dump][size][0], ref_spec[dump][size][1], "g", label="reference spectrum")
    plt.title("Reference and Testing Spectrums")
    plt.legend()
    plt.figtext(0.75, 0.33, "Sum of Squared Error: ")
    plt.figtext(0.75, 0.31, "%.2f " % compare_to_reference_spectrum(test_spect, dump, size))
    plt.savefig(plot_filename, format="png", dpi=150)


#################################################################
######## Testing functions

def test_failed(test_title, msg, code=1, terminate=True):
    tester_print("TEST FAILED: " + test_title + "\n" +msg + "\n", color=CRED)
    if terminate: finish_tester (code)

def test_succeeded(terminate=False):
    tester_print("Test succeeded", color=CGREEN)
    if terminate: finish_tester()

def extract_succeeded(terminate=False):
    tester_print("Extraction succeeded", color=CGREEN)
    if terminate: finish_tester()

def build ():
    tester_print("Making", sep=True)
    try:
        create_tests_build_dir()
        os.environ["GRMONTY_BASEBUILD"] = build_path
        print(run(["make", "-C", make_path]).stdout)
    except subprocess.CalledProcessError as exception:
        tester_error("Make failed:\n" + exception.stderr, exception.returncode)

def mk_infos_failed_msg (att, diff, limit, reference, infos_mean, diffs, infos):
    msg = ("Difference for " + att + " is " + "%.3f" % diff + "% (greater, in absolute value, than limit=" + str(limit) +
    "%)\n" + "Reference: " + str(reference) + "\nGot:       " + str(infos_mean) + "\nDiffs:     " +
    dict2str_float_formater(diffs, "%.3f%%"))
    if print_individual_test_info:
        msg += "\n\nInfos from each execution:\n"
        for info in infos:
            msg += str(info) + "\n"
    return msg

def validate_infos_outputs(dump, size, infos):
    infos_mean = mean_arr_of_dicts (infos)
    diffs = calc_diffs_from_ref(infos_mean, dump, size)
    for att, diff in diffs.items():
        limit = special_att_diff_limit[att] if att in special_att_diff_limit else att_diff_limit
        if math.isnan(diff) or math.isinf(diff) or abs(diff) > abs(limit):
            test_failed("Output infos test", mk_infos_failed_msg (att, diff, limit, ref[dump][size],
                                infos_mean, diffs, infos), terminate=False)
            return False
    if always_print_tests_info:
        info_str = ("Output infos test\nReference: " + str(ref[dump][size]) +
        "\nGot:       " + str(infos_mean) + "\nDiffs:     " + dict2str_float_formater(diffs, "%.3f%%"))
        if print_individual_test_info:
            info_str += "\n\nInfos from each execution:\n"
            for info in infos:
                info_str += str(info) + "\n"
        tester_print(info_str)
    return True

def validate_spectrum_output(dump, size, spects):
    test_spec = mean_spec(spects)
    spec_diff = compare_to_reference_spectrum(test_spec, dump, size)
    plot_filename = script_basedir + "diff_spect_" + dump + "_" + str(size) + ".png"
    if math.isnan(spec_diff) or math.isinf(spec_diff) or spec_diff > spec_diff_limit:
        plot_spec_diff(test_spec, dump, size, plot_filename)
        test_failed("Spectrum test", "Difference in spectrum is " + "%.3f" % spec_diff + " (greater, in absolute value, than limit="
                            + str(spec_diff_limit) + ")\nSaved testing spectrum difference over reference spectrum in '" +
                            plot_filename + "'", terminate=False)
        return False
    if always_print_tests_info:
        plot_spec_diff(test_spec, dump, size, plot_filename)
        tester_print("Spectrum test\nDifference in spectrum is " + "%.3f" % spec_diff +
        "\nSaved testing spectrum difference over reference spectrum in '" + plot_filename + "'")
    return True

def validate_test_outputs(dump, size, infos, spects):
    infos_validation = validate_infos_outputs(dump, size, infos)
    spec_validation = validate_spectrum_output(dump, size, spects)
    if not spec_validation or not infos_validation: finish_tester(1)

def run_tests():
    e_ref, e_ref_spec = {}, {} # Used if extracting info
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    for dump in dumps:
        for size, rounds in sizes:
            dumpname = ntpath.basename(dump)
            if exec_mode == Mode.Extract:
                e_ref[dumpname] = {}
                e_ref_spec[dumpname] = {}
            tester_print("Running " + dumpname + " N=" + str(size), sep=True)
            infos = []
            spects = []
            for round in range(rounds):
                tester_print("Rounds: " + str(round+1) + "/" + str(rounds), sep=True)
                try:
                    process = run(["./" + exec_name, str(size), dump, str(M_unit)], timeout=exec_timeout, cwd=exec_path)
                    if process.stdout: print(stdout)
                    infos.append(extract_infos(process.stderr))
                    spects.append (load_spectrum())
                except subprocess.CalledProcessError as exception:
                    tester_error("Error executing grmonty:\n" + exception.stderr, exception.returncode)
            if exec_mode == Mode.Extract:
                e_ref[dumpname][size] = mean_arr_of_dicts(infos)
                e_ref_spec[dumpname][size] = mean_spec(spects)
            else:
                validate_test_outputs(ntpath.basename(dump), size, infos, spects)
    if exec_mode == Mode.Extract:
        out = open(extractor_output_file, "w+")
        out.write("ref=")
        out.write(str(e_ref))
        out.write("\n")
        out.write("ref_spec=")
        out.write(str(e_ref_spec))
        out.close()
        extract_succeeded()
    else:
        test_succeeded()



#################################################################
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
    print("  extract - extracts outputs from current grmonty code to use as reference for tests")
    print("  test - tests current grmonty code against reference")
    print("  help - displays this help message")

def parse_args():
    global exec_mode
    if len(sys.argv) > 2: invalid_options()
    if len(sys.argv) == 2:
        opt = sys.argv[1]
        if opt == "--help" or opt == "-h": print_help (); sys.exit(1)
        elif opt == "--extract" or opt == "-e": exec_mode = Mode(Mode.Extract)
        elif opt ==  "--test" or opt == "-t": exec_mode = Mode(Mode.Test)
        else: invalid_options ()
    tester_print("MODE: " + str(exec_mode), sep=True)

def main():
    parse_args()
    start_tester()
    if exec_mode == Mode.Test:
        import_references()
        check_reference()
    else:
        if os.path.exists(extractor_output_file):
            display_ref_overwrite_warning()
        check_extraction_settings()
    build()
    run_tests()
    finish_tester ()

if __name__ == "__main__":
    main()
