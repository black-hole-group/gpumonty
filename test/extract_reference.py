#!/usr/bin/python3

import subprocess
import re
import sys
import os
import time
import numpy as np
import ntpath




#################################################################
#                    Extractor Settings
#                  (Change them as you will)
#################################################################
dumps = ["./data/dump1000", "./data/dump900"]
sizes = [(50000, 5)]
M_unit = 4.e19
num_threads = 8
exec_path = "./bin/"
make_path = "./"
exec_name = "grmonty"
spec_file = "spectrum.dat"
exec_timeout = 300 # seconds (timeout for each round execution)
output_file = "./test/references.py"
##################################################################





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


#################################################################
######## Some python helpers for overall extraction execution

def run (command, timeout=None):
    try:
        completed = subprocess.run(command, encoding="utf-8", stderr=subprocess.PIPE,
                                                            stdout=subprocess.PIPE, timeout=timeout)
        completed.check_returncode()
        return completed
    except FileNotFoundError:
        tester_error("Couldn't execute '" + command[0] + "'. File not found.")
    except subprocess.TimeoutExpired:
        tester_error("Timeout executing '" + command[0] + "' (" + str(exec_timeout) + " seconds)")

def tail (arr, n):
    return arr[len(arr) - n:]

def extractor_error(msg, code=1):
    print(CRED + "[EXTRACTOR] Extractor failed..." + CEND)
    print(CRED + msg + CEND)
    finish_extractor (code)

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

def finish_extractor(code=0):
    end_time = time.time()
    print("[EXTRACTOR]--------------- Extractor elapsed time: " + "%.3f" % (end_time - start_time)
            + " seconds")
    sys.exit(code)


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
        extractor_error("Attribute " + str(e) + " couldn't be captured in some execution.")
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
    data = np.loadtxt(spec_file)
    tdata = np.transpose(data)
    lw = tdata[0,:]	# log of photon energy in electron rest-mass units
    i = np.arange(0,nbins,1)
    nLn = np.log10(tdata[1+i*7,:] + 1.e-30)
    lw = lw + np.log10(me*c*c/h)   # convert to Hz from electron rest-mass energy
    nLn = nLn + np.log10(Lsol)  # convert to erg/s from Lsol
    nLn_mean = np.mean(nLn, axis = 0)
    # return[(lw[j], nLn_mean[j]) for j in range(len(lw))]
    return (lw, nLn_mean)


#################################################################
######## Testing functions


def make ():
    print("[EXTRACTOR]--------------- Making")
    try:
        print(run(["make", "-C", make_path]).stdout)
    except subprocess.CalledProcessError as exception:
        extractor_error("[EXTRACTOR] Make failed:\n" + exception.stderr, exception.returncode)

def run_and_extract():
    ref, ref_spec = {}, {}
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    for dumppath in dumps:
        dump = ntpath.basename(dumppath)
        ref[dump] = {}
        ref_spec[dump] = {}
        for size, rounds in sizes:
            print("[EXTRACTOR]----------------------- Running " + dumppath + " N=" + str(size))
            infos = []
            spects = []
            for round in range(rounds):
                print("[EXTRACTOR]----------------------- Rounds: " + str_range(round, rounds))
                try:
                    process = run([exec_path + exec_name, str(size), dumppath, str(M_unit)], timeout=exec_timeout)
                    if process.stdout: print(stdout)
                    infos.append(extract_infos(process.stderr))
                    spects.append (load_spectrum())
                except subprocess.CalledProcessError as exception:
                    extractor_error("[EXTRACTOR] Error executing grmonty:\n" + exception.stderr,
                                                exception.returncode)
            ref[dump][size] = mean_arr_of_dicts (infos)
            ref_spec[dump][size] = mean_spec(spects)
    out = open(output_file, "w")
    out.write("ref=")
    out.write(str(ref))
    out.write("\n")
    out.write("ref_spec=")
    out.write(str(ref_spec))
    out.close()



#################################################################
######## Main

def main():
    global start_time
    start_time = time.time()
    make()
    run_and_extract()
    finish_extractor ()

if __name__ == "__main__":
    main()
