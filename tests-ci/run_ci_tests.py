import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("Command failed")
        sys.exit(result.returncode)
        
def ensure_define(filepath, key, value):
    lines = Path(filepath).read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"#define {key}"):
            line = f"#define {key} {value}"
        new_lines.append(line)
    Path(filepath).write_text("\n".join(new_lines))
    
import re

def ensure_makefile_model():
    mf = Path("makefile")
    content = mf.read_text()
    content = re.sub(r'^(MODEL\s*=\s*).*$', r'\1sphere', content, flags=re.MULTILINE)
    mf.write_text(content)
    
def configure_headers():
    config = ROOT / "src/config.h"
    ensure_define(config, "SCATTERINGS_PER_PHOTON", "(1)")
    ensure_define(config, "MAX_LAYER_SCA", "(1)")
    sphere = ROOT / "src/sphere_model/model.h"
    defines = {
        "NUMIN": "1.e8", "NUMAX": "1.e24", "THETAE_MIN": "0.3", "THETAE_MAX": "1000.",
        "TP_OVER_TE": "(3.)", "WEIGHT_MIN": "(0.)", "RMAX": "(10000.)", "ROULETTE": "1.e4",
        "R_RECORD": "(3000.)", "RMIN": "(0.01)", "BHSPIN": "0", "NE_VALUE": "(1.e13)",
        "B_VALUE": "(1.)", "THETAE_VALUE": "(100.)", "SPHERE_RADIUS": "(1.)", "dlE": "(0.015)",
        "N_ESAMP": "2500", "N_EBINS": "2500", "NPRIM": "8", "SPHERE_TEST": "(1)",
    }
    for k, v in defines.items():
        ensure_define(sphere, k, v)
        
def compile_code():
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    
    print("Installing dependencies...")
    run("apt-get update && apt-get install -y libhdf5-dev libgsl-dev")
    
    for var in ["LDFLAGS", "LDLIBS", "LIBRARY_PATH", "CFLAGS", "CXXFLAGS"]:
        os.environ.pop(var, None)
    
    run("make clean")
    print("Building executable...")
    run(
        "make"
        " CUDA_PATH=/usr/local/cuda"
        " GSL_PATH=/usr"
        " HDF5_INCLUDE='-I/usr/include/hdf5/serial'"
        " 'HDF5_LIB=-L/usr/lib/x86_64-linux-gnu/hdf5/serial -L/usr/local/nvidia/lib64'"
        " -j$(nproc)"
    )
    if not Path("gpumonty").exists():
        print("ERROR: gpumonty binary not found after build — linker likely failed silently")
        sys.exit(1)

def run_test(par, script):
    os.environ["OMP_NUM_THREADS"] = "8"
    run(f"./gpumonty -par {par}")
    run(f"python3 {script}")
    
def main():
    print("Configuring CI test environment...")
    ensure_makefile_model()
    configure_headers()
    print("Compiling code...")
    compile_code()
    print("Running tests...")
    run_test("./tests-ci/sphere-emission-test-synchrotron.par", "./tests-ci/sphere-emission-test-synchrotron.py")
    run_test("./tests-ci/sphere-emission-test-bremsstrahlung.par", "./tests-ci/sphere-emission-test-bremsstrahlung.py")
    run_test("./tests-ci/sphere-emission-test-mixed.par", "./tests-ci/sphere-emission-test-mixed.py")
    print("\n All tests passed!")
if __name__ == "__main__":
    main()