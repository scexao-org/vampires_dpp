import multiprocessing
import os

# --- Step 1: Set environment variables for thread safety ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- Step 2: Import multiprocessing after setting environment ---
multiprocessing.set_start_method("spawn", force=True)
