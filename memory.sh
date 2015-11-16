#!/bin/bash
OMP_NUM_THREADS=1 THEANO_FLAGS=mode=FAST_COMPILE,device=cpu,floatX=float32,exception_verbosity=high python memory.py 



