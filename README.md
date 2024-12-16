# FlowSGVB

Running multivariate PSD estimation along signal PE using jax + variational inference



## Environment

Needs py 3.10

Locally:
```
pip install jax[cuda12] jimgw flowMC ripplegw
```

On OzStar 
```
module load apptainer
apptainer build --disable-cache turbo_pe.sif turbo_pe.def
```

Then to use the environment:
```
module load apptainer
apptainer run --nv -B $PWD turbo_pe.sif python demo_analysis.py
```

The `--nv` requests a GPU while the `-B $PWD` binds the current directory inside the container 





