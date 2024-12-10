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
apptainer build --disable-cache turbo_pe.sif turbo_pe.def
```

Then to use the environment:
```
apptainer run --nv -B $PWD turbo_pe.sif python3 demo_analysis.py
```

The `--nv` requests a GPU while the `-B $PWD` binds the current directory inside the container 





