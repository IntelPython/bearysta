# The benchmark runner

The benchmark runner constructs/modifies environments for benchmarking and manages environment variables
and arguments passed to benchmarks. It can run any program which can be invoked from the command line.

In general, the runner processes two types of configuration files. The **environment configurations**
specify what packages to install into the environment, from which channels they should be installed,
and the name of the environment. The **benchmark configurations** specify what commands should be run,
the environment variables, and the arguments to pass.

Most benchmark packages come with their own benchmark configurations, so if you want to tweak the
arguments or environment variables, the runner provides the ability to specify **configuration
overrides**. These allow specifying different "variables" from which environment variables and
arguments are constructed.

```
usage: benchmark.py [-h] [--clean] [--use-existing-env] [--skip-package-listing]
                    [--benchmarks BENCHMARKS [BENCHMARKS ...]] [--commands COMMANDS [COMMANDS ...]]
                    [--env-path ENV_PATH [ENV_PATH ...]] [--bench-path BENCH_PATH [BENCH_PATH ...]]
                    [--overrides-path OVERRIDES_PATH [OVERRIDES_PATH ...]] [--run-id RUN_ID]
                    [--run-path RUN_PATH] [--dry-run] [--quick]

optional arguments:
  -h, --help            show this help message and exit
  --clean, -C           Delete environments before installing packages
  --use-existing-env, -E
                        Use the current conda environment, install nothing but benchmarks
  --skip-package-listing
                        Skip 'pip freeze' and 'conda list' steps
  --benchmarks BENCHMARKS [BENCHMARKS ...], -b BENCHMARKS [BENCHMARKS ...]
                        Run benchmarks defined only in these configuration files
  --commands COMMANDS [COMMANDS ...], -c COMMANDS [COMMANDS ...]
                        Run these benchmark commands
  --env-path ENV_PATH [ENV_PATH ...]
                        Read these environment configurations
  --bench-path BENCH_PATH [BENCH_PATH ...]
                        Read these benchmark configurations. If left unspecified, benchmark
                        configurations will be read from $CONDA_PREFIX/benchmarks for each env.
  --overrides-path OVERRIDES_PATH [OVERRIDES_PATH ...]
                        Read these benchmark overrides.
  --run-id RUN_ID       Directory under the run path where outputs will go
  --run-path RUN_PATH   Directory where run id directories will go
  --dry-run, -d         Print benchmark commands instead of running them properly
  --quick, -Q           Enable all quick options, i.e. use_existing_env, skip_package_listing
```

## Environment configuration

Environment configurations define how automated_benchmarks should create environments.
They are located in the `envs/` directory, are in YAML format, and have file names ending in `.yaml`.
Note that all environments are conda environments. This means that one can only install python and pip from
conda, but all other packages can be installed from conda or pip.

### Example
```yaml
# Example configuration file for environment.py
# This defines an IDP environment.

# Name of the environment.
name: intelpython3

# Channel from which conda packages should be installed
channels:
- intel
- defaults

# Packages to install using conda
dependencies:
- numpy
- numba
- icc_rt # we should specify this for Numba SVML
- scipy
- scikit-learn
- jinja2
- daal
- daal4py
- tbb
- cython
- numexpr
- pip # conda asks that we specify pip explicitly
- pip: [rdtsc] # this list contains packages to install from pip

# The following is to be treated as a workaround. As of the time of writing,
# conda called with --no-deps would still run the dependency solver and fail
# if dependencies were not available in specified channels.
benchmark-channels:
- 'file://path/to/channel'

benchmarks:
- optimizations_bench_python
- blackscholes_bench_python
- fft_bench_python
- sklearn_bench_python
- sklearn_bench_daal4py
- sklearn_bench_data
- ibench
```

## Benchmark configuration

All benchmarks are defined in `.yaml` configs, and provided by packages
in `$CONDA_PREFIX/benchmarks/`. Each config defines global environment variables, command lines (run with `sh`),
and local variables. Command lines can thus contain parameter substitution.

We use the Cartesian product of all variables defined for a particular command (where local variables override
global ones) to run it with all combinations of variables.

For example, for ibench_native, one may write:

```yaml
variables:
  bench: [det_native, dot_native, inv_native, lu_native]
  size: [test, tiny, small, large]

commands:
  ibench_native: IBENCH_PLUGINS=ibench_native python -m ibench run --benchmarks $bench --size $size --runs 10
```

### Command-specific variables
If different problem sizes are desired for e.g. serial vs. parallel versions, a slightly more complicated
configuration is possible:

```yaml
variables:
  bench: [det_native, dot_native, inv_native, lu_native]
  size: [test, tiny, small, large]

commands:
  ibench_native_parallel: IBENCH_PLUGINS=ibench_native python -m ibench run --benchmarks $bench --size $size --runs 10
  ibench_native_serial: 
    variables:
      size: [test, tiny, small] # overrides globally-defined size for this command only
    command: OMP_NUM_THREADS=1 MKL_THREADING_LAYER=sequential IBENCH_PLUGINS=ibench_native python -m ibench run --benchmarks $bench --size $size --runs 10
```

### Predefined variables
The following variables are automatically defined by `benchmark.py`. If you set a variable with this name, it will
automatically be overridden by the benchmarking system.

- `env_name`: the name given in the environment configuration.
- `hostname`: the name of the machine, as given by `platform.node()`

### Config overrides
A user can override any aspect of benchmark configurations without directly editing any, by simply creating override configurations inside the `overrides/` directory, ending in `.yaml`. For example, the following could work to disable sequential mode for scikit-learn python benchmarks, and restrict the kmeans size to the given one:
```yaml
override:
  benchmark: sklearn_python
  envs: [intelpython3, stockpython3] # or omit to mean all envs

# Basically a benchmark configuration follows, except any values in
# `variables` are updated with these.
variables:
  threads: -1 # parallel mode only

commands:
  kmeans:
    variables:
      size: 500000x5 # pick this size only for kmeans
```

### Internal comments
- All constant variable definitions become a one-element list. That is, if you specify
  ```yaml
  variables:
    size: 1000
  ```
  the scripts will think internally
  ```yaml
  variables:
    size: [1000]
  ```
- Commands defined as strings become a dictionary. That is, if you specify
  ```yaml
  commands:
    my_bench: my_bench -v $size
  ```
  the scripts will think internally
  ```yaml
  commands:
    my_bench:
      command: my_bench -v $size
  ```
  This way, you need not worry about the way the benchmarks were defined or overridden, as long as you use one of the valid formats.

## Benchmark overrides

To specify different environment variables or arguments, write an override file.
Override files have the same format as benchmark configurations, but have an additional
header.

### Example
```yaml
override:
    benchmark: blackscholes_python # the benchmark to override
    envs: [intelpython3] # the environments to override in, or all environments if not present

# Overrides for global variables. Variables specified here basically replace
# variables which were defined in the benchmark config.
variables:
    batch: [0, 1, 2, 3]
    #impl: [dask, dask_numpy, naive, numba_guvec_par, numba_guvec_simd_par,
    #       numba_jit, numba_jit_par, numba_vec, numba_vec_par, numexpr,
    #       numexpr_crunched, numpy]
    impl: [unused]
    #steps: 15
    #step_size: 2
    #chunk: 2000000
    #size: 1024

# Similar behavior is with commands. If only a command is specified as in this
# example, it will only replace the command of the benchmark, not the
# command-specific variables
commands:
    blackscholes_python_thr: python -W ignore -m blackscholes_bench.bs_erf_numpy --steps $steps --step $step_size --chunk $chunk --size $size --text $env_name
    blackscholes_python_seq: MKL_THREADING_LAYER=sequential OMP_NUM_THREADS=1 python -W ignore -m blackscholes_bench.bs_erf_numpy --steps $steps --step $step_size --chunk $chunk --size $size --text $env_name
    blackscholes_numba_thr: NUMBA_THREADING_LAYER=omp python -W ignore -m blackscholes_bench.bs_erf_numba_jit_par --steps $steps --step $step_size --chunk $chunk --size $size --text $env_name
    blackscholes_numba_seq: python -W ignore -m blackscholes_bench.bs_erf_numba_jit --steps $steps --step $step_size --chunk $chunk --size $size --text $env_name
```
