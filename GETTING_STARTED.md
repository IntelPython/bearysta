# Getting started

This is a guide on how to get started with benchmarking using
`bearysta` on a clean system. The only assumption we make is that
you are running Linux and have access to the Internet (or can somehow transfer
files to this machine).

## An overview of the architecture

`bearysta` is composed of two parts: the *benchmark runner* and the *aggregator*.
The benchmark runner is a very simple program which handles conda environment
creation, benchmark installation, and running benchmarks with different combinations
of arguments and environment variable values. The aggregator is a more general
data processing tool, which is usually used in the context of benchmarks to
compare performance of different implementations of particular algorithms.

## Installing Python

The first thing to set up is a Python environment. You can install a fresh
Miniconda3 environment using an installer from
[this page](https://conda.io/en/latest/miniconda.html).
Specify some convenient location for the Python environment, and then
"activate" it:

```bash
. path/to/miniconda3/bin/activate
```

A `(base)` should be prepended to your prompt.

You now should install some required dependencies:

```bash
conda install -n base -c defaults pandas openpyxl ruamel_yaml
```

## Compiling benchmark packages

`bearysta` uses conda packages to manage its environments and
benchmarks. If you have access to the latest benchmark packages,
you can probably skip this step.

You can compile packages on a separate machine. Just make sure you create a
clean conda environment before following these instructions.

In order to compile packages, we need `icc`, `ifort`, and `conda-build`.
You can install `conda-build` in your base conda environment by running

```bash
conda install -n base -c defaults conda-build
```

Then, clone the *benchmark recipe repositories*: (**TODO: unpublished**)

Each of these repositories contains conda recipes for building packages.
Make a directory somewhere for compiled benchmark packages to go. We will call
this location the *benchmark channel* from here on out.

Certain recipe repositories contain a `meta.yaml` in their root, and others
contain some subdirectories. Wherever you find a `meta.yaml` file, run
`conda-build` to compile the packages:

```bash
conda build -c intel -c defaults --output-folder /path/to/benchmark/channel .
```

This should automatically send freshly compiled benchmarks to your benchmark
channel.

### If conda build complains about missing icc or ifort

If conda build complains about missing icc or ifort, you might have to modify
`build.sh` or `build_native.sh`, which contains a line which adds icc/ifort
to the path, like

```
. path/to/compiler/activation/script
```

## Preparing the benchmark channel

Now that benchmark packages are available, you will have to make them
accessible to the benchmarking machine.

If you have access to the benchmark packages on one machine, but
not on the benchmarking machine, you can run `python -m http.server` from that
directory on the machine you have access to. In that case, note the hostname of
the HTTP server machine and the port the server is running on. You will need to
modify the environment configuration files to use this benchmark *channel*.

If you have access to the benchmark packages directly (e.g. nfs) on the
benchmarking machine, note the path to the channel.

## Preparing environment configurations

You are now ready to clone `bearysta` on the benchmarking machine.
Once you clone the `bearysta` repo, you can first modify the
*environment configuration files*, which are `yaml` files in the `envs/`
directory.

You will find that each environment file follows conda's specification for
environment files (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)),
except for a special `benchmarks` and `benchmark-channels` sections. Replace
whatever `benchmark-channels` are specified in all configuration files with
the one that you want to use.

- For an http(s) channel, use `"http://hostname:port/"`
- For a file path, use `"file:///path/to/channel/"`.

At this time, you can also modify other parameters in the environment
configurations. See
(**TODO: unpublished documentation**)
for more details.


## Installing environments

You can now instruct `benchmark.py` to install environments. Make sure that
you are still in the `base` environment on the benchmarking machine, and run

```bash
python benchmark.py -C -b none --env-path envs/*
```

All environments will be created in the `envs/` directory.
The `-C` flag requests that the runner clean environments before
creating them. The `-b none` argument asks for all benchmarks called "none",
which is nothing. (this is a workaround)

You should now find that there are directories inside `envs/` corresponding
to each environment.

## Reading and writing benchmark configurations

Now that you have the environments installed with benchmarks, you can inspect
the benchmark configurations. Inside `envs/<envname>/benchmarks`, you can find
a bunch of yaml files which specify instructions for each "benchmark suite".

More detailed documentation on benchmark configurations is available here
(**TODO: unpublished documentation**)

## Writing overrides

If you find that the defaults for a benchmark suite are insufficient for
your purpose, you can write *benchmark overrides* which override run variables
or commands for a particular benchmark suite in particular environments.

The format is very similar to benchmark configurations, but contains an
`override:` section (usually at the top):

```yaml
override:
  benchmark: sklearn_daal4py # the name of the benchmark config this should override
  envs: [intelpython3] # this should only take effect in this environment
```

If the `envs` key is not specified, the override will take effect for all
environments.

## Running benchmarks

A good idea to understand how to run benchmarks can be found by running
`python benchmark.py -h`.

Generally, you can execute the command

```bash
python benchmark.py -C -b <suites to execute> -c <benchmarks to execute> --env-path envs/*.yml --overrides-path path/to/overrides/*.yaml --run-id 'descriptive_title_of_runs'
```

If you want to run all benchmark suites or all benchmarks, just omit the
`-b SUITES` and `-c BENCHMARKS` arguments. If you have no overrides, you can
omit the overrides-path argument. Be sure to set a descriptive run id.

If you want to use only a specific environment, you can specify the path to
its environment configuration directly after `--env-path`.

`benchmark.py` will dump results and metadata for benchmark runs into
`runs/run_id/...`

## Aggregating data

It's now time to invoke the aggregator. The `config/` directory contains
"aggregator recipes" which are used by the aggregator to read and transform data.
Some configurations reference other configurations and the aggregator will
fetch outputs from those configurations first and then run aggregation on the
current config file.

The aggregator supports Excel (`-x path`), CSV (`-o path`), HTML (`-H path`),
and pretty-printed (`-P [path]`) outputs. These are produced almost directly
from the corresponding pandas `DataFrame.to_*` methods.

The main structure here is as follows

- `config/raw/*.yml` - raw data. These configs actually read the data from the
  `runs/` directories and produce pandas dataframes readable by later configs.
- `config/indicators/*.yml` - aggregated and compared data. These configs take
  outputs from files in `config/raw/` and produce comparisons like
  Python/Native-C ratio.
- `config/*.yml` - main output configs. These are the ones that should be the
  nicest w.r.t. output formatting and should be exactly as we want them to be.
  These configs reference configs in `config/indicators/`.
  
Unfortunately, certain things are hardcoded into the configs, and the
aggregator isn't currently in the cleanest state (see #116), so just using
the existing configs doesn't always work. My only recommendation for now is to
write your own configuration file(s) using the configs in the configs directory
as examples.

An example invocation of the aggregator is

```bash
python aggregate.py -x sklearn.xlsx -H sklearn.html -P -- config/sklearn.yml
```

This will attempt to produce the output described in `config/sklearn.yml` in 
XLSX, HTML, and pretty-printed formats.

### Aggregator inputs
Currently, the aggregator will accept only CSV inputs with possible YAML metadata.
The YAML metadata is produced by the runner (`benchmark.py`) and consists of
lean environment information as well as all the environment variables with which
the benchmark was run. Because the benchmark might not always output all
relevant information regarding how it was run, this information might be useful
later for aggregation.

### Possible issues with aggregation

- If you're using the configs from `config/` directly and have multiple
  subdirectories under `runs/`, the aggregator will probably pick up all
  of them and have trouble differentiating between different runs.


## Notes from running 2020 benchmarks

Here is the procedure I followed to run benchmarks for 2020.

- Set up a new benchmark channel `packages_2020`
- Built all packages into that benchmark channel. For each benchmark recipe, I ran
  `conda build -c intel --output-folder packages_2020 .`
- Unpacked `pkgs` directory from the Linux 3 distribution tarball, created a conda package
  channel from there, and ran `python -m http.server` for serving packages to benchmark machines
- Started another `python -m http.server` for serving benchmark packages to benchmark machines
- Followed above instructions basically
