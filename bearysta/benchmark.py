import glob
import os
import shlex
from copy import deepcopy
from collections import namedtuple
from functools import reduce
import itertools
from .environment import CondaEnv
from subprocess import PIPE
import time
import re
import platform
try:
    from ruamel.yaml import YAML
except ImportError:
    from ruamel_yaml import YAML

yaml = YAML(typ='safe')


def setup_environments(env_metas, **kwargs):

    envs = []
    nenvs = len(env_metas)
    for i, f in enumerate(env_metas):
        print('# Creating environment "{}" ({}/{})'.format(f, i+1, nenvs))
        envs.append(CondaEnv(f, **kwargs))

    return envs


def run_benchmark(env, config, run_path='runs', run_id=None, commands=None,
                  overrides=[], suite='benchmark', dry_run=False, progress="undefined"):

    if not run_id:
        run_id = str(time.time())
    os.makedirs('{}/{}/{}/{}/'.format(run_path, run_id, suite, env.name), exist_ok=True)

    with open(config) as f:
        bench = yaml.load(f)

    # Make sure we at least have an empty vars list
    if 'variables' not in bench:
        bench['variables'] = {}

    # If any command is given directly as the command value, change it to dict
    for cmd, cmdc in bench['commands'].items():
        if type(cmdc) is not dict:
            bench['commands'][cmd] = {'command': cmdc}

    # Evaluate configuration overrides
    for override in overrides:
        bench['variables'].update(override.get('variables', {}))
        for cmd, cmdc in override.get('commands', {}).items():
            # Allow just a string
            if type(cmdc) is str:
                if cmdc.strip() == 'drop':
                    del bench['commands'][cmd]
                    continue
                cmdc = {'command': cmdc}
            if cmd in bench['commands']:
                # Change an already existent command config
                if 'variables' in bench['commands'][cmd]:
                    bench['commands'][cmd]['variables'].update(cmdc.pop('variables', {}))
                bench['commands'][cmd].update(cmdc)
            else:
                # Add a new command config
                bench['commands'][cmd] = cmdc

    ncommands = len(bench['commands'])
    for i, endpoint in enumerate(bench['commands']):
        progress_inside = '{} ({}/{}) in {}'.format(endpoint, i+1, ncommands, progress)
        print('### Running benchmark endpoint', progress_inside)

        if commands is not None and endpoint not in commands:
            print('[skipped this endpoint]')
            continue

        var_matrix = deepcopy(bench['variables'])

        cmd = bench['commands'][endpoint]

        if type(cmd) is dict:
            var_matrix.update(cmd.get('variables', {}))
            cmd = cmd['command']

        var_matrix = {k: (var_matrix[k] if type(var_matrix[k]) is list
                        else [var_matrix[k]]) for k in var_matrix}
        var_matrix = {k: [str(i) for i in var_matrix[k]] for k in var_matrix}

        # Allow running e.g. 'python -m ibench'
        if type(cmd) is str:
            cmd = shlex.split(cmd)

        if dry_run:
            cmd = ["echo", "\#skipped:"] + cmd

        # Cartesian product of arguments
        keys = var_matrix.keys()
        vals = var_matrix.values()
        product_length = reduce(lambda x, y: x * len(y), vals, 1)
        for i, values in enumerate(itertools.product(*vals)):
            print('## Running combination {}/{} of {}' .format(i+1, product_length, progress_inside))
            arg_run = dict(zip(keys, values))
            for k, v in arg_run.items():
                print(f'# {k} = {v}')

            # add automatically generated variables
            arg_run['env_name'] = env.name
            arg_run['hostname'] = platform.node()

            data = ''
            with env.call(cmd, env=arg_run, stdout=PIPE) as proc:
                for line in iter(proc.stdout.readline, b''):
                    data += line.decode()
                    print(line.decode(), end='')
            print("")

            # output to file
            output_prefix = '{}/{}/{}/{}/{}_{}.out'.format(run_path, run_id, suite,
                                                       env.name, time.time(),
                                                       endpoint)
            with open(output_prefix, 'w') as fd:
                fd.write(data)

            # output the environment we created as well.
            with open(output_prefix + '.meta', 'w') as fd:
                yaml.dump(arg_run, fd)


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', '-C', action='store_true', default=False,
                        help="Delete environments before installing packages")
    parser.add_argument('--use-existing-env', '-E', action='store_true', default=False,
                        help="Use the current conda environment, install nothing but benchmarks")
    parser.add_argument('--skip-package-listing', action='store_true', default=False,
                        help="Skip 'pip freeze' and 'conda list' steps")
    parser.add_argument('--benchmarks', '-b', default=None, nargs='+',
                        help='Run benchmarks defined only in these configuration files')
    parser.add_argument('--commands', '-c', default=None, nargs='+',
                        help='Run these benchmark commands')
    parser.add_argument('--env-path', default=glob.glob('envs/*.yml'), nargs='+',
                        help='Read these environment configurations')
    parser.add_argument('--bench-path', default=None, nargs='+',
                        help='Read these benchmark configurations. '
                        'If left unspecified, benchmark configurations will be '
                        'read from $CONDA_PREFIX/benchmarks for each env.')
    parser.add_argument('--overrides-path', default=[], nargs='+',
                        help='Read these benchmark overrides.')
    parser.add_argument('--run-id', default=time.strftime('%Y-%m-%d_%H_%M_%S'),
                        help='Directory under the run path where outputs will go')
    parser.add_argument('--run-path', default='runs',
                        help='Directory where run id directories will go')
    parser.add_argument('--dry-run', '-d', default=False, action='store_true',
                        help='Print benchmark commands instead of running them properly')

    quick = ['use_existing_env', 'skip_package_listing']
    class SetQuickAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            for q in quick:
                setattr(namespace, q, True)

    parser.add_argument('--quick', '-Q', action=SetQuickAction, nargs=0,
                        help="Enable all quick options, i.e. " + ', '.join(quick))
    args = parser.parse_args()

    print('###### Preparing environments... ######\n')
    env_kwargs = dict(clobber=args.clean, skip_listing=args.skip_package_listing)

    envs = setup_environments(args.env_path, **env_kwargs)

    overrides = []
    for f in args.overrides_path:
        with open(f) as fd:
            o = yaml.load(fd)
        if 'override' not in o:
            print('# WARNING: ignoring invalid override at {}'.format(f))
            continue
        for k, v in o['override'].items():
            if type(v) is str:
                o['override'][k] = [v]

        o['override']['envs'] = o['override'].get('envs', [e.name for e in envs])
        overrides.append(o)

    print('\n###### Running benchmarks... #######')
    nenvs = len(envs)
    for i, env in enumerate(envs):
        progress_env = 'environment "{}" ({}/{})'.format(env.name, i+1, nenvs)
        print('##### Selected', progress_env)
        # Get benchmark yamls
        if args.bench_path:
            benchmarks = args.bench_path
        else:
            benchmarks = glob.glob(os.path.join(env.prefix, 'benchmarks', '*.yaml'))

        # The user might have required specific benchmarks
        if args.benchmarks is not None:
            benchmarks = [b for b in benchmarks if
                    os.path.splitext(os.path.basename(b))[0] in args.benchmarks]

        nbenches = len(benchmarks)
        for j, f in enumerate(benchmarks):
            bname = os.path.splitext(os.path.basename(f))[0]
            progress_bench = 'benchmark "{}" ({}/{}) via {}'.format(bname, j+1, nbenches, progress_env)
            print('#### Running', progress_bench)
            # Run benchmark
            apply_overrides = [o for o in overrides if (bname in o['override']['benchmark'] and env.name in o['override']['envs'])]
            run_benchmark(env, f, run_id=args.run_id, commands=args.commands,
                          run_path=args.run_path, overrides=apply_overrides,
                          suite=bname, dry_run=args.dry_run, progress=progress_bench)
    return args.dry_run  # mark the build failed in CI because no meaningful result was produced


if __name__ == '__main__':
     import sys
     sys.exit(main())
