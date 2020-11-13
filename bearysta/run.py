import os
import sys
import shlex
from copy import deepcopy
from functools import reduce
import itertools
from subprocess import Popen, PIPE, run as run_process
import glob
import time
import platform
try:
    from ruamel.yaml import YAML
except ImportError:
    from ruamel_yaml import YAML

yaml = YAML(typ='safe')


def run_benchmark(env, config, run_path='runs', run_id=None, commands=None,
                  overrides=[], suite='benchmark', dry_run=False, progress="undefined"):
    if not run_id:
        run_id = str(time.time())
    os.makedirs('{}/{}/{}/{}/'.format(run_path, run_id, suite, env.name), exist_ok=True)

    with open(config) as f:
        bench = yaml.load(f)

    # Specify suffix for meta file of run configuration. E.g. '.csv' if $outprefix.csv is the consumable command output.
    meta_suff = bench.get('meta-suffix', '.out')

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
            # output to file
            output_prefix = '{}/{}/{}/{}/{}_{}'.format(run_path, run_id, suite,
                                                env.name, time.time(), endpoint)
            arg_run = dict(zip(keys, values))
            # a copy of visible variables
            args = arg_run.copy()
            # add automatically generated variables
            arg_run['env_name'] = env.name
            arg_run['hostname'] = platform.node()
            arg_run['outprefix'] = output_prefix

            for k, v in args.items():
                if v.startswith('$(') and v[-1] == ')':  # shell precomputed variables
                    v = v[2:-1]
                    a = os.environ.copy()
                    a.update(arg_run)
                    p = run_process(v, env=a, capture_output=True, shell=True)
                    if p.returncode:
                        print(f'# Error {p.returncode} returned while running: "{v}", stderr: {p.stderr.decode()} ...env: {a}')
                        sys.exit(2)
                    v = arg_run[k] = p.stdout.decode().strip()
                print(f'# {k} = {v}')
            del args  # used for vars dump only

            data = ''
            with env.call(cmd, env=arg_run, stdout=PIPE) as proc:
                for line in iter(proc.stdout.readline, b''):
                    data += line.decode()
                    print(line.decode(), end='')
            print("")

            with open(output_prefix + '.out', 'w') as fd:
                fd.write(data)

            # output the environment we created as well.
            with open(output_prefix + meta_suff + '.meta', 'w') as fd:
                yaml.dump(arg_run, fd)

class CurrentEnv:
    name = platform.node()

    @classmethod
    def call(self, cmd, env={}, **kwargs):
        '''Get a Popen object calling cmd, with kwargs
        passed directly to the Popen constructor.
        '''
        if isinstance(cmd, str):
            cmd = [cmd]

        cmd_str = ' '.join('"' + x.replace('"', '"\'"\'"') + '"' if any(c in x for c in list(' "$')) else x for x in cmd)
        # The previous line quotes each arg if it contains any of (space,
        # double quote, dollar sign) in double quotes, replacing any inner
        # double quotes with the sequence "'"'" (closing first quote, starting
        # single quote, actually writing the double quote, closing single
        # quote, then restarting the double quote)

        print('#$ ' + cmd_str.replace(' && ', ' && \\\n#> '))

        env = env.copy()
        if 'PATH' not in env:
            env['PATH'] = os.getenv('PATH', '')
        if 'LD_LIBRARY_PATH' not in env:
            env['LD_LIBRARY_PATH'] = os.getenv('LD_LIBRARY_PATH', '')
        return Popen(cmd_str, env=env, shell=True, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', '-b', default=None, nargs='+',
                        help='Run specified benchmark names only')
    parser.add_argument('--commands', '-c', default=None, nargs='+',
                        help='Run these benchmark commands only')
    parser.add_argument('--bench-path', default=None, nargs='+',
                        help='Read these benchmark configurations. benchmarks/ is by default')
    parser.add_argument('--overrides-path', default=[], nargs='+',
                        help='Read these benchmark overrides.')
    parser.add_argument('--run-id', default=time.strftime('%Y-%m-%d_%H_%M_%S'),
                        help='Directory under the run path where outputs will go')
    parser.add_argument('--run-path', default='runs',
                        help='Directory where run id directories will go')
    parser.add_argument('--dry-run', '-d', default=False, action='store_true',
                        help='Print benchmark commands instead of running them properly')
    args = parser.parse_args()

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
        overrides.append(o)

    # Get benchmark yamls
    if args.bench_path:
        benchmarks = args.bench_path
    else:
        benchmarks = glob.glob(os.path.join('benchmarks', '*.yaml'))

    # The user might have required specific benchmarks
    if args.benchmarks is not None:
        benchmarks = [b for b in benchmarks if
                os.path.splitext(os.path.basename(b))[0] in args.benchmarks]

    env = CurrentEnv()
    nbenches = len(benchmarks)
    for j, f in enumerate(benchmarks):

        bname = os.path.splitext(os.path.basename(f))[0]
        progress_bench = 'benchmark "{}" ({}/{})'.format(bname, j+1, nbenches)
        print('#### Running', progress_bench)
        # Run benchmark
        apply_overrides = [o for o in overrides if (bname in o['override']['benchmark'])]
        run_benchmark(env, f, run_id=args.run_id, commands=args.commands,
                        run_path=args.run_path, overrides=apply_overrides,
                        suite=bname, dry_run=args.dry_run, progress=progress_bench)
    return args.dry_run  # mark the build failed in CI because no meaningful result was produced


if __name__ == '__main__':
     sys.exit(main())
