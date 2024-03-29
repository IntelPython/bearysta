import glob
import os
import time

import run


def setup_environments(env_metas, **kwargs):
    import conda_env
    envs = []
    nenvs = len(env_metas)
    for i, f in enumerate(env_metas):
        print('# Creating environment "{}" ({}/{})'.format(f, i+1, nenvs))
        envs.append(conda_env.CondaEnv(f, **kwargs))

    return envs


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', '-C', action='store_true', default=False,
                        help="Delete environments before installing packages")
    parser.add_argument('--use-existing-env', '-E', action='store_true', default=False,
                        help="Do not modify environments, install nothing but benchmarks."
                             "(this is not the same as using the 'current' environment!)")
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
            o = run.yaml.load(fd)
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
            run.run_benchmark(env, f, run_id=args.run_id, commands=args.commands,
                          run_path=args.run_path, overrides=apply_overrides,
                          suite=bname, dry_run=args.dry_run, progress=progress_bench)
    return args.dry_run  # mark the build failed in CI because no meaningful result was produced


if __name__ == '__main__':
     import sys
     sys.exit(main())
