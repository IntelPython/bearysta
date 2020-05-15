from subprocess import Popen, PIPE
import os
import json
import platform
import shutil # python 3.3+
import shlex # python 3.3+ for shlex.quote
import tempfile
import re
try:
    from ruamel.yaml import YAML
except ImportError:
    from ruamel_yaml import YAML

yaml = YAML(typ='safe')

setupComponentsList = ['intelPython', 'stockPython', 'compiler', 'libraries']


class CondaError(Exception):
    '''Some error from conda'''
    pass


class CondaEnv:
    '''Represents a conda environment.'''


    def __init__(self, fn=None, prefix=None, clobber=False,
                 skip_listing=False):
        '''Create or reference an existing environment with the given prefix.
        If an environment already exists at the given or implied prefix,
        the specified packages will be installed.

        Try not to have the same packages with pip and conda.
        If you install a pip package, then install the same package with conda,
        pip will get confused and show the wrong output.

        One of fn, prefix must be non-None.

        Parameters
        ----------
        fn : str, optional
            The conda environment.yml file to use for env creation.
            This can include special sections "benchmark-channels"
            and "benchmarks" which are forced to be installed after the
            rest of the environment. If not specified, we will try to
            use an existing environment at the given prefix.
        prefix : str, optional
            The prefix for the conda environment. If not specified, we will
            use ./envs/<name>.
        clobber : bool, optional (default False)
            if True, destroy the current environment if any exists.
            This prevents weird things like updating certain packages
            and not getting the right dependencies.
        skip_listing : bool, optional (default False)
            if True, don't spend time listing packages.
        '''

        if fn is None and prefix is None:
            raise ValueError('One of fn, prefix must be non-None.')

        if fn:
            with open(fn) as f:
                config = yaml.load(f)
                self.name = config['name']

        # Deduce prefix from the env config if it wasn't given.
        if prefix is None:
            self.prefix = os.path.join(os.getcwd(), 'envs', self.name)
        else:
            self.prefix = prefix

        self.base_prefix = self.get_base_prefix()

        # If a configuration was passed, use it.
        if fn:

            # Check if the prefix already exists.
            if os.path.exists(self.prefix) and not clobber:

                # Install pip and conda packages separately...
                pip_pkgs = []
                conda_pkgs = []
                for pkg in config['dependencies']:
                    if type(pkg) is not str:
                        pip_pkgs = pkg
                    else:
                        conda_pkgs.append(pkg)

                self.install_packages(conda_pkgs, installer='conda',
                                      channels=config['channels'])
                self.install_packages(pip_pkgs, installer='pip')
            else:
                # We first should write a config for conda because it doesn't
                # like to see any extra keys in the config
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yml') as tmp:
                    conda_env_config = {k: config[k] for k in ['name', 'channels',
                                                               'dependencies']}
                    yaml.dump(conda_env_config, tmp)
                    tmp.flush()

                    # conda env create is for using environment files, while
                    # conda create is for command-line creation...
                    create_cmd = ['env', 'create']
                    create_cmd += self._get_prefix_args()
                    create_cmd += ['--file', tmp.name]

                    if clobber:
                        create_cmd += ['--force']

                    self._call_conda(create_cmd)

            self.packages = self.get_packages()


        # Install benchmarks if specified in the config.
        if 'benchmarks' in config:
            channels = config.get('benchmark-channels', [])
            # FIXME GH-100 conda still runs the solver here, even though we
            # ask for --no-deps, so we prevent it from failing by adding
            # our channels...
            channels.extend(config['channels'])
            self.install_packages(config['benchmarks'], installer='conda',
                                  no_deps=True, copy=True, channels=channels)

        if not skip_listing:
            self.packages = self.get_packages()


    # These conda-calling functions are simplified versions of those
    # found in the conda-api package, which is unfortunately very old now.
    def _call_conda(self, args, check=True, pipe=False):
        '''Calls the conda on the PATH with the given args.'''

        cmd = ['conda'] + args
        display_cmd = ' '.join(shlex.quote(x) for x in cmd)
        print('> ' + display_cmd)
        if pipe:
            process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        else:
            process = Popen(cmd)

        stdout, stderr = process.communicate()
        if check and process.returncode != 0:
            msg = 'conda {} returned code {}\n\n'.format(display_cmd, process.returncode)
            if pipe:
                msg += 'STDOUT:\n{}\n\nSTDERR:\n{}\n\n'.format(stdout.decode(), stderr.decode())
            raise Exception(msg)
        return stdout, stderr


    def _call_and_parse(self, args, check_stderr=True, check=True):
        '''Calls the conda on the PATH with the given args, and parses
        stdout. If there is anything on stderr, this will throw a CondaError.
        '''

        stdout, stderr = self._call_conda(args, check=check, pipe=True)
        if check_stderr and stderr.decode().strip():
            # There was something output to stderr
            raise CondaError(stderr.decode())

        return json.loads(stdout.decode())


    def _get_prefix_args(self):
        '''Get a conda arg pointing to this prefix'''
        return ['--prefix', self.prefix]


    def _get_channel_args(self, channels=None):
        '''Get conda args which point to these channels'''

        args = []
        for c in channels:
            args += ['--channel', c]
        return args


    def _get_prefix_bin(self, prefix=None):

        if platform.system() == 'Windows':
            dir = 'Scripts'
        else:
            dir = 'bin'

        if prefix is None:
            prefix = self.prefix

        return os.path.join(prefix, dir)


    def _get_source_cmd(self):

        return ['.', os.path.join(self.get_base_prefix(), 'etc', 'profile.d', 'conda.sh')]


    def _get_activate_cmd(self, prefix=None):

        if prefix is None:
            prefix = self.prefix

        return ['conda', 'activate', prefix]


    def conda_info(self):

        return self._call_and_parse(['info', '--json'])


    def get_base_prefix(self):

        if hasattr(self, 'base_prefix'):
            return self.base_prefix

        return self.conda_info()['sys.prefix']


    def call(self, cmd, env={}, **kwargs):
        '''Get a Popen object calling cmd in this env, with kwargs
        passed directly to the Popen constructor.
        '''
        if type(cmd) is str:
            cmd = [cmd]

        # Put the conda env's bin or Scripts in front of the current PATH.
        activate_cmd = ''
        activate_cmd += ' '.join(shlex.quote(x) for x in self._get_source_cmd())
        activate_cmd += ' && '
        activate_cmd += ' '.join(shlex.quote(x) for x in self._get_activate_cmd())
        activate_cmd += ' && '
        activate_cmd += ' '.join('"' + x.replace('"', '"\'"\'"') + '"' if any(c in x for c in list(' "$')) else x for x in cmd)
        # The previous line quotes each arg if it contains any of (space,
        # double quote, dollar sign) in double quotes, replacing any inner
        # double quotes with the sequence "'"'" (closing first quote, starting
        # single quote, actually writing the double quote, closing single
        # quote, then restarting the double quote)

        print('#$ ' + activate_cmd.replace(' && ', ' && \\\n#> '))

        env['LD_LIBRARY_PATH'] = os.path.join(self.prefix, 'lib') + \
                ((":" + env['LD_LIBRARY_PATH']) if 'LD_LIBRARY_PATH' in env else '')

        if 'PATH' not in env:
            env['PATH'] = os.environ['PATH']
        return Popen(activate_cmd, env=env, shell=True, **kwargs)


    def call_communicate(self, cmd, check=True, **kwargs):
        '''Return the results of communicating with the conda env call'''
        if type(cmd) is str:
            cmd = [cmd]

        process = self.call(cmd, **kwargs)
        stdout, stderr = process.communicate()
        if check and process.returncode != 0:
            display_cmd = ' '.join(shlex.quote(x) for x in cmd)
            msg = '{} returned code {}\n\n'.format(display_cmd, process.returncode)
            if stdout:
                msg += 'STDOUT:\n{}\n'.format(stdout.decode())
            if stderr:
                msg += 'STDERR:\n{}\n'.format(stderr.decode())
            raise Exception(msg)

        if stdout: stdout = stdout.decode()
        if stderr: stderr = stderr.decode()

        return stdout, stderr


    def call_stdout(self, cmd, **kwargs):
        '''Get stdout of the given command invoked in the conda env.'''
        stdout, _ = self.call_communicate(cmd, stdout=PIPE, **kwargs)
        return stdout


    def install_packages(self, pkgs, installer='conda', channels=[],
                         override_channels=False, no_deps=False, copy=False,
                         no_update_deps=False):
        '''Install the given package specifications into this env.'''

        if type(pkgs) is str:
            pkgs = [pkgs]

        if installer == 'conda':
            install_cmd = ['install', '--yes']
            install_cmd += self._get_prefix_args()
            install_cmd += self._get_channel_args(channels=channels)

            if override_channels:
                install_cmd += ['--override-channels']
            if no_deps:
                install_cmd += ['--no-deps']
            if no_update_deps:
                install_cmd += ['--no-update-deps']
            if copy:
                install_cmd += ['--copy']

            install_cmd.extend(pkgs)
            self._call_conda(install_cmd)
        elif installer == 'pip':
            # Ensure pip
            if not 'pip' in self.packages:
                self.install_packages('pip')

            for pkg in pkgs:
                install_cmd = ['pip', 'install', '-U', pkg]
                self.call_communicate(install_cmd, env=dict(os.environ))
        else:
            raise ValueError('Unsupported installer')
        self.packages = self.get_packages()


    def get_packages(self):
        '''Get a dict of package name -> package info'''

        found_pip = False
        packages = {}

        info_cmd = ['list', '--json']
        info_cmd += self._get_prefix_args()
        conda_list = self._call_and_parse(info_cmd)
        for pkg in conda_list:
            pkg['installer'] = 'conda'
            packages[pkg['name']] = pkg
            if pkg['name'] == 'pip':
                found_pip = True

        if found_pip:
            info_cmd = ['pip', 'freeze', '--quiet', '--local']
            pip_list = self.call_stdout(info_cmd).split()
            pip_list = [s.split('==') for s in pip_list]

            for pkg in pip_list:
                name = pkg[0]
                ver = pkg[1]

                # Sometimes the pip names for conda packages are different
                # for e.g. Werkzeug
                if name.lower() in packages: continue
                # for e.g. mkl_fft
                if name.lower().replace('-', '_') in packages: continue
                # for e.g. backports.weakref
                if '.' in name: continue
                # for e.g. pytables
                if 'py' + name in packages: continue

                packages[name.lower()] = {'name': name, 'version': ver, 'installer': 'pip'}

        return packages


