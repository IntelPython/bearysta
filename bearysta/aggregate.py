import logging
import os
import re
import pandas as pd
import numpy as np
import sys
import glob
import tempfile
import pkg_resources
try:
    from ruamel.yaml import YAML
except ImportError:
    from ruamel_yaml import YAML
yaml = YAML(typ='safe')

pd.options.display.max_colwidth = 50
pd.options.display.width = None


def groupby_empty(df, by, as_index=True):
    '''Similar to df.groupby(by), but don't
    fail if "by" is an empty list. This will
    instead return a list of length 1 containing
    a tuple of an empty tuple and the source dataframe.

    Note that in this case, the output will always be
    as_index=False!
    '''
    if len(by) == 0:
        dummy_name = '_dummy'
        while dummy_name in df.columns:
            dummy_name = '_' + dummy_name

        df = df.copy()
        df[dummy_name] = 0
        return df.groupby(dummy_name, as_index=False)
    else:
        return df.groupby(by, as_index=as_index)


class BenchmarkError(Exception):

    def __init__(self, message, original=None):

        self.message = message
        self.original = original


    def __str__(self):

        return self.message


class Benchmark:

    def __init__(self, config_path, logger_name='benchmark'):
        '''Load this benchmark configuration, given
        a path to a benchmark config file.
        '''

        self.logger = logging.getLogger(logger_name)

        self.config_path = config_path
        # Load the JSON configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.load(f)
        except yaml.YAMLError as e:
            raise BenchmarkError('Could not decode configuration at ' + config_path, e)
        except OSError as e:
            raise BenchmarkError('Could not open configuration at ' + config_path, e)

        # Check for configurations to add to input and get dataframes if any exist
        self.dataframes = []
        self.input_column = "File"
        if 'config' in self['input']:
            if type(self['input']['config']) == str:
                self.config['input']['config'] = [self['input']['config']]
            for config in self['input']['config']:
                if config[0] != '/':
                    config = os.path.join(os.path.dirname(config_path), config)

                self.logger.info('Reading config at '+ config)
                bench = Benchmark(config)
                df = bench.get_normalized_data()
                # overwrite existing values inherited from data loading
                df[self.input_column] = os.path.splitext(os.path.basename(config))[0]
                self.logger.debug('Imported dataframe:')
                self.logger.debug('\n%s' % df)
                self.dataframes.append(df)

        # Deal with aggregation functions pandas doesn't provide
        if self.config['aggregation'] == 'geomean':
            try:
                from scipy.stats.mstats import gmean as geomean
            except ImportError:
                geomean = lambda x: np.exp(np.mean(np.log(x)))
            self.config['aggregation'] = geomean


        # Deal with empty axis, variants
        if 'variants' not in self.config:
            self.config['variants'] = []
        if 'series' not in self.config:
            self.config['series'] = []
        if 'axis' not in self.config:
            self.config['axis'] = []

        # cache for read_csv_cached function
        self.cached_csvs = {}


    def write_config(self, config_path):
        '''Write this benchmark's configuration to
        the given path.
        '''

        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)


    def __getitem__(self, name):
        '''Get the config option with name 'name'.
        Returns None if that config option is undefined.
        '''

        try:
            return self.config[name]
        except KeyError:
            return None


    def get_raw_data(self, inputs=None):
        '''Get a pandas DataFrame containing all data for this benchmark.
        This is RAW DATA, and has not been aggregated or filtered yet.

        inputs: list of str (default None)
            Only effective when df is None. When None, load input files using the
            config-defined glob. Otherwise, use the passed in array of strings
            as paths to csv inputs which are to be concatenated in pandas.
        '''

        # Get list of all files in the output location
        if inputs is None:
            if 'path' in self['input']:
                paths = self['input']['path']
                globbed = paths
                if type(paths) is str:
                    matching_files = glob.glob(paths)
                else:
                    matching_files = []
                    for i in paths:
                        matching_files += glob.glob(i)
            else:
                matching_files = []
                globbed = None
        else:
            globbed = inputs
            matching_files = []
            for i in inputs:
                matching_files += glob.glob(i)

        if len(matching_files) == 0 and globbed is not None:
            raise BenchmarkError('No matching files found for %s' % (globbed))

        dataframes = self.dataframes

        self.logger.debug('Opening %s for reading...' % matching_files)
        for f in matching_files:
            dataframes.append(self.read(f))

        return pd.concat(dataframes, ignore_index=True, sort=True)


    def read(self, fn):

        # Perform filtering. The way this works is:
        # - if no filter key exists, use all lines
        # - if a filter key exists, for each (key, value) pair in the dict,
        #   perform any replacements using key as regex and value as repl,
        #   unless value is null.
        #   If a line is not matched by any key, do not parse the line.
        # - If a line is not parsed, it can be logged, or TODO dropped
        #   from the log.
        if 'filter' in self['input']:
            filters = self['input']['filter']
            replacements = [(re.compile(k), v) for k, v in filters.items()]

            # Implicit drop for empty lines
            replacements.append((re.compile(r'^[\s]*$'), 'drop'))

            dropped_any_lines = False
            with tempfile.SpooledTemporaryFile(max_size=2**24, mode='w+') as temp, open(fn) as fd:
                self.logger.debug("Pre-processing of '%s':" % fn)
                prev_line = ""
                for line in fd:
                    drop_line, log_line = True, True
                    for reg, rep in replacements:
                        if reg.search(line):
                            drop_line, log_line = False, False
                            if rep:
                                if rep == 'drop':
                                    self.logger.debug(f'* dropping "{line.strip()}"')
                                    drop_line, log_line = True, False
                                    break
                                if rep == 'append':
                                    self.logger.debug(f'* appending "{line.strip()}"')
                                    drop_line, log_line = True, False
                                    line = prev_line.rstrip() + ' ' + line.lstrip()  # strip the midpoint to help re match
                                else:
                                    self.logger.debug(f'* replacing "{line.strip()}" with "{rep}"')
                                    line = reg.sub(rep, line)
                                    drop_line = False
                    if drop_line:
                        if log_line:
                            if not dropped_any_lines:
                                dropped_any_lines = True
                                self.logger.info("Dropped unexpected lines from '%s':" % fn)
                            self.logger.info('- ' + line.strip())
                    else:
                        self.logger.debug('+ ' + line.strip())
                        temp.write(line)
                    prev_line = line
                if temp.tell() == 0:
                    if fd.tell() != 0:
                        self.logger.warning("Dropped all lines from '%s':" % fn)
                    else:
                        self.logger.warning("Input file is empty '%s':" % fn)
                    df = pd.DataFrame()
                else:
                    temp.seek(0)
                    df = self.read_fd(temp)
                    self.logger.debug('Raw data after pre-processing:\n'+str(df))
            #if dropped_any_lines: TODO: print once after all the read calls for each file
            #    self.logger.info('## End of dropped lines')
        else:
            with open(fn) as fd:
                try:
                    df = self.read_fd(fd)
                except Exception as e:
                    if fd.tell() == 0:
                        self.logger.warning("Input file is empty '%s':" % fn)
                    else:
                        self.logger.error("Error reading from input file '%s': " % fn, e.message)
                    df = pd.DataFrame()
                else:
                    self.logger.debug('Raw data:\n'+str(df))

        # Add file, directory, path...
        df['Path'] = fn
        df['File'] = os.path.basename(fn)
        df['Directory'] = os.path.dirname(fn) or '.'

        # Add metadata passed from the benchmark runner...
        if os.path.isfile(fn + '.meta'):
            try:
                with open(fn + '.meta') as fd:
                    meta_map = yaml.load(fd)

                df = df.assign(**meta_map)
            except Exception as e:
                self.logger.warning("Applying metadata from '%s' failed: \n%s" % (fn + '.meta', e))

        return df


    def read_fd(self, fd, **kwargs):
        read_funcs = {
                'csv': self.read_csv
        }
        return read_funcs[self['input']['format']](fd, **kwargs)


    def read_csv(self, fd, **kwargs):
        read_csv_params = dict(skipinitialspace=True)

        # Check if header is present
        if 'csv-header' in self['input']:
            header = self['input']['csv-header']
            line = fd.readline()
            if line[:-1].replace(' ', '') != header.replace(' ', ''):
                read_csv_params.update(dict(header=None, names=[x.strip() for x in header.split(',')]))
            fd.seek(0)

        return pd.read_csv(fd, **read_csv_params)

    def read_csv_cached(self, filepath_or_buffer, *args, **kwargs):

        csv_name = os.path.abspath(filepath_or_buffer) if os.path.isfile(filepath_or_buffer) else filepath_or_buffer
        if csv_name not in self.cached_csvs:
            self.cached_csvs[csv_name] = pd.read_csv(csv_name, *args, **kwargs)

        return self.cached_csvs[csv_name]

    def get_normalized_data(self, df=None, inputs=None, **kwargs):
        '''Get a pandas DataFrame containing normalized data for this benchmark.
        No aggregation is yet performed, only filtering.

        df: pd.DataFrame (default None)
            if None, data will come from self.get_raw_data.
            Otherwise, use the passed in dataframe as a starting point.

        inputs: list of str (default None)
            Only effective when df is None. When None, load input files using the
            config-defined glob. Otherwise, use the passed in array of strings
            as paths to csv inputs which are to be concatenated in pandas.
        '''

        if df is not None:
            df = df.copy()
        else:
            df = self.get_raw_data(inputs=inputs, **kwargs)

        # Rename columns with the given dict
        if self['rename'] is not None:
            df.columns = df.columns.map(lambda x: self['rename'][x] if x in self['rename'] else x)
        self.logger.debug('After renaming:\n'+str(df))

        # Filter out values
        if self['filter-out'] is not None:
            for col in self['filter-out']:
                df = df[~df[col].isin(self['filter-out'][col])]
        self.logger.debug('After filter-out:\n'+str(df))

        # Now that we're done filtering stuff, infer the best dtypes
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Compute precomputed columns
        if self['precomputed'] is not None:

            # Define built-in functions for use in these configurations
            def ratio_of(column, invert=False, **kwargs):
                '''Compute a column as percent of some series combination'''

                # Ensure that all value columns are float
                df[column] = df[column].astype('float64')

                # Group by series. These are the columns we are using as keys
                # to compare
                dfgby = groupby_empty(df, self['series'])

                # Turn the kwargs into a tuple...
                if len(self['series']) > 1:
                    series = tuple(kwargs[k] for k in self['series'])
                else:
                    series = kwargs[self['series'][0]]

                # Try to get the base values specified for this operation
                # e.g. native C would make this a "ratio of C" type operation
                try:
                    base = dfgby.get_group(series)
                except KeyError:
                    raise BenchmarkError('Trying to use series value %s in ratio_of operation, '
                                         'but that value/combination doesn\'t exist in the '
                                         'dataframe!' % (series,))

                # Indices here are now the axis+variants
                # Aggregate out anything which isn't specified in axis and variants.
                # base will be a dataframe with a MultiIndex of axis+variants, and
                # one column of values
                base = groupby_empty(base, self['axis'] + self['variants'])
                base = base[[column]].agg(self['aggregation'])

                # Initialize a list in which we can put dataframes of computed values
                values = []

                # For each series combination
                for name, group in dfgby:

                    # Do the same thing we did for base combination. Aggregate out
                    # anything which isn't in axis/variants.
                    group = groupby_empty(group, self['axis'] + self['variants'])
                    group = group[[column]].agg(self['aggregation'])

                    # Depending on if we want base as numerator or denominator, calculate...
                    if invert:
                        group[column] = base[column] / group[column]
                    else:
                        group[column] = group[column] / base[column]

                    # Bring in the series combination as columns
                    if len(self['series']) > 1:
                        for i, key in enumerate(self['series']):
                            group[key] = name[i]
                    else:
                        group[self['series'][0]] = name

                    # Append to the list of dataframes this particular dataframe,
                    # bringing in any series+axis+variants values which went into the
                    # index as columns, as preparation for a merge into the original df.
                    # If there were no axis/variants columns to group by, just drop
                    # the groupby index since it was some dummy column anyway. Otherwise,
                    # bring it back in as columns.
                    if len(self['axis']+self['variants']) == 0:
                        group = group.reset_index(drop=True)
                    else:
                        group = group.reset_index()
                    values.append(group)

                # Concatenate all the computed values, ignoring their indices.
                # Since we did reset_index() to bring in series+axis+variants values
                # into the dataframe as columns, the indices are meaningless and
                # not ignoring them would cause issues with duplicate indices.
                values = pd.concat(values, ignore_index=True)

                # Bring in the original dataframe, drop the column we were operating on
                # and bring its index as a column. This last step is very important so
                # we can retain the indices of the original dataframe and simply return
                # a Series with the corresponding indices. This also means this becomes
                # a composable operation with other 1D vector operations!
                tomerge = df.drop(column, axis=1, errors='ignore').reset_index()

                # Merge in our values by series+axis+variants columns, keeping everything
                # in the original dataframe (LEFT OUTER JOIN). This obliterates the indices
                # of the dataframe, which is why we had to reset_index() beforehand.
                merged = pd.merge(tomerge, values, how='left', on=(self['series']+self['axis']+self['variants']))

                # Set the index of the merged dataframe to the indices we brought in
                # earlier as a column, then return only the column of computed values.
                return merged.set_index('index')[column]


            def ratio_diff(column, invert=False, **kwargs):
                '''Compute a column as a percent difference from some series combination'''
                result = ratio_of(column, invert=invert, **kwargs)
                return result - 1


            def ratio_of_inv(column, **kwargs):
                return ratio_of(column, invert=True, **kwargs)


            def ratio_diff_inv(column, **kwargs):
                return ratio_diff(column, invert=True, **kwargs)


            def percent_of(column, invert=False, **kwargs):
                result = ratio_of(column, invert=invert, **kwargs)
                result = result * 100
                return result


            def percent_diff(column, invert=False, **kwargs):
                result = ratio_diff(column, invert=invert, **kwargs)
                result = result * 100
                return result


            def percent_of_inv(column, **kwargs):
                return percent_of(column, invert=True, **kwargs)


            def percent_diff_inv(column, **kwargs):
                return percent_diff(column, invert=True, **kwargs)

            # For each column to compute... (We can also overwrite columns!)
            for col in self['precomputed']:

                func = self['precomputed'][col]
                eval_globals = dict(locals())
                eval_globals.update({"pd": pd, "read_csv": self.read_csv_cached})

                try:
                    if 'row[' in func:
                        # Execute this as a row function
                        # Evaluate the lambda function on the df, passing in our locals
                        result = df.apply(eval('lambda row:'+func, eval_globals), axis=1)
                    else:
                        # Execute this as a dataframe function
                        result = eval('lambda df: '+func, eval_globals)(df)
                except KeyError as e:
                    raise BenchmarkError('Row or column index "%s" specified for precomputed '
                                         'columns not found' % (e.args[0],))

                # If we got a dataframe out, merge it in (this means we did some advanced
                # operation like a percent of). We must do this because the indices are
                # almost certainly not the same.
                # Otherwise, just set it equal, assuming the indices are the same
                if isinstance(result, tuple):
                    result, origin = result
                    result = result.rename(columns={origin: col})
                    df = pd.merge(result, df.drop(col, axis=1, errors='ignore'), how='left')
                else:
                    df[col] = result
        self.logger.debug('After column precompute:\n'+str(df))


        # Perform packing/unpacking of values
        if self['pack'] is not None:
            for packconf in self['pack']:
                self.logger.debug('Packing (melting) columns '+
                                  (', '.join(packconf['columns'])) + ' into '
                                  'name: "%s", value: "%s".' % (packconf['name'], packconf['value']))
                df = pd.melt(df, id_vars=df.columns[~df.columns.isin(packconf['columns'])],
                             var_name=packconf['name'], value_name=packconf['value'])

        if self['unpack'] is not None:
            for packconf in self['unpack']:

                self.logger.debug('Unpacking (pivoting) columns from '
                                  'name: "%s", value: "%s".' % (packconf['name'], packconf['value']))

                index = df.columns[~df.columns.isin([packconf['name'], packconf['value']])].tolist()

                # Pivot the column over, just 1:1'ing all values
                # Incorporates an *awful* hack of replacing NaN values with
                # the string 'nan' in order to allow those NaN-valued groups to continue
                # through aggregation. Right after we are done with unpacking stuff,
                # we will infer dtypes again for these columns, trying to promote to
                # numeric types where possible. This will automatically convert the 'nan'
                # values to the actual floating-point NaN value.
                df = df.fillna('NaN').groupby(index+[packconf['name']]).first().unstack()

                # Rename column index so we can reset index
                df.columns = df.columns.droplevel().rename(None)

                # Reset index
                df = df.reset_index()
        self.logger.debug('After pack/unpack:\n'+str(df))

        # Now that we're done unpacking stuff, infer the best dtypes
        df = df.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Filter values
        if self['filter-in'] is not None:
            for col in self['filter-in']:
                df = df[df[col].isin(self['filter-in'][col])]

        if len(df) == 0:
            self.logger.warning('Filtering on input data has removed all data. Expect empty output.')
        self.logger.debug('After filter-in:\n'+str(df))

        # Make sure we have necessary columns
        for col in self['axis'] + self['series'] + self['variants']:
            if not col in df.columns:
                raise BenchmarkError('Column "%s" specified in config but not found' % col)

        # Now that we're done filtering stuff, infer the best dtypes
        df = df.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Get value columns
        values = self['values']

        # Make sure we have all value columns
        missing_columns = []
        for col in values:
            if col not in df.columns:
                missing_columns.append(col)
        if len(missing_columns) > 0:
            raise BenchmarkError("Missing values columns: " + (', '.join(missing_columns)))

        # We expect numbers in value columns
        try:
            df = df.astype(dict((col, 'float64') for col in values))
        except:
            print(df.head(10))
            raise BenchmarkError('Found non-numerical data in values columns!')

        return df


    def get_aggregated_data(self, df=None, inputs=None, **kwargs):
        '''Get a pandas DataFrame containing aggregated data for this benchmark.

        df: pd.DataFrame (default None)
            if None, data will come from self.get_normalized_data.
            Otherwise, use the passed in dataframe as a starting point.

        inputs: list of str (default None)
            Only effective when df is None. When None, load input files using the
            config-defined glob. Otherwise, use the passed in array of strings
            as paths to csv inputs which are to be concatenated in pandas.
        '''

        if df is not None:
            df = df.copy()
        else:
            df = self.get_normalized_data(inputs=inputs, **kwargs)

        # Having too many expected prefix warnings is just clutter
        expected_prefix_warning_count = 0
        expected_performance_count = 0

        toconcat = []

        # For each combination of these columns...
        axis = self['axis']
        series = self['series']
        values = self['values']
        variants = self['variants']

        # Check if we need to check expected series values.
        if self['expected'] is not None:
            expect_impl = pd.DataFrame([tuple(x) for x in self['expected']], columns=series)
        else:
            expect_impl = None

        for name, group in groupby_empty(df, axis+variants):

            group = group.copy()

            # If we need to check expected series values:
            if expect_impl is not None:

                # Check if we got all the series values we wanted...
                # Get the series values we have (as a dataframe)
                present = group[series].drop_duplicates()

                # Get the series values we don't have (as a dataframe).
                # We concatenate the 'present' dataframe with the expected implentations dataframe,
                # and then remove any instances of series values appearing twice (which means they
                # were found in both the present and expected series values dataframes.)
                needed = pd.concat([expect_impl, present], ignore_index=True).drop_duplicates(keep=False)

                # We now limit the needed dataframe to contain only expected implementations.
                # If we didn't do this step, it would also contain extra implementations that we
                # have but haven't specified in the expected list
                needed = pd.merge(needed, expect_impl, how='inner')

                # If we didn't get something we needed, emit a warning and fill this space with NaNs.
                if len(needed) > 0:

                    # String describing where the data was missing and what data was missing
                    logger_string = 'data missing for %s' % repr(name) + \
                                    '\t(present: %s, needed: %s)' % \
                                        (', '.join([repr(tuple(x)) for x in present.values]), 
                                            ', '.join([repr(tuple(x)) for x in needed.values]))

                    # Only print one warning, unless we are asking for verbose output.
                    if expected_prefix_warning_count == 0:
                        self.logger.warning(logger_string)
                    else:
                        self.logger.debug(logger_string)

                    expected_prefix_warning_count += 1

                    # Deal with these missing series values. Add one row per missing value, setting
                    # any value/unspecified columns to NaN/empty.
                    minitoconcat = [group]
                    for prefix in needed.values:
                        prefixdf = pd.DataFrame(dict(zip(axis+variants+series, name + tuple(prefix))), index=[0])
                        minitoconcat.append(prefixdf)

                    # Concatenate the group with these extra rows before proceeding.
                    group = pd.concat(minitoconcat, ignore_index=True)

            # Check if we have an expected ordering of values
            if self['expected'] is not None:
                config_expected = self['expected']
                # Display warnings if we got unexpected performance differences
                for value in values:

                    # Perform aggregation for each implementation just for this comparison
                    actual_ordering = groupby_empty(group, series, as_index=False).agg(self['aggregation'])

                    # Sort such that the aggregated dataframe has the worst performers at lower indices
                    # and best performers at higher indices
                    actual_ordering = actual_ordering.sort_values(value, ascending=self['higher-is-better'])

                    # Take the first appearance of each series combination.
                    actual_ordering = actual_ordering[series].drop_duplicates(series).values

                    # Remove missing prefixes from actual and expected orderings
                    # These list comprehensions preserve the order of the original lists.
                    needed = needed.values
                    actual_ordering = [tuple(x) for x in actual_ordering if tuple(x) not in needed]
                    expected_ordering = [tuple(x) for x in config_expected if tuple(x) not in needed]

                    if actual_ordering != expected_ordering:
                        logger_string = ('Unexpected performance ordering for value %s in '
                                        'series combination %s (from slowest to fastest): '
                                        'expected %s but got %s') % (value, name, expected_ordering, actual_ordering)
                        if expected_performance_count == 0:
                            self.logger.warning(logger_string)
                        else:
                            self.logger.debug(logger_string)
                        expected_performance_count += 1

            toconcat.append(group)

        if expected_prefix_warning_count > 1:
            self.logger.warning('%d warnings about missing data' % (expected_prefix_warning_count))

        if expected_performance_count > 1:
            self.logger.warning('%d warnings about performance expectations' % (expected_performance_count))

        try:
            df = pd.concat(toconcat, ignore_index=True)
        except ValueError:
            pass

        # Actually perform aggregation
        othercols = df.columns[~df.columns.isin(series+axis+variants+values)]
        aggby = {v: self['aggregation'] for v in values}
        aggby.update({v: 'first' for v in othercols})

        # Show columns aggregated out
        dfuniq = df[othercols].nunique().to_dict()
        for col in dfuniq:
            self.logger.debug('Aggregated out column "{col}" with {num} unique values'.format(col=col, num=dfuniq[col]))
        df = groupby_empty(df, series + axis + variants).agg(aggby).reset_index()

        return df[df.columns[~df.columns.str.startswith('_')]]


    def format_column(self, col, boundaries, colors):
        '''Color values in the given column (as Series),
        using the given colors, divided at the given boundaries.

        Value boundaries are sorted before use.'''
        boundaries = list(sorted(boundaries))
        if len(boundaries) != len(colors) - 1:
            raise ValueError('Number of boundaries must be one less than number of colors!')

        indices = col.copy()
        indices[:] = 0
        for b in boundaries:
            indices += (col > b)
        indices[col.isnull()] = -1
        indices = indices.astype('int64')
        
        return ['background-color: ' + (colors[i] if i != -1 else '#ffffff') for i in indices]

    def create_html_pivot_table(self, df, f, plot=False):
        '''Return a pivot table created from df, and
        outputs it with conditional formatting to the given
        open HTML document
        '''

        for i, (variant, group) in enumerate(groupby_empty(df, self['variants'])):

            if type(variant) is not tuple:
                variant = (variant,)

            f.write('<hr>')
            f.write('<h3>Variant {}: {}</h3>\n'.format(i+1, ', '.join(str(v) for v in variant)))

            for k, v in zip(self['variants'], variant):
                f.write('<p>{} = <b>{}</b></p>\n'.format(k, v))

            f.write('<br>\n')

            pt = group.pivot_table(values=self['values'],
                                   index=self['axis'],
                                   columns=self['series'],
                                   aggfunc=self['aggregation'])

            pt.to_html(buf=f)
            f.write('<br>\n')

            if plot:
                import matplotlib.pyplot as plt
                import base64
                from io import BytesIO
                fig, ax = plt.subplots()
                pt.plot(kind='bar', ax=ax)
                plt.ylabel(self['values'])
                plt.tight_layout()

                # Save the plot to bytesIO which we can write as base64 into html
                with BytesIO() as buf:
                    fig.savefig(buf, format='png')
                    b64 = base64.b64encode(buf.getbuffer()).decode()
                    f.write('<img src="data:image/png;base64,{}" /><br>\n'.format(b64))


    def create_pandas_pivot_table(self, df, excel=None, raw=False):
        '''Return a pivot table created from df, and
        outputs it with conditional formatting to the given
        Excel spreadsheet, if not None.
        '''

        if excel is not None:
            writer = pd.ExcelWriter(excel)

            position = 0

            for variant, group in groupby_empty(df, self['variants']):

                if type(variant) is not tuple:
                    variant = (variant,)

                varindex = pd.DataFrame(dict(zip(self['variants'], variant)), index=[0])
                varindex.to_excel(writer, 'summary', startrow=position, index=False)
                position += len(varindex) + 2

                pt = group.pivot_table(values=self['values'],
                                       index=self['axis'],
                                       columns=self['series'],
                                       aggfunc=self['aggregation'])

                if pt.size == 0:
                    self.logger.warning('Skipping pivot table of length zero, for '
                                        'variant {}'.format(variant))
                    continue

                if self['indicator'] is not None:
                    for col in self['indicator']:
                        ss = pt.style.apply(self.format_column, boundaries=col['ranges'], colors=col['colors'], subset=col['column'])
                else:
                    ss = pt

                ss.to_excel(writer, sheet_name='summary', startrow=position)
                position += len(pt) + df.columns.nlevels + len(self['values']) + 4

            df.to_excel(writer, sheet_name='data')
            if raw:
              rdf=pd.concat(self.dataframes, ignore_index=True, sort=True)
              rdf.to_excel(writer, sheet_name='raw')
            
            writer.save()


    def create_excel_pivot_table(self, df, outfile):
        import excel_pivot

        pivot_formatter = {}

        pivot_table(df, outfile,
                    values=self['values'], columns=self['series'],
                    index=self['axis'], formatter=pivot_formatter,
                    filters=self['variants'], value_format='0.000',
                    aggfunc=self['aggregation'],
                    show_excel=True)


    def pivot_string(self, df):

        # If there are no axis, series or variants, just aggregate
        # everything for each value column

        ret = ''

        # If the user hasn't defined any variants, make this
        # data structure which mimics the output of the
        # enumerate function. Otherwise actually group by
        # the variants and then enumerate the output of that.
        dfgby = groupby_empty(df, self['variants'])

        for i, (variant, group) in enumerate(dfgby):

            if type(variant) is not tuple:
                variant = (variant,)

            ret += ('Pivot table for variant %d:\n' % (i+1))
            for (k, v) in zip(self['variants'], variant):
                ret += '%s = %s\n' % (k, v)

            ret += '\n'
            if len(self['axis'] + self['series']) == 0:
                pt = group[self['values']].agg(self['aggregation'])
            else:
                pt = group.pivot_table(values=self['values'], index=self['axis'],
                                       columns=self['series'], aggfunc=self['aggregation'])

            if type(pt) is pd.Series:
                pt = pd.DataFrame(pt)

            # Format the numbers inside the pivot table.
            # If we got a format string in the config, use it.
            # Otherwise, assume we got a number of some sort, which tells
            # the number of digits of precision we want to keep in the value.
            if isinstance(self['number-format'], str):
                pt = pt.applymap(lambda x: self['number-format'].format(x))
            else:

                precision = self['number-format']
                log10 = np.log(10)
                def get_precision(num):
                    order = np.log(num) / log10
                    try:
                        return max(precision - int(order), 0)
                    except OverflowError:
                        return 0

                def apply_precision(num, decimals=None):
                    try:
                        num = float(num)
                    except ValueError:
                        return num

                    # If we got NaN
                    if num != num:
                        return num

                    if decimals is None:
                        decimals = get_precision(num)

                    return ('{:.' + str(decimals) + 'f}').format(num)

                if self['number-format-max-only']:
                    max_decimals = get_precision(np.nanmax(pt.values.flatten()))
                    pt = pt.applymap(lambda x: apply_precision(x, decimals=max_decimals))
                else:
                    pt = pt.applymap(apply_precision)

            ret += str(pt) + '\n\n\n'

        return ret


def main():

    parser = argparse.ArgumentParser(description='aggregate benchmarking results')
    parser.add_argument('--verbose', '-v', default=0, action='count', help='debug logging')
    parser.add_argument('--input', '-i', default=None, nargs='+',
                        help='input files. If specified, the input file glob in the config is ignored.')
    parser.add_argument('config', nargs='+', help='configuration file in YAML format')
    parser.add_argument('--excel-pivot-table', '-p', default='pandas', 
                        help='When outputting to an Excel spreadsheet, '
                             'use the specified style of generating a pivot table.\n'
                             'When not specified, output the data only to the Excel spreadsheet.\n'
                             'Has no effect when Excel output is disabled.\n\n'
                             'Choices:\n'
                             'pandas: output a "pandas-style" pivot table, which is non-'
                             'interactive.\n'
                             'excel: output a native Excel pivot table, which is interactive '
                             'and has drilldown functionality in Excel.',
                        choices=['pandas', 'excel'])
    parser.add_argument('--excel', '-x', default=None, const='{filename}.xlsx',
                        action='store', nargs='?', help='Output to this Excel file')
    parser.add_argument('--csv', '-o', default=None, const='{filename}.csv', action='store', nargs='?',
                        help='CSV file to output to, or "-" for stdout')
    parser.add_argument('--pretty-print', '-P', default=None, const='-', action='store', nargs='?',
                        help='Pretty-print pivot tables')
    parser.add_argument('--html', '-H', default=None, const='{filename}.html', action='store', nargs='?',
                        help='Output tables to HTML with pd.DataFrame.to_html')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='Add plots to HTML')
    parser.add_argument('--raw', default=False, action='store_true', help='Add excel sheet with raw data. '
                                                                      'Ignored if --excel is not specified')    

    args = parser.parse_args()

    # Set up logger
    logger = logging.getLogger('benchmark')
    logger.setLevel(20 - args.verbose*10)

    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setLevel(20 - args.verbose*10)
    log_formatter = logging.Formatter(fmt='[%(levelname)s] %(message)s')
    log_handler.setFormatter(log_formatter)
    log_handler.flush = sys.stdout.flush
    logger.addHandler(log_handler)

    for path in args.config:

        conf_name = os.path.splitext(os.path.basename(path))[0]

        try:
            bench = Benchmark(path)
            df = bench.get_aggregated_data(inputs=args.input)
        except BenchmarkError as e:
            logger.error(str(e))
            sys.exit(1)

        if args.csv is not None:
            if args.csv == '-':
                csv_out = sys.stdout
            else:
                csv_out = args.csv.format(filename=conf_name)

            df.to_csv(csv_out, float_format='%.3f', index=False)

        if args.pretty_print:
            pd.options.display.max_rows = None
            if args.pretty_print == '-':
                print(bench.pivot_string(df))
            else:
                with open(args.pretty_print, 'w') as f:
                    f.write(bench.pivot_string(df))

        if args.html:
            def write_html(f):
                header = pkg_resources.resource_string(__name__,
                                                       'html/header.html')
                header = header.decode()
                for line in header:
                    f.write(line)
                f.write('<h1><code>{}</code> Performance Results</h1>\n'.format(conf_name))
                bench.create_html_pivot_table(df, f, plot=args.plot)
                footer = pkg_resources.resource_string(__name__,
                                                       'html/footer.html')
                footer = footer.decode()
                for line in footer:
                    f.write(line)

            with open(args.html.format(filename=conf_name), 'w') as f:
                write_html(f)

        if args.excel_pivot_table is not None:
            if args.excel is not None:
                if args.excel_pivot_table == 'pandas':
                    bench.create_pandas_pivot_table(df, args.excel, raw=args.raw)
                elif args.excel_pivot_table == 'excel':
                    bench.create_excel_pivot_table(df, args.excel)
        elif args.excel is not None:
            df.to_excel(args.excel.format(filename=conf_name), index=False)


if __name__ == '__main__':
    import argparse
    main()
