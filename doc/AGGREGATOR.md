# Aggregator

The aggregator provides generic data processing and presentation facilities, originally
designed to provide reports with comparisons of benchmark measurements between
different implementations of algorithms.

The current aggregator simply executes pre-defined steps in a pre-defined order.
It operates on **aggregator recipes**, which are configurations in YAML format
specifying the input paths and formats, processing steps, and output options.

## Usage
```
usage: aggregate.py [-h] [--verbose] [--input INPUT [INPUT ...]] [--excel-pivot-table {pandas,excel}] [--excel [EXCEL]] [--csv [CSV]] [--pretty-print [PRETTY_PRINT]] [--html [HTML]] [--plot] config [config ...]

aggregate benchmarking results

positional arguments:
  config                configuration file in YAML format

optional arguments:
  -h, --help            show this help message and exit
  --verbose, -v         verbosity, -vv for even more
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        input files. If specified, the input file glob in the config is ignored.
  --excel-pivot-table {pandas,excel}, -p {pandas,excel}
                        When outputting to an Excel spreadsheet, use the specified style of generating a pivot table. When not specified, output the data only to the Excel spreadsheet. Has no effect when Excel
                        output is disabled. Choices: pandas: output a "pandas-style" pivot table, which is non-interactive. excel: output a native Excel pivot table, which is interactive and has drilldown
                        functionality in Excel.
  --excel [EXCEL], -x [EXCEL]
                        Output to this Excel file
  --csv [CSV], -o [CSV]
                        CSV file to output to, or "-" for stdout
  --pretty-print [PRETTY_PRINT], -P [PRETTY_PRINT]
                        Pretty-print pivot tables
  --html [HTML], -H [HTML]
                        Output tables to HTML with pd.DataFrame.to_html
  --plot                Add plots to HTML
```

## Configuration

### Input
The aggregator currently supports two types of input: benchmark outputs (csv)
and configuration files. Config input is specified by writing a list of
configurations to run before this one under `config` in the `input` section.
Benchmark outputs are specified by writing a list of globs (see example)
under `path` in the `input` section.

In case benchmark outputs have no header (column names), the `csv-header`
option allows adding a header before passing to `pandas.read_csv`.

Line-by-line filtering is also available for result files that contain
errors or comments. Each (key, value) pair under `filter` in the `input`
section is treated as (regex, replacement). A replacement of null (empty)
means to do nothing but mark for inclusion. A replacement of `drop` means
to actually drop the line and do not report that. A replacement of `append`
concats current line with previous line enabling regex to match across lines.
If any filter lines exist, completely unmatched lines will be dropped with
logging. If the filter section is omitted, no filtering is performed.

Once the input section is finished, each parsed file will be converted to
a pandas dataframe (using `pd.read_csv`) for further processing

### Fixed pipeline processing steps
Currently, the aggregator always processes the data in the following order.
- Rename columns using the `rename` section (a dict of column names ->
  new column names). Each (key, value) under the
  `rename` section is used to rename a column called `key` with a column
  called `value`.
- Filter out data. For each key in the `filter-out` section (a dict of column
  names -> lists of values to filter out), remove
  any rows where the value in the column with the name of the key is
  included in the list.
- Precompute columns in the `precompute` section (a dict of column name
  destinations -> string representation of Python lambda functions).
  This allows the user to specify Python lambda functions
  in the configuration, which produce output to be assigned to the column
  specified. If the function references `row`, then the function will be
  applied row-by-row. If the function references `df`, then the function
  will be passed the entire dataframe at once.
  - The precompute section provides special functions in the `ratio_of`
    family (see [here](https://github.com/IntelPython/bearysta/blob/master/bearysta/aggregate.py#L407)
    for the full listing). These only apply to `df` and reference the
    `series` section (covered later) directly, as their main purpose is to
    find the ratio between all combinations of series values and some
    fixed "reference" set of data. This "reference" set is specified
    with the keyword arguments of `ratio_of`. See the `indicators`
    examples for more details.
- "Pack" columns (in the `pack` section, which is a list of dicts
  with keys `name`, `value`, and `columns`). This basically pivots
  some columns to extra rows. For example, if a benchmark has a column
  for fit time and a column for predict time, but it would be instead
  better to have a column for "function" and a column for "time", then
  a user could specify the name `function`, value `time`, and columns
  `[fit time, predict time]`.
- "Unpack" columns (in the `unpack` section, which is a list of dicts
  with keys `name` and `value`). This is basically the opposite of
  `pack`: for each value in the `name` column, it creates a new column
  in the resultant dataframe with the value from the `value` column.
- Filter in data (in the `filter-in` section). This is the same as
  filter-out, except that lines that do not match are
  removed instead.
- Create any columns in `series`, `axis`, `variants` which do not yet
  exist (not configurable)


Finally, at the end, the aggregator creates multiple pivot tables for each
value of the column(s) specified in `variants`, with
a MultiIndex column index (which is generated by taking all combinations
of the columns specified in `series`), a MultiIndex row index
(which is generated by taking all combinations of the columns specified
in `axis`). Generally, the `series` is used for comparisons between
different environments, implementations, or something else, the
`axis` is used for showing different benchmarks or parameters to benchmarks,
and the `variants` is used for showing different machines or parameters
that should result in completely different tables.

### Output
The aggregator supports CSV, pretty-print, HTML (with plotting), and Excel
formats. See command-line help for details.
  

### Example
```yaml
input:
    # If this key exists, run the specified configurations first and 
    # use their outputs as the input to this configuration.
    config:
    - path/to/config1.yaml
    - path/to/config2.yaml
    
    # File names which should be inputs to this benchmark.
    # Unix-style globbing is supported.
    path:
       - 'runs/*/sklearn_native/*/*_distances*'
       - 'runs/*/sklearn_native/*/*_ridge*'
       - 'runs/*/sklearn_native/*/*_linear*'
       - 'runs/*/sklearn_native/*/*_kmeans*'
    format: csv # only csv is supported for now
    # If the files don't have a CSV header, use this as the header.
    csv-header: 'Batch,Arch,Prefix,Threads,Size,Function,Time'
    
    # Line-by-line filtering.
    # - if no filter is specified, use all lines
    # - if filters are specified, for each (key, value) in this dict,
    #   perform replacement using key as regex and value as repl,
    #   or drop the line if "drop" is specified as the replacement.
    # - All unmatched lines will also be dropped and logged.
    filter:
        "@ Package 'daal4py' was not found. Number of threads is being ignored": drop
        "WARNING: Number of actual iterations.*": drop
        "Tolerance: .*": drop
        '':


# Aggregation method (e.g. min, median, max, mean)
aggregation: median

# Axis and series column names (analogous to Excel)
# For us, axis columns usually represent different parameters to benchmarks which we do
# not compare to each other.
axis:
    - Function
    - Size

# For us, series columns usually represent different parameters or implementations of
# benchmarks which we explicitly compare to each other.
series:
    - Prefix

# Output a different table for each combination of the given columns in variants.
variants:
    - Arch
    - Mode

# Values columns are the columns that actually contain the real values which are to
# be aggregated or tabulated.
values:
    - Time

# Are higher values better? (obsolete)
higher-is-better: false

# Precompute columns using lambda functions
precomputed:
    #Function: "row['Function'].capitalize()"
    Mode: "'Serial' if row['Threads'] == 1 else 'Parallel'"
    Arch: "(row['Directory'].split('/')[-3].split('_')[-3:-2]+['Unknown'])[0]"

# Filter: require certain values for columns (after precompute and rename)
# any rows not matching these filters will be dropped.
filter-in:
    Mode:
    - Parallel
    Size:
#    - 524288
#    - 1048576
    - 1000000x50

# Filter out: require columns to NOT be certain values (after filter-keep)
# any rows matching these filters will be dropped.
filter-out:
    #Prefix: [Prefix]
    #Implementation: [VML, Numpy]

# Number format in Python str.format() style, or the number of digits of
# precision to show in numbers in pretty-printed pivot tables
number-format: 2

# Do we figure out the number of decimals only once, using the max value,
# and disregard smaller precision numbers?
number-format-max-only: false
```
