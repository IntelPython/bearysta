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


### Example
```yaml
# File names which should be inputs to this benchmark.
# Unix-style globbing is supported.
input:
    path:
       - 'runs/*/sklearn_native/*/*_distances*'
       - 'runs/*/sklearn_native/*/*_ridge*'
       - 'runs/*/sklearn_native/*/*_linear*'
       - 'runs/*/sklearn_native/*/*_kmeans*'
    format: csv # only csv is supported for now
    # If the files don't have a CSV header, use this as the header.
    csv-header: 'Batch,Arch,Prefix,Threads,Size,Function,Time'
    # Line-by-line filtering.
    # This supports regex/replacement. The replacement "drop" indicates that the line
    # should be dropped entirely. An empty replacement indicates that the line should
    # be kept as-is. If this section is omitted, no filtering is performed.
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

# Number format in Python str.format() style, or the number of digits of
# precision to show in numbers in pretty-printed pivot tables
number-format: 2

# Do we figure out the number of decimals only once, using the max value,
# and disregard smaller precision numbers?
number-format-max-only: false
```
