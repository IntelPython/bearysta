import win32com.client as win32
from pythoncom import com_error
import os
#import comtypes, comtypes.client
import pandas as pd
import logging


# Define a mapping of aggfunc strings to their
# Excel constant counterparts.
def get_win32_aggfunc_mapping(): 
    return {
            'mean': win32.constants.xlAverage,
            'average': win32.constants.xlAverage,
            'count': win32.constants.xlCount,
            'count_nums': win32.constants.xlCountNums,
            'max': win32.constants.xlMax,
            'min': win32.constants.xlMin,
            'product': win32.constants.xlProduct,
            'stdev': win32.constants.xlStDev,
            'stdev_p': win32.constants.xlStDevP,
            'sum': win32.constants.xlSum,
            'var': win32.constants.xlVar,
            'var_p': win32.constants.xlVarP
    }


def pivot_table(data, outfile, values, index, columns, filters=[],
                formatter={}, show_excel=False, value_format='0.00%',
                aggfunc='mean'):
    '''Create a native Excel pivot table.

    This function is intended to be similar to 
    pandas.pivot_table() but makes a native Excel
    pivot table instead.

    ARGUMENTS:

    data: pandas.DataFrame
        the data to write to the Excel spreadsheet
    outfile: str
        the path to the Excel spreadsheet.
        ANYTHING EXISTING IN THIS PATH WILL BE DELETED.
    values: str or list of str
        column names to use as values
    index: str or list of str
        column names to use as indices (rows)
    columns: str or list of str
        column names to use as columns
    filters: str or list of str
        column names to use as filters, can take a cross-
        section of the data
    formatter: function: float -> int
        a function taking in floating point numbers
        and spitting out the color as an integer
        in the format:
            RED: bits 7:0
            GREEN: bits 15:8
            BLUE: bits 23:16
    show_excel: boolean
        if True, show the opened Excel
        application window. If False, keep it hidden
        and close it after completion.
    value_format: str
        the Excel format string used to format
        values in the Excel table.
    aggfunc: str
        the aggregation function Excel should use

    NOTES:

    Note that because this uses pywin32 COM APIs,
    this function will ONLY RUN ON WINDOWS.
    This unfortunately seems like the only way to 
    create a native Excel pivot table directly from
    Python. (At least we don't need to write VBA)
    '''

    logger = logging.getLogger('benchmark')

    # Before we begin, make sure values, index, columns 
    # are iterables.
    try:
        _ = iter(values)
    except TypeError:
        values = [values]
    try:
        _ = iter(index)
    except TypeError:
        index = [index]
    try:
        _ = iter(columns)
    except TypeError:
        columns = [columns]
    try:
        _ = iter(filters)
    except TypeError:
        filters = [filters]

    
    # Save to Excel spreadsheet
    writer = pd.ExcelWriter(outfile)
    data.to_excel(writer, sheet_name='data')

    try:
        writer.save()
    except PermissionError as e:
        logger.critical("Couldn't write {}! "
                        "Try closing anything using that file."
                        " (This is probably Excel)".format(outfile))
        raise


    # The following code is a rough translation of VBA code for
    # the same purpose.
    
    # Start up Excel
    excel = win32.gencache.EnsureDispatch('Excel.Application')
    excel.Visible = show_excel

    # Open worksheet
    workbook = excel.Workbooks.Open(os.path.abspath(outfile))
    if workbook is None:
        raise RuntimeError("Couldn't open workbook {}! Try "
                           "closing Excel if you have it open."
                           "".format(os.path.abspath(outfile)))
    sheet = workbook.Sheets('data')

    # Select worksheet (move the GUI to this sheet basically)
    sheet.Select()
    # Select cell A1 in that worksheet
    sheet.Range('A1').Select()

    # Get PivotTableWizard (opens up the GUI for PivotTable)
    otable = sheet.PivotTableWizard()

    # Rename new sheet
    workbook.ActiveSheet.Name = 'pivot_table'

    # Disable automatic updating
    otable.ManualUpdate = True

    # In order for data to show in columns, it is required to
    # add this "Data" field
    otable.AddFields(RowFields=index, ColumnFields=["Data"] + columns, PageFields=filters)

    # Figure out the win32 constant corresponding to the
    # aggregation function we are using.
    win32_aggfunc_mapping = get_win32_aggfunc_mapping()
    if aggfunc in win32_aggfunc_mapping:
        aggfunc = win32_aggfunc_mapping[aggfunc]
    else:
        raise ValueError('Unsupported Excel aggregation function %s' % aggfunc)

    # Each value
    for i in values:
        ofield = otable.PivotFields(i)
        ofield.Orientation = win32.constants.xlDataField
        ofield.NumberFormat = value_format
        # ofield.Function = win32.constants.xlAverage
        ofield.Function = aggfunc


    # Disable subtotals...
    for field in otable.PivotFields():
        try:
            st = list(field.Subtotals)
            st[0] = False
            field.Subtotals = tuple(st)
        except com_error:
            pass

    # Disable grand totals
    otable.ColumnGrand = False
    otable.RowGrand = False

    # Manually do conditional formatting with the
    # passed in dict
    for i, data_field in enumerate(values):
        # The PivotItems are 1-indexed, so we add 1 to our Python 0-indices
        if not data_field in formatter:
            continue

        for cell in otable.PivotFields('Data').PivotItems(i+1).DataRange.Cells:
            cell.Interior.Color = formatter[data_field](cell.Value)

    # Re-enable automatic updating
    otable.ManualUpdate = False

    # Save workbook
    workbook.Save()
    logger.info('Saved pivot table.')

    if not show_excel:
        workbook.Close()
    

def get_column_formatter(bounds, colors):
    '''Get a function which returns the 
    appropriate color for a data point,
    given bounds and colors.

    ARGUMENTS:
        bounds: list of float
            the boundaries between each given color.
            This function will sort the boundaries,
            but not the colors.
        colors: list of int
            colors in format of 0xRRGGBB

    EXAMPLE:

        get_column_formatter([0.2, 0.6], [0xFF0000, 0xFFFF00, 0x00FF00])

        This returns a function which will output 0x0000FF for any input
        between -infinity and 0.2, 0x00FFFF for any input from 0.2 to until
        0.6, and 0x00FF00 for any input from 0.6 to infinity.

        (Note that the colors in the output are in 0xBBGGRR rather than
        0xRRGGBB because that's the way the Excel API requires colors be
        sent.)
    '''

    if len(bounds) != len(colors) - 1:
        raise ValueError('Number of boundaries must be one less than number of colors!')
    bounds = list(sorted(bounds))

    for i, c in enumerate(colors):
        # colors are weirdly reversed in Excel as BGR
        c = (c & 0xff, (c>>8) & 0xff, (c>>16) & 0xff)
        colors[i] = c[0] << 16 | c[1] << 8 | c[2]

    def formatter(data):

        for i, b in enumerate(bounds):
            if data is None or b is None:
                return 0xffffff
            if data < b:
                return colors[i]
        return colors[len(bounds)]

    return formatter
