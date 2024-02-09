#################################
# pandas_writer.py
# Author: Juha-Matti Rouvinen
# Date: 2022-02-09
# Updated: 2024-02-09
# Version V1
##################################

# Imports
from pandas import DataFrame


def df_to_excel(data,header,filepath):

    if len(data) == 8:
        df = DataFrame({header[0]: data[0], header[1]: data[1],header[2]: data[2],header[3]: data[3],
                        header[4]: data[4],header[5]: data[5],header[6]: data[6],header[7]: data[7]
                        })
        try:
            df.to_excel(filepath, sheet_name='training_plots', index=False)
        except PermissionError:
            pass

    elif len(data) == 7:
        df = DataFrame({header[0]: data[0], header[1]: data[1], header[2]: data[2], header[3]: data[3],
                        header[4]: data[4],header[5]: data[5],header[6]: data[6]
                        })
        try:
            df.to_excel(filepath, sheet_name='evaluation_plots', index=False)
        except PermissionError:
            pass

    else:
        pass