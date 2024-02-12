#################################
# pandas_writer.py
# Author: Juha-Matti Rouvinen
# Date: 2022-02-09
# Updated: 2024-02-09
# Version V1
##################################

# Imports
import pandas as pd
from pandas import DataFrame


def df_create_files(logpath):
    df1 = DataFrame({})
    df2 = DataFrame({})
    df3 = DataFrame({})
    df4 = DataFrame({})
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(logpath, engine="xlsxwriter")

    df1.to_excel(writer, sheet_name='training')
    df2.to_excel(writer, sheet_name='evaluation')
    df3.to_excel(writer, sheet_name='common')
    df4.to_excel(writer, sheet_name='class')

    writer.close()


def df_to_excel(train_data, train_header, common_data, common_header, eval_data, eval_header, class_data, class_header,
                filepath):
    train_df = DataFrame({})
    common_df = DataFrame({})
    eval_df = DataFrame({})
    class_df = DataFrame({})

    if len(train_data) > 0:
        train_df = DataFrame({train_header[0]: train_data[0], train_header[1]: train_data[1],
                              train_header[2]: train_data[2], train_header[3]: train_data[3],
                              train_header[4]: train_data[4], train_header[5]: train_data[5],
                              train_header[6]: train_data[6]
                              })
    if len(common_data) > 0:
        common_df = DataFrame({common_header[0]: common_data[0], common_header[1]: common_data[1],
                               common_header[2]: common_data[2], common_header[3]: common_data[3],
                               common_header[4]: common_data[4],common_header[5]: common_data[5]
                               })
    if len(eval_data) > 0:
        eval_df = DataFrame({eval_header[0]: eval_data[0], eval_header[1]: eval_data[1], eval_header[2]: eval_data[2],
                             eval_header[3]: eval_data[3], eval_header[4]: eval_data[4],
                             eval_header[5]: eval_data[5], eval_header[6]: eval_data[6]
                             })
    if len(class_data) > 0:
        index = 0
        frames = []
        for i in class_data:
            df = DataFrame({class_header[0]: i[0], class_header[1]: i[1],class_header[2]: i[2]}, index=[index])
            frames.append(df)
            index += 1
        class_df = pd.concat(frames)
    try:
        writer = pd.ExcelWriter(filepath, engine="xlsxwriter")
        train_df.to_excel(writer, sheet_name='training', index=False)
        common_df.to_excel(writer, sheet_name='common', index=False)
        eval_df.to_excel(writer, sheet_name='evaluation', index=False)
        class_df.to_excel(writer, sheet_name='class', index=False)
        writer.close()

    except PermissionError:
        pass
