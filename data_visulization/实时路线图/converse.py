from collections import OrderedDict
import json
import codecs
import xlrd



wb = xlrd.open_workbook('1.xlsx')

convert_list = []
sh = wb.sheet_by_index(0)
title = sh.row_values(0)
for rownum in range(1, sh.nrows):
    rowvalue = sh.row_values(rownum)

    print(str(rowvalue)+',')


    convert_list.append(str(rowvalue))



with codecs.open('positive_previous1.txt', "w", "utf-8") as f:
    for item in convert_list:
      f.write(item+',')

