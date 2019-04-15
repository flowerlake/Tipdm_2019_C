from collections import OrderedDict
import json
import codecs
import xlrd
import pandas
import os

src_path = r'/Users/summerone/Desktop/Taidi/test/fen_data/data'
dst_path = r'/Users/summerone/Desktop/Taidi/test/fen_data/result_file'


def convert(loc):
    filename = src_path + '/' + loc
    data = pandas.read_csv(filename)
    print(data.shape[0])
    col = data.shape[0]

    convert_list = []
    for rownum in range(col):
        rowvalue = [data['lng'][rownum], data['lat'][rownum]]
        convert_list.append(rowvalue)
    loc1 = loc.split('.')[0]
    print(loc1)

    with codecs.open(dst_path + '/' + str(loc1) + '.json', "w", "utf-8") as f:
        tmp1='{\r\n\
  "type": "FeatureCollection",\r\n\
  "features": [\r\n\
    {\r\n\
      "type": "Feature",\r\n\
      "geometry": {\r\n\
        "type": "LineString",\r\n\
        "coordinates": ['

        print(tmp1)
        f.write(tmp1)
        for i,item in enumerate(convert_list):
            if i !=len(convert_list):
                f.write(str(item) + ',')
            else:
                f.write(str(item))
        f.write(']}}]}')
def main():
  filenames = os.listdir(src_path)
  print(filenames)

  for i in filenames:
    print(i)
    convert(i)

if __name__ == '__main__':
    main()

