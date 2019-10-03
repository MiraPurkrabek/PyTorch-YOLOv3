import csv
import ast

with open('via_export_csv.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            # for i in range(len(row)):
            #     print(row[i], end='\t')
            # print('')
            coors = ast.literal_eval( row[5] )
            x_center = coors["x"] + coors["width"]/2
            y_center = coors["y"] + coors["height"]/2
            img_name = row[0]
            print(f'image name: "{img_name}", {x_center}, {y_center}, {coors["width"]}, {coors["height"]}')
            print(row[6])
            info = ast.literal_eval( row[6] )
            line_count += 1
    print(f'Processed {line_count} lines.')