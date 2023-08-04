import pandas as pd
import json
from geojson import GeoJSON


def convert_csv_to_geojson(input_file, output_file):

    # Read the CSV data into a pandas DataFrame

    cols = ['x', 'y', 'values'] 
    df = pd.read_csv(input_file, names=cols)
    
    print(df.keys())

    # Create a GeoJSON object

    geojson = {

        'type': 'FeatureCollection',

        'features': []

    }

    # Loop through each row in the DataFrame and convert to a GeoJSON feature

    for value_id in np.unique(df['values'].values):

        pixels = df.loc[df['values'] == value_id]

        xs = pixels['x'].values

        ys = pixels['y'].values
        
        order = np.argsort(xs)
        tmp = order[-1]
        order[-1] = order[-2]
        order[-2] = tmp
        # print(order)
        
        xs = xs[order]
        ys = ys[order]

        coords = [[int(xi), int(yi)] for (xi, yi) in zip(np.append(xs, np.array(xs[0])), np.append(ys, np.array(ys[0])))] 

        properties = {

        'name': str(value_id),

        'description': "a pixel"

        }

        features = {

        "type": "Feature",  
        "id": f"roi_{value_id}",
        "originalObjectId": str(value_id),
        "otherProperty": str(value_id),
        "name": str(value_id),

        "geometry": {

        "type": "Polygon",

        "coordinates": [

        coords

        ]

        },

        "properties": {
            "objectType": "annotation",
            "originalObjectId": str(value_id),
            "otherProperty": str(value_id),
            "name": str(value_id),
        }

            }

        #print(features)

        geojson['features'].append(features)
    
    with open(output_file, "w") as geojson_file:
        json.dump(geojson, geojson_file, indent=2)
    

    # with open(output_file, 'w') as outfile:
    #     GeoJSON.dump(geojson,output_file)


# Example usage

input_file = '~/Downloads/new_csv_file.csv'

output_file = 'output.geojson'

convert_csv_to_geojson(input_file, output_file)
