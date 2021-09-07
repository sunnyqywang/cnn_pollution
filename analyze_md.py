import numpy as np
import pandas as pd

output_folder = "200905"
index=[87,40,122,7]
for i in index:
    md = pd.read_csv("../output/"+output_folder+"/"+str(i)+"_md_mean.csv")

    md = md[md['id10'] < 8888888]
    print('Model', i)
    print("%.2f, %.2f" % (np.mean(md[['INDCT','INDOT','SVC','OIL','COAL']].to_numpy().flatten()) / 10000, \
          np.std(md[['INDCT','INDOT','SVC','OIL','COAL']].to_numpy().flatten()) / 10000))

    #print(np.mean(md[['INDOT','OIL']].to_numpy().flatten()))
    #print(np.mean(md[['INDCT','SVC','COAL']].to_numpy().flatten()))
