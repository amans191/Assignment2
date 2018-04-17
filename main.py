import pandas as pd

encoding = 'utf-8-sig'

DataFrame = pd.read_csv('./queries.txt', header=None)

Location2 = r'./datadescription.txt'
df2 = pd.read_csv(Location2, names=['Names'])

sk = df2.iloc[0]
sk1 = df2.iloc[1]
sk2 = df2.iloc[2]
sk3 = df2.iloc[3]
sk4 = df2.iloc[4]
sk5 = df2.iloc[5]
sk6 = df2.iloc[6]
sk7 = df2.iloc[7]
sk8 = df2.iloc[8]
sk9 = df2.iloc[9]
sk10 = df2.iloc[10]
sk11 = df2.iloc[11]
sk12 = df2.iloc[12]
sk13 = df2.iloc[13]
sk14 = df2.iloc[14]
sk15 = df2.iloc[15]
sk16 = df2.iloc[16]
sk17 = df2.iloc[17]

one = sk['Names']
two = sk1['Names']
three = sk2['Names']
four = sk3['Names']
five = sk4['Names']
six = sk5['Names']
seven = sk6['Names']
eight = sk7['Names']
nine = sk8['Names']
ten = sk9['Names']
eleven = sk10['Names']
twelve = sk11['Names']
thirteen = sk12['Names']
fourteen = sk13['Names']
fifteen = sk14['Names']
sixteen = sk15['Names']
seventeen = sk16['Names']
eighteen = sk15['Names']

Location = r'./queries.txt'
DataFrame = pd.read_csv(Location, names=[one, two, three, four, five, six, seven,eight,nine,ten, eleven, twelve, thirteen,
                                  fourteen, fifteen, sixteen, seventeen, eighteen])

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

DataFrame['Outlier'] = True

print(DataFrame)
