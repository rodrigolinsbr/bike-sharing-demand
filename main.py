import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")
#lendo dados

h =df.head(10)
#mostrando data
print  df
d=df.describe()
#descricao

# name = df['Name'].value_counts()
# sex = df['Survived'].hist(bins=100)
# s=df['Survived'].hist(bins=50)


#plt.plot(str(sex))
#plt.title("Muito Fcil")
#plt.show()


