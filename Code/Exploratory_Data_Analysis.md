## Distribution of Biopsy Results

**Here, I show the distribution of negative and positive biopsy results in the data.**

```Python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cervical_risk.csv', na_values='?')

biopsy_counts = df['Biopsy'].value_counts()
biopsy_counts.plot(kind='bar')
plt.xticks(ticks=[0,1], labels=['Negative','Positive'], rotation= 1)
plt.ylabel('Count')
```

![Count of Biopsy Results](../Images/biopsy.png)

## Histogram of all the features 

```python
def plot_hist(df):
    num_feat = df.shape[1]
    num_rows = (num_feat//7)+1
    fig, axes = plt.subplots(num_rows, 6, figsize=(20, num_rows*3))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor='k')
        axes[i].set_title(col, fontsize=14)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()

plot_hist(df)
```
![Raw Data Histogram](../Images/total_hist.png)

## Histogram of all the features for positive biopsy data

```Python
pos_biopsy_df = df[df['Biopsy'] == 1]
plot_hist(pos_biopsy_df)
```
![Positive Biopsy Data Histogram](../Images/pos_biopsy_hist.png)

## The number of null values for each feature

```Python
null_count = df.isnull().sum()
null_df = pd.DataFrame(null_count, columns=['Number of Null Values'])
null_df.reset_index(inplace=True)
null_df.rename(columns={'index': 'Feature'}, inplace=True)
null_df = null_df.to_markdown(index=False)
```

![Null Values](../Images/NullVal.png)

## Data Preprocessing 

**The majority of the two features "STDs: Time since first diagnosis", and "STDs: Time since last diagnosis" are null values. Therefore, I drop these two features.**

```Python
df_drop = df.drop(columns=["STDs: Time since first diagnosis", "STDs: Time since last diagnosis"])
```

**The datatype for all the features is either int or float. I convert the binary variables to boolean to explicitly mark them as binary/categorical variables.**

```Python
def convert_to_bool(dfi):
    for k in dfi.columns:
        if dfi[k].dropna().value_counts().index.isin([0, 1]).all():
            dfi[k] = dfi[k].astype('boolean')
    return dfi


df_type = convert_to_bool(df_drop)
```
**I will use the mode of the categorical variables to impute the missing values. Furthermore, given the non-normal distribution of the numerical variables, I will use the median rather than the mean to impute the missing values.**

```Python
for col in df_type.select_dtypes(include='boolean').columns:
    df_type[col].fillna(df_type[col].mode()[0], inplace=True)

df_impute = df_type.fillna(df_type.median())
```



