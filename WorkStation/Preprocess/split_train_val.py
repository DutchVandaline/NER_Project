import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\junha\Datasets\WikiBio_wikipedia-biography-dataset\wikibio_preprocess.csv")

# 'birth_date'와 'death_date' 컬럼 삭제
df = df.drop(columns=['birth_date', 'death_date'])

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)

print("Train 파일 저장 완료: train.csv")
print("Validation 파일 저장 완료: validation.csv")