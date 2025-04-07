import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import operator
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score

df = pd.read_csv("HR_comma_sep.csv")
# No null values, no cleaning required
print(df.isnull().sum())

print(df.head())

c = df.corr(numeric_only = True)
sns.heatmap(c, annot = True, fmt = ".1f")
plt.title("Corr of Numeric Only")
plt.show()

cate_features = ["satisfaction_level", "last_evaluation"]
plt.figure(figsize = (20, 15))
for i, col in enumerate(cate_features, 1):
	plt.subplot(len(cate_features), 1, i)
	sns.boxplot(y = col, data = df)
	plt.title(f"Distribution of {col}")
	plt.ylabel(col)
plt.tight_layout()
plt.show()

sns.boxplot(y = df["average_monthly_hours"], data = df)
plt.title("Distrobution of Avg M Hours")
plt.ylabel("Hours/Month")
plt.show()

# Bar Plot of #Projects and Employees who left and stayed
group_df = df.groupby(['number_project', 'left']).size().reset_index(name = 'employee_count')
sns.barplot(data = group_df, y = 'employee_count', x = 'number_project', hue = 'left', legend = True)
plt.ylabel('Count')
plt.xlabel('Number of Projects')
plt.title('# of Projects for Past and Current Ems.')
plt.show()

# Selecting only employees who left
leavers = df[df['left'] == 1][['satisfaction_level', 'last_evaluation']]

# Apply Kmeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=41)
leavers['cluster'] = kmeans.fit_predict(leavers)

cluster_centers = kmeans.cluster_centers_

sns.scatterplot(data=leavers, x='satisfaction_level', y='last_evaluation', hue='cluster', legend = True, palette='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', marker='X', s=200, label='Cluster Centers')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('K-Means Cluster of Employees Who Left')
plt.show()

# interpretation:

df_num_features = df.select_dtypes(include='number')
df_cat_features = df.select_dtypes(include='object')
print(df_cat_features.head())
df_cat_features = pd.get_dummies(df_cat_features)
df = pd.concat([df_num_features, df_cat_features], axis=1) 


X = df.drop("left", axis = 1)
y = df["left"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

smote = SMOTE(random_state = 41) 
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.fit_transform(X_test)

X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for i, model in models.items():
	print("Model Name: ", i)
	model.fit(X_scaled, y_resampled)
	y_pred = model.predict(X_test)
	cv_results = cross_validate(model, X_scaled, y_resampled, cv=5, verbose=True)
	report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
	sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Blues", fmt=".2f")
	plt.title(f'Classification Report with {i}')
	plt.show()
# Finding and plotting ROC and AUC
for i, model in models.items():
	y_pred_proba = model.predict_proba(X_test)[:, 1]
	fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
	roc_auc = auc(fpr, tpr)
	print(f'auc for {model}: {roc_auc}')
	plt.figure()
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve for Employee Turnover Rates')
	plt.legend()
	plt.show()

# Best model is Gradient Boosting

model = GradientBoostingClassifier()
model.fit(X_scaled, y_resampled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

zones = [
		'Safe_Zone(Green)',
		'Low_Risk_Zone(Yellow)',
		'Medium_Risk_Zone(Orange)',
		'High_Risk_Zone(Red)'
]

df = X_test
df['Turnover_Probability'] = y_test_proba
df['Zone'] = pd.cut(
		df['Turnover_Probability'],
		bins = [0.0, 0.2, 0.6, 0.9, 1.0],
		labels = zones
)

print(df[["Turnover_Probability", "Zone"]].head(5))
