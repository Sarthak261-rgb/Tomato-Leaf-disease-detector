# results_summary.py
import csv
import matplotlib.pyplot as plt

# 👉 Enter your numbers here (replace with your actual outputs)
results = [
    {"model":"SVM (RBF)",      "accuracy": 0.9124},
    {"model":"RandomForest",   "accuracy": 0.9388},
    {"model":"KNN (k=5)",      "accuracy": 0.8842},
    {"model":"DecisionTree",   "accuracy": 0.8560},
    {"model":"CNN MobileNetV2","accuracy": 0.9785},
]

# Write CSV (Accuracy % and Misclassification %)
with open("results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Model","Accuracy (%)","Misclassification (%)"])
    for r in results:
        acc = r["accuracy"]*100
        mis = 100 - acc
        w.writerow([r["model"], f"{acc:.2f}", f"{mis:.2f}"])

# Bar chart
labels=[r["model"] for r in results]
accs=[r["accuracy"]*100 for r in results]

plt.figure()
plt.title("Model Comparison (Accuracy %)")
plt.bar(labels, accs)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig("results_bar.png", dpi=220)
print("✅ Saved: results.csv, results_bar.png")
