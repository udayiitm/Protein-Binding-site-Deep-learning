import pickle


results = None
with open("folds_results.pickle", "rb") as f:
    results = pickle.load(f)


print(results[0].keys())

loss = []
acc = []
f1 = []
f2 = []
iou_metric = []
val_iou_metric = []
dice_loss = []
val_loss = []
val_acc = []
val_f1 = []
val_f2 = []

for fold in results:
    loss.extend(fold["loss"])
    acc.extend(fold["acc"])
    f1.extend(fold["f1_score"])
    f2.extend(fold["f2_score"])
    iou_metric.extend(fold["iou_metric"])
    dice_loss.extend(fold["dice_loss"])
    val_loss.extend(fold["val_loss"])
    val_acc.extend(fold["val_acc"])
    val_f1.extend(fold["val_f1_score"])
    val_f2.extend(fold["val_f2_score"])
    val_iou_metric.extend(fold["val_iou_metric"])
    dice_loss.extend(fold["val_dice_loss"])


print(loss)
print(acc)
print(f1)
print(f2)
print(iou_metric)
print(val_iou_metric)
print(dice_loss)
print(val_loss)
print(val_acc)
print(val_f1)
print(val_f2)

import matplotlib.pyplot as plt

plt.plot(loss, label="Training Loss")
# plt.plot(val_loss, label="Validation Loss")

# plt.plot(acc, label="Accuracy")
plt.plot(val_acc, label="Val Accuracy")

# plt.plot(f1, label="F1 Score")
# plt.plot(val_f1, label="F1 Val Score")

# plt.plot(f2, label="F2 Score")
# plt.plot(val_f2, label="F2 Val Score")

plt.plot(iou_metric, label="IOU")
plt.plot(val_iou_metric, label="Val IOU")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.title(f"Whole Training History")
plt.savefig(fname=f"training")
plt.show()
