import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

def visualize_training_history(model_history):
    plt.figure(figsize=(15, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Baseline Model - Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(model_history.history['loss'], label='Training Loss', marker='o')
    plt.plot(model_history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Baseline Model - Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print final metrics
    final_train_acc = model_history.history['accuracy'][-1]
    final_val_acc = model_history.history['val_accuracy'][-1]
    final_train_loss = model_history.history['loss'][-1]
    final_val_loss = model_history.history['val_loss'][-1]

    print("=== FINAL TRAINING METRICS ===")
    print(f"Training Accuracy: {final_train_acc:.3f}")
    print(f"Validation Accuracy: {final_val_acc:.3f}")
    print(f"Training Loss: {final_train_loss:.3f}")
    print(f"Validation Loss: {final_val_loss:.3f}")


def evaluate_model(model, test_data, class_names):
    # 1. Test Set Performance
    print("\n=== TEST SET EVALUATION ===")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_data, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # 2. Per-Class Metrics
    print("\n=== PER-CLASS METRICS ===")
    test_data.reset()
    y_true = test_data.classes
    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Continue with classification report and confusion matrix...
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))
    
    # Confusion Matrix
    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 5))
    
    # Confusion matrix (counts)
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Counts)\nBaseline Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Confusion matrix (percentages)
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (%)\nBaseline Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()