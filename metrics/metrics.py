import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings

def convert_to_one_hot(labels, num_classes):
    """
    Convert integer labels to one-hot encoding.
    """
    one_hot_labels = np.eye(num_classes)[labels]
    return one_hot_labels

def calculate_roc_auc(probabilities, labels):
    """
    Calculate ROC curve and AUC for multi-class or binary classification.

    Inputs:
    - probabilities: numpy array of shape (N, C), probabilities for each class.
    - labels: numpy array of shape (N, C), labels in one-hot encoding with values 0 and 1.

    Returns:
    - fpr: dict of false positive rates for each class.
    - tpr: dict of true positive rates for each class.
    - roc_auc: dict of AUC values for each class.
    """
    probabilities_np = np.array(probabilities, dtype=float)
    labels_np = np.array(labels, dtype=int)
    
    # Ensure probabilities and labels are at least 2D arrays
    if probabilities_np.ndim == 1:
        probabilities_np = probabilities_np.reshape(-1, 1)
    if labels_np.ndim == 1:
        labels_np = labels_np.reshape(-1, 1)
    
    # Ensure the labels and probabilities have the same shape
    if probabilities_np.shape != labels_np.shape:
        raise ValueError("The shape of probabilities and labels must be the same.")
    
    N, C = probabilities_np.shape
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for class_idx in range(C):
        y_true = labels_np[:, class_idx]
        y_score = probabilities_np[:, class_idx]
        
        # Check if both classes are present
        if np.unique(y_true).size < 2:
            warnings.warn(f"Only one class present in y_true for class {class_idx}. ROC AUC is not defined.")
            roc_auc[class_idx] = np.nan
            fpr[class_idx] = np.array([0, 1])
            tpr[class_idx] = np.array([0, 1]) if y_true[0] == 1 else np.array([0, 0])
        else:
            fpr[class_idx], tpr[class_idx], _ = roc_curve(y_true, y_score)
            roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
    
    return fpr, tpr, roc_auc






def find_best_threshold(probabilities, all_labels):
    probabilities_np = np.array(probabilities)
    all_labels_np = np.array(all_labels)

    if len(all_labels_np.shape) == 1:  # Convert labels to one-hot if not already
        num_classes = len(np.unique(all_labels_np))
        all_labels_np = convert_to_one_hot(all_labels_np, num_classes)

    num_classes = all_labels_np.shape[1]
    thresholds = dict()

    for class_idx in range(num_classes):
        y_true = all_labels_np[:, class_idx]
        y_prob = probabilities_np[:, class_idx]

        p, r, t = precision_recall_curve(y_true.astype(int), y_prob)

        # Initialize F1 scores array
        f1_scores = np.zeros_like(t)

        # Compute denominators
        denominators = p[:-1] + r[:-1]

        # Create mask for denominators that are not zero
        non_zero_denominator = denominators != 0

        # Compute F1 scores where denominator is not zero
        f1_scores[non_zero_denominator] = 2 * p[:-1][non_zero_denominator] * r[:-1][non_zero_denominator] / denominators[non_zero_denominator]

        # The index of the best F1 score
        best_index = np.argmax(f1_scores)

        # Best threshold corresponds to the best F1 score
        best_threshold = t[best_index]
        thresholds[class_idx] = best_threshold

    return thresholds



def calculate_fpr_fnr(probability, ground_truth_label, threshold):
    """
    Calculate FPR and FNR for each class.

    :param probability: numpy array of shape (N, C)
    :param ground_truth_label: numpy array of shape (N, C), values 0 or 1
    :param threshold: dict {class_index: threshold_value}

    :return: dict {'FPR': {class_index: value}, 'FNR': {class_index: value}}
    """
    import numpy as np

    N, C = probability.shape
    fpr = {}
    fnr = {}

    for c in range(C):
        # Get threshold for class c, default to 0.5 if not provided
        thresh = threshold.get(c, 0.5)
        # Predicted labels based on threshold
        pred = (probability[:, c] > thresh).astype(int)
        # Ground truth labels
        gt = ground_truth_label[:, c]

        # Calculate True Positives, True Negatives, False Positives, False Negatives
        TP = np.sum((pred == 1) & (gt == 1))
        TN = np.sum((pred == 0) & (gt == 0))
        FP = np.sum((pred == 1) & (gt == 0))
        FN = np.sum((pred == 0) & (gt == 1))

        # Calculate FPR and FNR, handle division by zero
        fpr_c = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr_c = FN / (TP + FN) if (TP + FN) > 0 else 0.0

        # Store results in dictionaries
        fpr[c] = fpr_c
        fnr[c] = fnr_c

    return fpr, fnr




if __name__ == '__main__':
    # Test
    probabilities_test = np.random.rand(100000, 14)
    labels_test = np.random.randint(0, 2, size=(100000, 14))

    # Test calculate_roc_auc function
    print("Test calculate_roc_auc:")
    fpr_test, tpr_test, roc_auc_test = calculate_roc_auc(probabilities_test, labels_test)
    # print("FPR:", fpr_test)
    # print("TPR:", tpr_test)
    print("ROC AUC:", roc_auc_test)
    print()

    # Test find_best_threshold function
    print("Test find_best_threshold:")
    thresholds_test = find_best_threshold(probabilities_test, labels_test)
    print("Best Thresholds:", thresholds_test)
    print()

    # Test calculate_fpr_fnr function
    print("Test calculate_fpr_fnr:")
    fpr_numbers_test, fnr_numbers_test = calculate_fpr_fnr(probabilities_test, labels_test, thresholds_test)
    print("FPR Numbers:", fpr_numbers_test)
    print("FNR Numbers:", fnr_numbers_test)