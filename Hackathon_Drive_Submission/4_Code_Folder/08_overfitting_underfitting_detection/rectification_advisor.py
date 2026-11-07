"""Step 8: Rectification Advisor for Overfitting and Underfitting.

Generates targeted, systematic engineering interventions based on the fit diagnosis:
- For Overfitting: Dropout increases, Weight Decay (L2), Data Augmentations, Early Stopping, Capacity reduction.
- For Underfitting: Model capacity scaling (e.g. transfer learning with EfficientNet/ResNet50),
  learning rate scheduling with warmup, reducing excessive regularization, class balancing.
"""
from __future__ import annotations


def get_rectification_plan(diagnosis_result: dict) -> dict:
    diagnosis = diagnosis_result["diagnosis"]
    plan = {
        "diagnosis": diagnosis,
        "recommended_techniques": [],
        "hyperparameter_adjustments": {},
        "action_items": [],
    }

    if diagnosis == "OVERFITTING":
        plan["recommended_techniques"] = [
            "Increased Dropout Regularization",
            "L2 Weight Decay Regularization",
            "Label Smoothing Cross-Entropy",
            "Advanced Data Augmentation (Mixup / CutMix / ColorJitter)",
            "Early Stopping with Patience",
        ]
        plan["hyperparameter_adjustments"] = {
            "dropout_rate": "Increase from 0.2 to 0.4 - 0.5",
            "weight_decay": "Increase to 1e-4 or 5e-4",
            "label_smoothing": "Set to 0.1",
            "patience": "Early stopping after 3 non-improving validation epochs",
        }
        plan["action_items"] = [
            "Inject stronger regularizers into classifier head.",
            "Apply random spatial translations, rotations, and color jitter to training data.",
            "Retrain model in Step 9 with these corrections enabled.",
        ]

    elif diagnosis == "UNDERFITTING":
        plan["recommended_techniques"] = [
            "Upgrade Model Backbone (Transfer Learning via EfficientNet-B0 or ResNet-50)",
            "Learning Rate Warmup with Cosine Annealing",
            "Extended Training Epochs (Allow model more convergence time)",
            "Class-Balanced Loss (Weighted Cross-Entropy or Focal Loss for minority classes)",
            "Reduce Excessive Regularization",
        ]
        plan["hyperparameter_adjustments"] = {
            "model_architecture": "Switch to pretrained EfficientNet-B0 or ResNet-50",
            "learning_rate": "Use lr=5e-4 with CosineAnnealingLR scheduler",
            "epochs": "Increase training duration to 5-10 epochs",
            "loss_function": "Weighted Cross-Entropy with inverse class frequencies",
        }
        plan["action_items"] = [
            "Adopt pretrained weights to provide expressive feature representations.",
            "Increase epochs and apply cosine learning rate schedule.",
            "Retrain model in Step 9 with these corrections enabled.",
        ]

    else:
        plan["recommended_techniques"] = [
            "Model is well-fitted; fine-tune with Cosine Annealing scheduler and test on held-out split.",
        ]
        plan["action_items"] = [
            "Proceed to Step 10 for final evaluation on untouched test set.",
        ]

    return plan
