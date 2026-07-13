from pathlib import Path
import csv
import gc
import json
import time
import traceback

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAST_CHECKPOINT_PATH = MODEL_DIR / "efficientnetv2_last_checkpoint.pth"
BEST_MODEL_PATH = MODEL_DIR / "efficientnetv2_best.pth"
FINAL_MODEL_PATH = MODEL_DIR / "efficientnetv2_final.pth"

HISTORY_CSV_PATH = RESULTS_DIR / "efficientnetv2_history.csv"
RESULTS_JSON_PATH = RESULTS_DIR / "efficientnetv2_results.json"
RESULTS_TEXT_PATH = RESULTS_DIR / "efficientnetv2_results.txt"

ACCURACY_CHART_PATH = RESULTS_DIR / "efficientnetv2_accuracy.png"
LOSS_CHART_PATH = RESULTS_DIR / "efficientnetv2_loss.png"

IMAGE_SIZE = 224
BATCH_SIZE = 4
MAX_EPOCHS = 10
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 3
LR_PATIENCE = 1

START_FROM_SCRATCH = False
RANDOM_SEED = 42


# =========================================================
# DEVICE AND REPRODUCIBILITY
# =========================================================

torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("CONVNEXT-TINY TRAINING")
print("=" * 60)
print(f"Using device: {device}")
print(f"Image size: {IMAGE_SIZE} × {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {MAX_EPOCHS}")


# =========================================================
# CHECK DATA FOLDERS
# =========================================================

for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Required folder not found: {directory}"
        )


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.90, 1.0)
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3)],
        p=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

evaluation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# DATASETS AND LOADERS
# =========================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=evaluation_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=evaluation_transform
)

class_names = train_dataset.classes
number_of_classes = len(class_names)

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError(
        "Training and validation class mappings do not match."
    )

if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise ValueError(
        "Training and test class mappings do not match."
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"Number of classes: {number_of_classes}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Testing images: {len(test_dataset)}")


# =========================================================
# PRETRAINED CONVNEXT-TINY
# =========================================================

print("\nLoading pretrained ConvNeXt-Tiny...")

weights = ConvNeXt_Tiny_Weights.DEFAULT

model = convnext_tiny(weights=weights)

# Freeze the pretrained feature extractor.
for parameter in model.features.parameters():
    parameter.requires_grad = False

classifier_input_features = model.classifier[2].in_features

model.classifier[2] = nn.Linear(
    classifier_input_features,
    number_of_classes
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=LR_PATIENCE,
    min_lr=0.00001
)


# =========================================================
# HISTORY AND STATE
# =========================================================

history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "learning_rate": []
}

start_epoch = 0
best_val_accuracy = 0.0
epochs_without_improvement = 0
total_training_seconds = 0.0


# =========================================================
# CHECKPOINT FUNCTIONS
# =========================================================

def save_checkpoint(
    path: Path,
    completed_epoch: int,
    current_best_accuracy: float,
    unimproved_epochs: int,
    elapsed_seconds: float
):
    torch.save(
        {
            "epoch": completed_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_accuracy": current_best_accuracy,
            "epochs_without_improvement": unimproved_epochs,
            "history": history,
            "class_names": class_names,
            "class_to_idx": train_dataset.class_to_idx,
            "image_size": IMAGE_SIZE,
            "training_seconds": elapsed_seconds
        },
        path
    )


def load_checkpoint():
    global start_epoch
    global best_val_accuracy
    global epochs_without_improvement
    global total_training_seconds
    global history

    if START_FROM_SCRATCH:
        print("Starting from scratch.")
        return

    if not LAST_CHECKPOINT_PATH.exists():
        print("No checkpoint found. Starting from epoch 1.")
        return

    print(f"Loading checkpoint: {LAST_CHECKPOINT_PATH}")

    checkpoint = torch.load(
        LAST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    best_val_accuracy = checkpoint.get(
        "best_val_accuracy",
        0.0
    )

    epochs_without_improvement = checkpoint.get(
        "epochs_without_improvement",
        0
    )

    history = checkpoint.get(
        "history",
        history
    )

    total_training_seconds = checkpoint.get(
        "training_seconds",
        0.0
    )

    print(
        f"Resuming from epoch {start_epoch + 1}"
    )


# =========================================================
# OUTPUT FUNCTIONS
# =========================================================

def save_history_csv():
    with HISTORY_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "learning_rate"
        ])

        for index in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][index],
                history["train_loss"][index],
                history["train_accuracy"][index],
                history["val_loss"][index],
                history["val_accuracy"][index],
                history["learning_rate"][index]
            ])


def save_charts():
    if not history["epoch"]:
        return

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        label="Training accuracy"
    )

    plt.plot(
        history["epoch"],
        history["val_accuracy"],
        label="Validation accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ConvNeXt-Tiny Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        ACCURACY_CHART_PATH,
        dpi=200
    )
    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Training loss"
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ConvNeXt-Tiny Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        LOSS_CHART_PATH,
        dpi=200
    )
    plt.close()


# =========================================================
# TRAINING FUNCTIONS
# =========================================================

def train_one_epoch():
    model.train()

    # Keep frozen feature layers in evaluation mode.
    model.features.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total_examples += batch_size

        del images, labels, outputs, loss

    return (
        total_loss / total_examples,
        total_correct / total_examples
    )


def validate_one_epoch():
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total_examples += batch_size

            del images, labels, outputs, loss

    return (
        total_loss / total_examples,
        total_correct / total_examples
    )


def evaluate_test_set():
    model.eval()

    total_loss = 0.0
    total_examples = 0
    top_1_correct = 0
    top_5_correct = 0

    evaluation_start = time.perf_counter()

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

            top_1_predictions = outputs.argmax(dim=1)

            top_1_correct += (
                top_1_predictions == labels
            ).sum().item()

            top_5_predictions = outputs.topk(
                k=5,
                dim=1
            ).indices

            top_5_correct += (
                top_5_predictions
                == labels.unsqueeze(1)
            ).any(dim=1).sum().item()

            del (
                images,
                labels,
                outputs,
                loss,
                top_1_predictions,
                top_5_predictions
            )

    evaluation_seconds = (
        time.perf_counter() - evaluation_start
    )

    return {
        "test_loss": total_loss / total_examples,
        "top_1_accuracy":
            top_1_correct / total_examples,
        "top_5_accuracy":
            top_5_correct / total_examples,
        "evaluation_seconds": evaluation_seconds,
        "average_inference_ms_per_image":
            (
                evaluation_seconds
                / total_examples
            ) * 1000
    }


# =========================================================
# LOAD CHECKPOINT
# =========================================================

load_checkpoint()


# =========================================================
# TRAINING LOOP
# =========================================================

try:
    for epoch_index in range(
        start_epoch,
        MAX_EPOCHS
    ):
        epoch_number = epoch_index + 1

        print("\n" + "-" * 60)
        print(f"Epoch {epoch_number}/{MAX_EPOCHS}")
        print("-" * 60)

        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch()

        val_loss, val_accuracy = (
            validate_one_epoch()
        )

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        total_training_seconds += epoch_seconds

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch_number)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(
            train_accuracy
        )
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(
            val_accuracy
        )
        history["learning_rate"].append(
            current_lr
        )

        print(f"Train loss: {train_loss:.4f}")
        print(
            f"Train accuracy: {train_accuracy:.4f}"
        )
        print(
            f"Validation loss: {val_loss:.4f}"
        )
        print(
            f"Validation accuracy: {val_accuracy:.4f}"
        )
        print(f"Learning rate: {current_lr:.6f}")
        print(
            f"Epoch time: {epoch_seconds:.2f} seconds"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            save_checkpoint(
                BEST_MODEL_PATH,
                epoch_number,
                best_val_accuracy,
                epochs_without_improvement,
                total_training_seconds
            )

            print("New best model saved.")
        else:
            epochs_without_improvement += 1

        scheduler.step(val_accuracy)

        save_checkpoint(
            LAST_CHECKPOINT_PATH,
            epoch_number,
            best_val_accuracy,
            epochs_without_improvement,
            total_training_seconds
        )

        save_history_csv()
        save_charts()

        print("Latest checkpoint saved.")

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping activated."
            )
            break

except KeyboardInterrupt:
    print("\nTraining stopped manually.")

    save_checkpoint(
        LAST_CHECKPOINT_PATH,
        history["epoch"][-1]
        if history["epoch"]
        else start_epoch,
        best_val_accuracy,
        epochs_without_improvement,
        total_training_seconds
    )

    save_history_csv()
    save_charts()

    print(
        "Progress saved. Run the script again "
        "to resume."
    )

except RuntimeError as error:
    print("\nRuntime error:")
    print(error)

    save_checkpoint(
        LAST_CHECKPOINT_PATH,
        history["epoch"][-1]
        if history["epoch"]
        else start_epoch,
        best_val_accuracy,
        epochs_without_improvement,
        total_training_seconds
    )

    save_history_csv()
    save_charts()

    print(
        "Progress saved. Run the script again "
        "to resume."
    )

    raise

except Exception:
    traceback.print_exc()
    save_history_csv()
    save_charts()
    raise


# =========================================================
# LOAD BEST MODEL
# =========================================================

if BEST_MODEL_PATH.exists():
    print(f"\nLoading best model: {BEST_MODEL_PATH}")

    best_checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    best_val_accuracy = best_checkpoint[
        "best_val_accuracy"
    ]


# =========================================================
# FINAL TEST
# =========================================================

print("\nEvaluating best model on test set...")

test_results = evaluate_test_set()

total_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": train_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
        "number_of_classes": number_of_classes,
        "best_val_accuracy": best_val_accuracy,
        "test_results": test_results
    },
    FINAL_MODEL_PATH
)

final_results = {
    "model": "ConvNeXt-Tiny",
    "number_of_classes": number_of_classes,
    "training_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "testing_images": len(test_dataset),
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "best_validation_accuracy":
        best_val_accuracy,
    "test_loss":
        test_results["test_loss"],
    "top_1_accuracy":
        test_results["top_1_accuracy"],
    "top_5_accuracy":
        test_results["top_5_accuracy"],
    "average_inference_ms_per_image":
        test_results[
            "average_inference_ms_per_image"
        ],
    "total_parameter_count":
        total_parameter_count,
    "trainable_parameter_count":
        trainable_parameter_count,
    "training_seconds":
        total_training_seconds
}

with RESULTS_JSON_PATH.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        final_results,
        file,
        indent=4
    )

with RESULTS_TEXT_PATH.open(
    "w",
    encoding="utf-8"
) as file:
    file.write("CONVNEXT-TINY FINAL RESULTS\n")
    file.write("=" * 40 + "\n")
    file.write(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}\n"
    )
    file.write(
        f"Test loss: "
        f"{test_results['test_loss']:.4f}\n"
    )
    file.write(
        f"Top-1 accuracy: "
        f"{test_results['top_1_accuracy']:.4f}\n"
    )
    file.write(
        f"Top-5 accuracy: "
        f"{test_results['top_5_accuracy']:.4f}\n"
    )
    file.write(
        f"Average inference time: "
        f"{test_results['average_inference_ms_per_image']:.4f} "
        f"ms/image\n"
    )
    file.write(
        f"Total parameters: "
        f"{total_parameter_count}\n"
    )
    file.write(
        f"Trainable parameters: "
        f"{trainable_parameter_count}\n"
    )
    file.write(
        f"Training time: "
        f"{total_training_seconds:.2f} seconds\n"
    )

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)
print(
    f"Test loss: "
    f"{test_results['test_loss']:.4f}"
)
print(
    f"Top-1 accuracy: "
    f"{test_results['top_1_accuracy']:.4f}"
)
print(
    f"Top-5 accuracy: "
    f"{test_results['top_5_accuracy']:.4f}"
)
print(
    f"Average inference time: "
    f"{test_results['average_inference_ms_per_image']:.4f} "
    f"ms/image"
)
print(
    f"Total parameters: "
    f"{total_parameter_count}"
)
print(
    f"Trainable parameters: "
    f"{trainable_parameter_count}"
)
print(
    f"Training time: "
    f"{total_training_seconds:.2f} seconds"
)

print("\nSaved files:")
print(FINAL_MODEL_PATH)
print(BEST_MODEL_PATH)
print(LAST_CHECKPOINT_PATH)
print(HISTORY_CSV_PATH)
print(ACCURACY_CHART_PATH)
print(LOSS_CHART_PATH)
print(RESULTS_JSON_PATH)
print(RESULTS_TEXT_PATH)
