import os
import json
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score

import model as my_model
from extract_features import extract_features


def load_data(data_dir, device):
    # model and transforms for extract_features
    resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")

    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # starting data loading
    videos = []
    labels = []
    class_names = os.listdir(data_dir)
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for video_name in tqdm(os.listdir(class_dir), desc=f"Process '{class_name}' class", unit="clip"):
            video_path = os.path.join(class_dir, video_name)
            try:
                features = extract_features(video_path, device, resnet, transform)
                if features:
                    videos.append(np.array(features))
                    labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"Error during processing video {video_path}: {e}")
                continue

    return videos, labels


def collate_fn(batch):
    '''
    Added padding for each batch
    '''
    videos, labels = zip(*batch)

    max_len = max([len(video) for video in videos])

    padded_videos = []
    for video in videos:
        pad_len = max_len - len(video)
        padded_video = np.pad(video, ((0, pad_len), (0, 0)), 'constant')
        padded_videos.append(padded_video)

    padded_videos = torch.tensor(np.array(padded_videos), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_videos, labels


def test(device, test_loader, model):
    model.eval()
    with tqdm(enumerate(test_loader), desc=f"Testing", unit="iter") as pbar:
        all_preds = []
        all_labels = []
        for i, (videos, labels) in pbar:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_accuracy = accuracy_score(all_labels, all_preds)
            running_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            running_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)

            pbar.set_postfix({"acc": running_accuracy, "prec": running_precision, "rec": running_recall})

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    
    print("Total testing accuracy:", accuracy)
    print("Total testing precision:", precision)
    print("Total testing recall:", recall)
    

def train(num_epochs, device, train_loader, model, criterion, optimizer, work_save_directory, save_period=-1):
    save_directory = f"{work_save_directory}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    results = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}
    for epoch in range(num_epochs):
        all_preds = []
        all_labels = []
        all_loss = 0
        with tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="iter") as pbar:
            for i, (videos, labels) in pbar:
                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_loss += loss.item()

                running_accuracy = accuracy_score(all_labels, all_preds)
                running_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
                running_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)

                pbar.set_postfix({"loss": loss.item(), "acc": running_accuracy, "prec": running_precision, "rec": running_recall})

        loss = all_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)

        results["Loss"].append(loss)
        results["Accuracy"].append(accuracy)
        results["Precision"].append(precision)
        results["Recall"].append(recall)

        print(f"Epoch {epoch + 1}/{num_epochs}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        if save_period > 0:
            if (epoch + 1) % save_period == 0:
                torch.save(model.state_dict(), f"{save_directory}/model_{epoch + 1}.pth")

    torch.save(model.state_dict(), f"{save_directory}/model_last.pth")

    with open(f"{save_directory}/results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)
    return results


if __name__ == "__main__":
    MODE = "test"
    weights_path = "work/20250406_234306/model_last.pth"

    input_size = 512
    hidden_size = 256
    num_layers = 2
    num_classes = 2
    learning_rate = 0.001
    num_epochs = 400
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting work on device:", device)
    
    # model creation
    model = my_model.VideoClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    if MODE == "test":
        model.load_state_dict(torch.load(weights_path))
    
    # loading data
    data_dir = "dataset_img"
    print("Loading data...")
    videos, labels = load_data(data_dir, device)
    print("Data loading done.")

    # train and test partition
    videos_train, videos_test, labels_train, labels_test = train_test_split(
        videos,
        labels,
        test_size=0.2,
        random_state=42
    )

    # creating datasets and dataloaders
    train_dataset = my_model.VideoDataset(videos_train, labels_train)
    test_dataset = my_model.VideoDataset(videos_test, labels_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    one_class = np.sum(train_dataset.labels) / np.size(train_dataset.labels)
    print("Class weights:", 1 - one_class, one_class)
    class_weights = torch.tensor([1 - one_class, one_class], dtype=torch.float).to(device)

    # set optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if MODE == "train":
        print("Starting training process...")
        train(num_epochs, device, train_loader, model, criterion, optimizer, "work", 10)
        print("Train done.")
    elif MODE == "test":
        print("Starting testing process...")
        test(device, test_loader, model)
        print("Test done.")
    else:
        print("ERROR: Unknown mode.")
