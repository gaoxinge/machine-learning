import os
import numpy as np 
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg11


def preprocess(image):
    image = image.to(torch.float32)
    image /= 255
    return image


def deprocess(image):
    image *= 255
    image = image.to(torch.uint8)
    return image


def convert1(coord):
    cx, cy, cw, ch = coord[0], coord[1], coord[2], coord[3]
    cx, cy, cw, ch = (cx - 112) / 224, (cy - 112) / 224, torch.log(cw / 224), torch.log(ch / 224)
    return torch.tensor([cx, cy, cw, ch])


def convert2(coord):
    cx, cy, cw, ch = coord[0], coord[1], coord[2], coord[3]
    cx, cy, cw, ch = cx * 224 + 112, cy * 224 + 112, torch.exp(cw) * 224, torch.exp(ch) * 224 
    return torch.tensor([cx, cy, cw, ch])


def process(image_path, label_path=None):
    image = Image.open(image_path)
    
    w, h = image.size
    s = 224 / min(w, h)
    image = image.resize((int(w * s), int(h * s)))
      
    w, h = image.size
    w0, h0 = (w - 224) // 2, (h - 224) // 2
    image = image.crop((w0, h0, w - w0, h - h0))
        
    w, h = image.size
    if w != 224 or h != 224:
        image = image.resize((224, 224))

    image = np.array(image)
    image = torch.from_numpy(image)
    image = preprocess(image)
    
    if label_path is not None:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                _, cx, cy, cw, ch = line.strip().split(" ")
                cx, cy, cw, ch = float(cx), float(cy), float(cw), float(ch)
                break

        cx, cy, cw, ch = cx * s, cy * s, cw * s, ch * s
        xx1, yy1, xx2, yy2 = cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2
        xx1 = min(max(xx1 - w0, 0), 224)
        yy1 = min(max(yy1 - h0, 0), 224)
        xx2 = max(min(xx2 - w0, 224), 0)
        yy2 = max(min(yy2 - h0, 224), 0)
        if xx1 >= xx2 or yy1 >= yy2:
            return None
        cx, cy, cw, ch = (xx1 + xx2) / 2, (yy1 + yy2) / 2, xx2 - xx1, yy2 - yy1
        label = torch.tensor([cx, cy, cw, ch])
        label = convert1(label)

    if label_path is not None:
        return image, label
    else:
        return image


def draw(image, label, path):
    image = deprocess(image)
    image = image.numpy()
    image = Image.fromarray(image)
        
    label = convert2(label)
    cx, cy, cw, ch = label
    cx, cy, cw, ch = cx.numpy(), cy.numpy(), cw.numpy(), ch.numpy()
        
    draw = ImageDraw.Draw(image)
    draw.rectangle(((cx - cw / 2, cy - ch / 2), (cx + cw / 2, cy + ch / 2)), outline=(0, 0, 0), width=1)
    image.save(path)


class ODDataset(Dataset):

    def __init__(self):
        if not os.path.exists("data/train_cache.txt"):
            with open("data/train_cache.txt", "w", encoding="utf-8") as f:
                with open("data/train.txt", "r", encoding="utf-8") as g:
                    for idx, line in enumerate(g):
                        image_path = line.strip()
                        label_path = image_path.replace("JPEGImages", "labels").replace(".jpg", ".txt")
                        r = process(image_path, label_path)
                        if r is not None:
                            print(idx, image_path, "yes")
                            f.write(image_path)
                            f.write("\n")
                        else:
                            print(idx, image_path, "no")

        with open("data/train_cache.txt", "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f]


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_path = self.lines[idx]
        label_path = image_path.replace("JPEGImages", "labels").replace(".jpg", ".txt")
        return process(image_path, label_path)


class ODModel(torch.nn.Module):

    def __init__(self):
        super(ODModel, self).__init__()
        vgg_model = vgg11()
        self.vgg_model_features = vgg_model.features
        self.layer = torch.nn.Linear(in_features=25088, out_features=4)

    def __call__(self, x):
        x = self.vgg_model_features(x)
        x = x.reshape(x.size(0), -1)
        x = self.layer(x)
        return x


def test_dataset():
    train_dataset = ODDataset() 
    train_dataloader = DataLoader(train_dataset, batch_size=1) 
    print(len(train_dataset))

    for batch, (images, labels) in enumerate(train_dataloader):
        if batch == 10:
            image = images.squeeze(0)
            label = labels.squeeze(0)
            draw(image, label, "test.jpg")
            break


def train(model_path=None, model_dir="checkpoint"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    train_dataset = ODDataset() 
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) 

    od_model = ODModel()
    od_model = od_model.to(device)
    if model_path is not None:
        od_model.load_state_dict(torch.load(model_path))
        od_model.train()
    print(od_model)

    # optimizer = torch.optim.SGD(od_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(od_model.parameters(), lr=1e-4, weight_decay=5e-4)

    loss_fn = torch.nn.MSELoss()

    loss = 0
    for epoch in range(50):
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            images = images.permute(0, 3, 1, 2)
            coords = od_model(images)
        
            loss0 = loss_fn(coords, labels)
            loss = loss * 0.9 + loss0.cpu().detach().numpy() * 0.1
            print(epoch, batch, loss)
            optimizer.zero_grad()
            loss0.backward()
            optimizer.step()
        
            if batch % 150 == 0:
                torch.save(od_model.state_dict(), f"{model_dir}/model_{epoch}_{batch}.pth")


def eval(model_path, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    od_model = ODModel()
    od_model = od_model.to(device)
    od_model.load_state_dict(torch.load(model_path))
    od_model.eval()
    print(od_model)

    with torch.no_grad():
        image = process(image_path)
        image = image.to(device)
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        coord = od_model(image)
    
        image = image.permute(0, 2, 3, 1)
        image = image.squeeze(0)
        image = image.cpu()
        coord = coord.squeeze(0)

        draw(image, coord, "test.jpg")


if __name__ == "__main__":
    # test_dataset()
    # train(model_path="checkpoint/model_49_150.pth", model_dir="checkpoint1")
    eval(model_path="checkpoint/model_44_150.pth", image_path="/workspace/od/data/VOCdevkit/VOC2007/JPEGImages/000017.jpg")
    # eval(model_path="checkpoint/model_44_150.pth", image_path="/workspace/od/data/VOCdevkit/VOC2007/JPEGImages/000055.jpg")
    # eval(model_path="checkpoint/model_44_0.pth", image_path="dog.jpg")

