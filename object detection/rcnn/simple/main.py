import random
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


def process():
    image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    image = Image.fromarray(image)

    x1, y1, x2, y2 = random.randint(0, 223), random.randint(0, 223), random.randint(0, 223), random.randint(0, 223)
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    cx, cy, cw, ch = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    r, g, b = random.randint(0, 224), random.randint(0, 224), random.randint(0, 224)

    if cw == 0 or ch == 0:
        return None

    draw = ImageDraw.Draw(image)
    draw.ellipse((x1, y1, x2, y2), fill=(r, g, b))

    image = np.array(image)
    image = torch.from_numpy(image)
    image = preprocess(image)

    label = torch.tensor([cx, cy, cw, ch])
    label = convert1(label)

    return image, label


def draw(image, label):
    image = deprocess(image)
    image = image.numpy()
    image = Image.fromarray(image)

    label = convert2(label)
    cx, cy, cw, ch = label
    cx, cy, cw, ch = cx.numpy(), cy.numpy(), cw.numpy(), ch.numpy()

    draw = ImageDraw.Draw(image)
    draw.rectangle(((cx - cw / 2, cy - ch / 2), (cx + cw / 2, cy + ch / 2)), outline=(0, 0, 0), width=1)
    image.show()


class ODDataset(Dataset):

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        while True:
            r = process()
            if r is not None:
                return r


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
    image, label = process()
    draw(image, label)


def train(model_path=None, model_dir="checkpoint"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    train_dataset = ODDataset(10000)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(len(train_dataset))

    od_model = ODModel()
    od_model = od_model.to(device)
    if model_path is not None:
        od_model.load_state_dict(torch.load(model_path))
        od_model.train()
    print(od_model)

    optimizer = torch.optim.SGD(od_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(od_model.parameters(), lr=1e-4, weight_decay=5e-4)

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


def eval(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    od_model = ODModel()
    od_model = od_model.to(device)
    od_model.load_state_dict(torch.load(model_path))
    od_model.eval()
    print(od_model)

    with torch.no_grad():
        image, _ = process()
        image = image.to(device)
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        coord = od_model(image)

        image = image.permute(0, 2, 3, 1)
        image = image.squeeze(0)
        image = image.cpu()
        coord = coord.squeeze(0)

        draw(image, coord)


if __name__ == "__main__":
    # test_dataset()
    # train(model_path="checkpoint1/model_6_150.pth", model_dir="checkpoint2")
    eval("checkpoint1/model_6_150.pth")

