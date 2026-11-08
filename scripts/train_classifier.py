"""Train the defect-type classifier and export ONNX (run on the laptop GPU).

Data: an ImageFolder tree, one sub-folder per defect type, e.g. built from
NEU-CLS (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)
plus MVTec AD / DeepPCB defect crops:
    data/cls/train/<label>/*.png   data/cls/val/<label>/*.png

    python scripts/train_classifier.py --data data/cls --out models/defect_cls.onnx
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, models, transforms

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="models/defect_cls.onnx")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
a = ap.parse_args()

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3), transforms.ToTensor(), norm])
tf_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
tr = datasets.ImageFolder(Path(a.data) / "train", tf_train)
va = datasets.ImageFolder(Path(a.data) / "val", tf_val)
assert tr.classes == va.classes, "train/val label folders differ"
dl_tr = torch.utils.data.DataLoader(tr, a.bs, shuffle=True, num_workers=4)
dl_va = torch.utils.data.DataLoader(va, a.bs, num_workers=4)

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(tr.classes))
m = m.to(dev)
opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

best, out = 0.0, Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
for ep in range(a.epochs):
    m.train()
    for x, y in dl_tr:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss_fn(m(x), y).backward()
        opt.step()
    sched.step()
    m.eval()
    ok = n = 0
    with torch.no_grad():
        for x, y in dl_va:
            ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item()
            n += y.numel()
    acc = ok / n
    print(f"epoch {ep + 1}/{a.epochs} val_acc={acc:.4f}")
    if acc > best:
        best = acc
        torch.save(m.state_dict(), out.with_suffix(".pt"))

m.load_state_dict(torch.load(out.with_suffix(".pt")))
m.eval().cpu()
torch.onnx.export(m, torch.zeros(1, 3, 224, 224), str(out), input_names=["input"], output_names=["logits"],
                  opset_version=17, dynamo=False)
out.with_suffix(".labels.json").write_text(json.dumps(tr.classes))
print(f"best val_acc={best:.4f} -> {out}")
