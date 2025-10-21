from config import *
from utils import *
from model import *
from model_dev import dev
import random
import numpy as np
from sklearn.metrics import accuracy_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # 设置随机种子
    SEED = 42  # 可以设置为任意整数
    set_seed(SEED)
    id2label, _ = get_label()

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(EPOCH):
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input, mask)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算当前 batch 的 P/R/F1
            y_pred = torch.argmax(pred, dim=1)
            report = evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), id2label, output_dict=True)

        # 验证集评估
        dev_report = dev(model)

        # 输出训练集（最后一个 batch）和验证集的 P/R/F1/Accuracy
        print(
            f">> epoch: {e}",
            f"batch: {b}",
            f"loss: {round(loss.item(), 5)}",
            f"A: {dev_report['accuracy']:.4f}",  # 添加准确率
            f"P: {dev_report['macro avg']['precision']:.4f}",
            f"R: {dev_report['macro avg']['recall']:.4f}",
            f"F1: {dev_report['macro avg']['f1-score']:.4f}",
        )
