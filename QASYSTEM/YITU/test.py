from config import *
from utils import *
from model import *

if __name__ == '__main__':

    id2label, _ = get_label()

    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = TextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + 'model_30.pth', map_location=DEVICE))
    loss_fn = nn.CrossEntropyLoss()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for b, (input, mask, target) in enumerate(test_loader):

            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            test_pred = model(input, mask)
            loss = loss_fn(test_pred, target)

            print('>> batch:', b, 'loss:', round(loss.item(), 5))

            test_pred_ = torch.argmax(test_pred, dim=1)

            y_pred += test_pred_.data.tolist()
            y_true += target.data.tolist()

    print(evaluate(y_pred, y_true, id2label))