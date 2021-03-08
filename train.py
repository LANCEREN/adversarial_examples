import os

import misc
import model
import dataset

import torch
import torch.nn.functional as F


def train(model, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        pred = model(data)
        loss = F.nll_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Train Epoch: {}, iterantion: {}, Loss: {}".format(epoch, idx, loss.item()))


def test(model, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

        total_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset) * 100
        print("Test loss: {}, Accuracy: {}".format(total_loss, acc))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(
    'nccl',
    init_method='env://'
)
device = torch.device(f'cuda:{args.local_rank}')


num_eopchs = 200
batch_size = 128
lr = 0.01
momentum = 0.5

# gpu = misc.auto_select_gpu(
#         utility_bound=0,
#         num_gpu=1,
#         selected_gpus=None)
# ngpu = len(gpu)
cuda = torch.cuda.is_available()

train_loader, test_loader = dataset.generate_dataset(batch_size=batch_size)
model = model.LeNet()
#model = torch.nn.DataParallel(model, device_ids=range(ngpu))
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


model.to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)

# train
for eopch in range(num_eopchs):
    train(model, train_loader, optimizer, eopch)
    test(model, test_loader)

torch.save(model.state_dict(), "mnist_lenet.pt")