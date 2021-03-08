import os
import time
import argparse



import misc
import model
import dataset

import torch
import torch.nn.functional as F
from torch import distributed as dist


def train(model, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        pred = model(data)
        loss = F.nll_loss(pred, target)
        reduced_loss = reduce_tensor(loss.data)
        epoch_loss += reduced_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0 and dist.get_rank() == 0 :
            print("Train Epoch: {}, iterantion: {}, Loss: {}".format(epoch, idx, epoch_loss.item()))


def test(model, test_loader):
    model.eval()
    total_loss = 0.
    total_correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = F.nll_loss(output, target, reduction="sum")
            reduced_loss = reduce_tensor(loss)
            total_loss += reduced_loss

            pred = output.argmax(dim=1)
            correct = pred.eq(target.view_as(pred)).sum().item()
            total_correct += correct

        total_loss /= len(test_loader.dataset)
        acc = total_correct / len(test_loader.dataset) * 100

        if dist.get_rank() == 0:
            print("Test loss: {}, Accuracy: {}".format(total_loss, acc))


def reduce_tensor(tensor):
    with torch.no_grad():

        dist.reduce(tensor, dst=0)
        if dist.get_rank() == 0:
            tensor /= dist.get_world_size()

    return tensor
import torch.distributed.launch

time_ = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
device = torch.device(f'cuda:{args.local_rank}')
torch.distributed.init_process_group(
    'nccl',
    init_method='env://'
)

gpu = misc.auto_select_gpu(
        num_gpu=dist.get_world_size(),
        selected_gpus=None)

num_eopchs = 20
batch_size = 128
lr = 0.01
momentum = 0.5

train_loader, test_loader = dataset.generate_dataset(batch_size=batch_size)
model = model.LeNet()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
model.to(device)

model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)

# train
for eopch in range(num_eopchs):
    train(model, train_loader, optimizer, eopch)
    test(model, test_loader)
a = time.time()-time_
print("{:.2f}s".format(a))
torch.save(model.state_dict(), "mnist_lenet.pt")