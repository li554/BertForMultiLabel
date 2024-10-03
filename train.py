import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from bert_multi_label_model import BertForMultiLabel
from bert_processor import BertProcessor
from metrics import Metric


def evaluate(model, valid_iter, loss, writer, step):
    total_val_loss, total_val_acc, total_mean_auc = 0.0, 0.0, 0.0
    # 每16个batch计算一次val_loss,val_acc,val_auc,val_mean_auc
    temp_logits = None
    temp_labels = None
    all_logits = None
    all_labels = None

    count = 0
    total_count = len(valid_iter)
    batch_count = 0
    auc_count = 0
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, label_ids in tqdm(valid_iter):
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            logits = model(input_ids, segment_ids, input_mask)
            label_ids = label_ids.float().cuda()
            temp_logits = logits if temp_logits is None else torch.cat((temp_logits, logits), dim=0)
            temp_labels = label_ids if temp_labels is None else torch.cat((temp_labels, label_ids), dim=0)
            if count == 15:
                all_logits = temp_logits if all_logits is None else torch.cat((all_logits, temp_logits), dim=0)
                all_labels = temp_labels if all_labels is None else torch.cat((all_labels, temp_labels), dim=0)

                temp_val_loss = loss(temp_logits, temp_labels).item()
                metric = Metric(temp_logits.cpu().detach().numpy(), temp_labels.cpu().detach().numpy())
                temp_val_all_acc = metric.accuracy_all()
                temp_val_mean_acc = metric.accuracy_mean()
                temp_val_auc = metric.auc()
                # 绘制
                writer.add_scalar('val loss', temp_val_loss, global_step=step)
                writer.add_scalar('val mean acc', temp_val_mean_acc, global_step=step)
                writer.add_scalar('val all acc', temp_val_all_acc, global_step=step)
                if temp_val_auc is not None:
                    auc_count += 1
                    temp_val_mean_auc = temp_val_auc.mean()
                    total_mean_auc += temp_val_mean_auc
                    writer.add_scalar('mean auc score', temp_val_mean_auc, global_step=step)
                    label_list = processor.get_labels()
                    for i in range(len(label_list)):
                        writer.add_scalar(label_list[i] + ' auc score', temp_val_auc[i], global_step=step)
                total_val_acc += temp_val_mean_acc
                total_val_loss += temp_val_loss
                step += 1
                batch_count += 1
                # 重置
                temp_logits = None
                temp_labels = None
                count = 0
            count += 1

    return total_val_acc / batch_count, total_val_loss / total_count, total_mean_auc / auc_count, step


# 训练
def train(model, train_iter, valid_iter, n_epoch, loss, optimizer, scheduler):
    writer = SummaryWriter('runs/exp')
    train_step = 0
    val_step = 0
    for epoch in range(init_epoch, n_epoch):
        pbar = tqdm(train_iter)
        for input_ids, input_mask, segment_ids, label_ids in pbar:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            logits = model(input_ids, segment_ids, input_mask)
            l = loss(logits, label_ids.float().cuda())
            l.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            metric = Metric(logits.cpu().detach().numpy(), label_ids.cpu().detach().numpy())
            writer.add_scalar('train loss', l.item(), global_step=train_step)
            writer.add_scalar('train all acc', metric.accuracy_all(), global_step=train_step)
            writer.add_scalar('train mean acc', metric.accuracy_mean(), global_step=train_step)
            train_step += 1
        model.save_pretrained("output/checkpoint-%s/" % (epoch + 1))
        val_acc, val_loss, val_mean_auc, val_step = evaluate(model, valid_iter, loss, writer, val_step)
        print("epoch %s  val loss:%s val acc:%s auc:%s" % (epoch + 1, val_loss, val_acc, val_mean_auc))


if __name__ == '__main__':
    # 定义超参数
    init_epoch = 2
    batch_size = 16
    lr = 2e-5
    weight_decay = 0.01
    warmup_proportion = 0.1
    adam_epsilon = 1e-8
    grad_clip = 1
    start_layer = 0  # [0,11]
    end_layer = 11  # [start_layer,end_layer]
    n_epoch = 6

    # 加载模型
    if init_epoch == 0:
        model_name = "models/base-uncased"
    else:
        model_name = "output/checkpoint-%s" % init_epoch
    model = BertForMultiLabel.from_pretrained(model_name)
    model.unfreeze(start_layer, end_layer)
    model = model.cuda()
    # 加载数据集
    vocab_path = "models/base-uncased/vocab.txt"
    processor = BertProcessor(vocab_path, do_lower_case=True, max_seq_length=256)
    dataset = processor.read_dataset("dataset/train.csv")
    train_iter, valid_iter = processor.train_val_split(dataset, batch_size)

    # 定义损失函数
    loss = nn.BCELoss()

    # 定义优化器
    t_total = int(len(train_iter) * n_epoch)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * warmup_proportion)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    train(model, train_iter, valid_iter, n_epoch, loss, optimizer, scheduler)
