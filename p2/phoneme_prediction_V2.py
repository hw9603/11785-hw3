import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import Levenshtein as L
from ctcdecode import CTCBeamDecoder
from phoneme_list import PHONEME_MAP


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
INTERVAL = 10
PHONEME_SIZE = 40
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 3
BEAM_WIDTH = 100


class PhonemeDataset(Dataset):
    def __init__(self, x, y=None, is_test=False):
        self.is_test = is_test
        self.x = [torch.tensor(x_) for x_ in x]
        if not is_test:
            self.total_phonemes = sum(len(y_) for y_ in y)
            self.y = [torch.tensor(y_) for y_ in y]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if not self.is_test:
            return self.x[item].to(DEVICE), (self.y[item] + 1).to(DEVICE)
        else:
            return self.x[item].to(DEVICE), torch.tensor([-1]).to(DEVICE)


def collate_phonemes(seq_list):
    seq_list = sorted(seq_list, key=lambda seq: seq[0].shape[0], reverse=True)
    x = []
    y = []
    x_lens = []
    y_lens = []
    for i, (x_, y_) in enumerate(seq_list):
        x.append(x_)
        y.append(y_)
        x_lens.append(x_.shape[0])
        y_lens.append(y_.shape[0])
    return x, y, x_lens, y_lens


class PhonemeModel(nn.Module):
    def __init__(self):
        super(PhonemeModel, self).__init__()
        self.phoneme_size = PHONEME_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS

        self.rnn = nn.LSTM(input_size=self.phoneme_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=False)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=2 * self.hidden_size, out_features=len(PHONEME_MAP)+1)
        )

        self.init_weights()

    def forward(self, features, lens):
        batch_size = len(features)
        packed_input = rnn.pack_sequence(features)

        hidden = None
        output_packed, hidden = self.rnn(packed_input.float().to(DEVICE), hidden)

        output_padded, _ = rnn.pad_packed_sequence(output_packed)
        # print(output_padded.shape)
        # output_flatten = torch.cat([output_padded[:lens[i], i] for i in range(batch_size)])
        scores_flatten = self.output_layer(output_padded).log_softmax(2)
        return scores_flatten

    def init_weights(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight.data)
            elif type(m) == nn.LSTM:
                torch.nn.init.xavier_normal_(m.weight_hh_l0)


class ER:
    def __init__(self):
        self.label_map = [' '] + PHONEME_MAP
        self.decoder = CTCBeamDecoder(
            labels=self.label_map,
            blank_id=0,
            beam_width=BEAM_WIDTH
        )

    def __call__(self, prediction, seq_size, target=None):
        return self.forward(prediction, seq_size, target)

    def forward(self, prediction, seq_size, target=None):
        prediction = torch.transpose(prediction, 0, 1)
        prediction = prediction.cpu()
        probs = F.softmax(prediction, dim=2)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=torch.IntTensor(seq_size))

        pred = []
        for i in range(output.size(0)):
            pred.append("".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]]))

        if target is not None:
            true = []
            for t in target:
                true.append("".join(self.label_map[o] for o in t))

                ls = 0.
                for p, t in zip(pred, true):
                    ls += L.distance(p.replace(" ", ""), t.replace(" ", ""))
                print("PER:", ls * 100 / sum(len(s) for s in true))
                return ls
        else:
            return pred


def train(train_loader, dev_loader, model, optimizer, e):
    model.train()
    model.to(DEVICE)

    # ctc = CTCCriterion()
    ctc = CTCLoss(reduction='none')
    avg_loss = 0.0
    t = time.time()
    print("epoch", e)
    epoch_loss = 0
    for batch_idx, (data_batch, label_batch, input_lengths, target_lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        # data_batch = data_batch.to(DEVICE)
        logits = model(data_batch, input_lengths)
        # print(logits)  # shape: [max_seq_len, batch_size, 47]
        loss = ctc(logits, torch.cat(label_batch).int(), torch.IntTensor(input_lengths), torch.IntTensor(target_lengths))
        loss.mean().backward()
        optimizer.step()
        epoch_loss += loss.mean().item()
        avg_loss += loss.mean().item()
        if batch_idx % INTERVAL == INTERVAL - 1:
            print("[Train Epoch %d] batch_idx=%d [%.2f%%, time: %.2f min], loss=%.4f" %
                  (e, batch_idx, 100. * batch_idx / len(train_loader), (time.time() - t) / 60,
                   avg_loss / INTERVAL))
            torch.save(model.state_dict(), "models/checkpoint.pt")
            avg_loss = 0.0

    print("Loss: {}".format(epoch_loss / len(train_loader)))


def eval(loader, model):
    model.eval()
    model.to(DEVICE)
    error_rate_op = ER()
    error = 0
    for data_batch, labels_batch, input_lengths, target_lengths in loader:
        # data_batch = data_batch.to(DEVICE)
        predictions_batch = model(data_batch, input_lengths)
        error += error_rate_op(predictions_batch, input_lengths, labels_batch)
    print("total error: ", error / loader.dataset.total_phonemes)
    return error / loader.dataset.total_phonemes


def prediction(loader, model, output_file):
    model.eval()
    model.to(DEVICE)

    fwrite = open(output_file, "w")
    fwrite.write("Id,Predicted\n")
    line = 0
    error_rate_op = ER()
    for data_batch, _, input_lengths, _ in loader:
        predictions_batch = model(data_batch, input_lengths)
        decode_strs = error_rate_op(predictions_batch, input_lengths)
        for s in decode_strs:
            if line % INTERVAL == 0:
                print(line, s)
            fwrite.write(str(line) + "," + s + "\n")
            line += 1
    return


def main():
    print(DEVICE)
    print("loading data...")
    train_x = np.load(train_x_file, encoding='bytes')
    train_y = np.load(train_y_file, encoding='bytes')
    dev_x = np.load(dev_x_file, encoding='bytes')
    dev_y = np.load(dev_y_file, encoding='bytes')
    test_x = np.load(test_x_file, encoding='bytes')

    train_dataset = PhonemeDataset(train_x, train_y, is_test=False)
    dev_dataset = PhonemeDataset(dev_x, dev_y, is_test=False)
    test_dataset = PhonemeDataset(test_x, y=None, is_test=True)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_phonemes)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_phonemes)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_phonemes)

    print("begin running!")
    model = PhonemeModel()
    model.load_state_dict(torch.load("models/finetune00.006995144978048452.pt"))
    model.to(DEVICE)
    # for param in model.parameters():
    #     param.requires_grad = False

    # model.rnn = nn.LSTM(input_size=PHONEME_SIZE, hidden_size=512, num_layers=3, bidirectional=True, batch_first=False)
    # model.output_layer = nn.Sequential(
    #     nn.Linear(in_features=1024, out_features=len(PHONEME_MAP) + 1)
    # )
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # train(train_loader, dev_loader, model, optimizer, -1)
    # per = eval(dev_loader, model)
    # torch.save(model.state_dict(), "models/finetune" + str(per) + ".pt")

    # for param in model.parameters():
    #     param.requires_grad = True

    for e in range(EPOCHS):
        train(train_loader, dev_loader, model, optimizer, e)
        # per = eval(dev_loader, model)
        torch.save(model.state_dict(), "models/finetune" + str(e) + ".pt")
        prediction(test_loader, model, "submission_" + str(e) + ".csv")

    # prediction(test_loader, model, "submission.csv")


if __name__ == "__main__":
    data_path = "../hw3p2-data-V2/"
    train_x_file = data_path + "wsj0_train.npy"
    train_y_file = data_path + "wsj0_train_merged_labels.npy"
    dev_x_file = data_path + "wsj0_dev.npy"
    dev_y_file = data_path + "wsj0_dev_merged_labels.npy"
    test_x_file = data_path + "transformed_test_data.npy"
    main()