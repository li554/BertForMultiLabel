import torch
from tqdm import tqdm
from bert_multi_label_model import BertForMultiLabel
from bert_processor import BertProcessor


class Model:
    def __init__(self, model_name="models/checkpoint-4/", vocab_path="base-uncased/vocab.txt"):
        self.model = BertForMultiLabel.from_pretrained(model_name)
        self.model.eval()
        self.model.cuda()
        self.processor = BertProcessor(vocab_path, do_lower_case=True, max_seq_length=256)

    def predict_batch(self, input_ids):
        with torch.no_grad():
            input_ids = input_ids.cuda()
            output = self.model(input_ids)
            return output

    def predict_iter(self, test_iter):
        preds = None
        for input_ids, input_mask, segment_ids, label_ids in tqdm(test_iter):
            if preds is None:
                preds = self.predict_batch(input_ids)
            else:
                preds = torch.cat((preds, self.predict_batch(input_ids)), dim=0)
        return preds

    def predict_sen(self, sentences):
        return self.predict_batch(self.processor.encode(sentences))

    def predict(self, sentence):
        sentences = [sentence]
        results = self.predict_sen(sentences)
        results = torch.where(results > 0.5, 1, 0)
        labels = self.processor.get_labels()
        batch_preds = []
        for label_ids in results:
            preds = []
            for i, label_id in enumerate(label_ids):
                if label_id == 1:
                    preds.append(labels[i])
            batch_preds.append(preds)
        return batch_preds
