from torch import nn
from transformers import BertModel, BertPreTrainedModel


class BertForMultiLabel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLabel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)

    def unfreeze(self, start_layer, end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())

        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b

        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)

        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer + 1):
            set_trainable(self.bert.encoder.layer[i], True)
