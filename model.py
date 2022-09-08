import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear_layer_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeded = self.word_embedding(captions[:,:-1])

        input_embed = torch.cat((features.unsqueeze(dim=1),embeded),dim=1)

        lstm_output, _ = self.lstm(input_embed)
        output=self.linear_layer_out(lstm_output)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        generated_caption = []
        inputs = inputs.view(-1, 1, self.embed_size) # (batch_size, 1, embed_size)
        for i in range(max_len):
            hidden, states = self.lstm(inputs, states) # (batch_size, 1, hidden_size), _
            outputs = self.linear_layer_out(hidden.squeeze(1))  
            _, output = outputs.max(dim=1)                   
           
            # appending on list
            generated_caption.append(output.item())
            
            # for next iteration
            inputs = self.word_embedding(output).unsqueeze(1)
        return generated_caption