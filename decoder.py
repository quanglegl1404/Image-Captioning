import torch.nn as nn
import torchvision.models as models
from config import Config
import torch

class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):

        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
    
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

####################
# Attention Decoder
####################
class Decoder(nn.Module):

    def __init__(self, vocab_size, use_glove, use_bert, vocab, device, BertTokenizer, BertModel):
        super(Decoder, self).__init__()
        self.config = Config()
        self.encoder_dim = self.config.encoder_dim
        self.attention_dim = self.config.attention_dim
        self.use_bert = use_bert

        self.vocab = vocab
        self.device = device
        self.BertTokenizer = BertTokenizer
        self.BertModel = BertModel
        # self.device = device

        if use_glove:
            self.embed_dim = self.config.glove_embed_dim
        elif use_bert:
            self.embed_dim = self.config.bert_embed_dim
        else:
            self.embed_dim = self.config.embed_dim

        self.decoder_dim = self.config.decoder_dim
        self.vocab_size = vocab_size
        self.decoder_dim = self.config.decoder_dim
        self.dropout = self.config.dropout
        
        # soft attention

        self.attention = Attention(encoder_dim = self.encoder_dim, decoder_dim = self.decoder_dim, attention_dim = self.attention_dim)
        # self.enc_att = nn.Linear(2048, 512)
        # self.dec_att = nn.Linear(512, 512)
        # self.att = nn.Linear(512, 1)
        # self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

        # decoder layers
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        # init variables
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
        if not use_bert:
            print("Using Baseline model")
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

            # # load Glove embeddings
            # if use_glove:
            #     self.embedding.weight = nn.Parameter(glove_vectors)

            # always fine-tune embeddings (even with GloVe)
            for p in self.embedding.parameters():
                p.requires_grad = True
        else:
            print("Using Bert Model")
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """

        mean_encoder_out = encoder_out.mean(dim = 1)
        h = self.h_lin(mean_encoder_out)
        c = self.c_lin(mean_encoder_out)
        return h,c

    def forward(self, encoder_out, encoded_captions, caption_lengths):    
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        dec_len = [x-1 for x in caption_lengths]
        max_dec_len = max(dec_len)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        tokenizer = self.BertTokenizer
        # tokenizer = self.BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = True, strip_accents=False)
        # load bert or regular embeddings
        if not self.use_bert:
            embeddings = self.embedding(encoded_captions)
        elif self.use_bert:
            embeddings = []
            for cap_idx in  encoded_captions:
                
                # padd caption to correct size
                # while len(cap_idx) < max_dec_len:
                #     cap_idx.append(self.config.PAD)
                    
                cap = ' '.join([self.vocab.idx2word[word_idx.item()] for word_idx in cap_idx])
                #print(f"Cap: {cap}")
                cap = u'[CLS] '+cap

                tokenized_cap = tokenizer.tokenize(cap.lower())      
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_cap)
                tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

                with torch.no_grad():
                    encoded_layers, _ = self.BertModel(tokens_tensor)

                bert_embedding = encoded_layers[-1].squeeze(0)
                
                split_cap = cap.split()
                tokens_embedding = []
                j = 0
                try:
                    for full_token in split_cap:
                        curr_token = ''
                        x = 0
                        for i,_ in enumerate(tokenized_cap[1:]): # disregard CLS
                            token = tokenized_cap[i+j]
                            piece_embedding = bert_embedding[i+j]
                            
                            # full token
                            if token == full_token and curr_token == '' :
                                tokens_embedding.append(piece_embedding)
                                j += 1
                                break
                            else: # partial token
                                x += 1
                                
                                if curr_token == '':
                                    tokens_embedding.append(piece_embedding)
                                    curr_token += token.replace('#', '')
                                else:
                                    tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                                    curr_token += token.replace('#', '')
                                    
                                    if curr_token == full_token: # end of partial
                                        j += x
                                        break
                except Exception as ex:
                    print(f"Exception for token: {curr_token}")
                    pass                       
                cap_embedding = torch.stack(tokens_embedding)
                embeddings.append(cap_embedding)

  
            embeddings = torch.stack(embeddings)
            #print(f"Embeddings after stack: {embeddings}")

        # init hidden state
        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(batch_size, max_dec_len, vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max_dec_len, num_pixels).to(self.device)

        for t in range(max(dec_len)):
            batch_size_t = sum([l > t for l in dec_len ])
            
            # soft-attention
            # enc_att = self.enc_att(encoder_out[:batch_size_t])
            # dec_att = self.dec_att(h[:batch_size_t])
            # att = self.att(self.relu(enc_att + dec_att.unsqueeze(1))).squeeze(2)
            # alpha = self.softmax(att)
            # attention_weighted_encoding = (encoder_out[:batch_size_t] * alpha.unsqueeze(2)).sum(dim=1)

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
        
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            batch_embeds = embeddings[:batch_size_t, t, :]            
            cat_val = torch.cat([batch_embeds.double(), attention_weighted_encoding.double()], dim=1)
            
            h, c = self.decode_step(cat_val.float(),(h[:batch_size_t].float(), c[:batch_size_t].float()))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        # preds, sorted capts, dec lens, attention wieghts
        return predictions, encoded_captions, dec_len, alphas