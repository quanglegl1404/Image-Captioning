import torch.nn as nn
import torchvision.models as models
from config import Config
import torch
import torch.nn.functional as F

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
        print("call forward")   
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
        print(f"Max decoder length: {max(dec_len)}")
        print(f"Decoder length: {dec_len}")

        for t in range(max(dec_len)):
            print(f"t: {t},{[l > t for l in dec_len ]}")
            batch_size_t = sum([l > t for l in dec_len ])
            print(f"batch size t: {batch_size_t}")

            # print(f"hidden state: {h[:batch_size_t]}")
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
        
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            batch_embeds = embeddings[:batch_size_t, t, :]        
            cat_val = torch.cat([batch_embeds.double(), attention_weighted_encoding.double()], dim=1)
            
            h, c = self.decode_step(cat_val.float(),(h[:batch_size_t].float(), c[:batch_size_t].float()))
            preds = self.fc(self.dropout(h))
            # print(f"prediction: {preds}")
            # print(f"torch max score: {torch.max(preds)}")
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        # preds, sorted capts, dec lens, attention wieghts
        return predictions, encoded_captions, dec_len, alphas

    def sample(self, encoder_out, beam_size=3):
        k = beam_size
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.vocab.word2idx['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != self.vocab.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

        return seq, alphas