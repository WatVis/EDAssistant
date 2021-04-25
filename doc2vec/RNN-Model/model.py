import torch.nn as nn
import torch    

# BertModel from https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch
# Licensed under the MIT License. Copyright (c) Microsoft Corporation.
class BertModel(nn.Module):   
    def __init__(self, encoder):
        super(BertModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

# RNN generator to generate embedding
# TODO: try other arch https://arxiv.org/pdf/2005.09471.pdf
class Generator(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(Generator, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.gru_unit = nn.GRU(self.embed_size, self.hidden_size, num_layers=1)

        # self.attn = multi_attention(in_size=self.hidden_size, hidden_size=self.hidden_size, n_heads=10)
        self.trans = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.embed_size)
        )

    def forward(self, data, lengths, criterion, device="cuda"):
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(data, lengths)
        packed_data, hiddens = self.gru_unit(packed_data)
        unpacked_data, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_data)
        # unpacked_data = self.attn(unpacked_data)
        unpacked_data = self.trans(unpacked_data)


        loss = torch.FloatTensor([0]).to(device)
        for idx in range(len(lengths)):
          l = lengths[idx]

          predict_data = unpacked_data[:, idx, :][:l][:-1]
          tgt_data = data[:, idx, :][:l][1:]
          
          loss += criterion(predict_data, tgt_data)
        return loss

    def generate_embedding(self, embed_list):
      embed = self.trans(self.gru_unit(torch.cat(embed_list).view(-1, 1, 768))[0])[-1]
      return embed
 
