#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import torch
import torch.nn as nn
from . import layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        question_emb_dim = args.embedding_dim
        if args.use_charemb and args.before_qemb:
            question_emb_dim += args.charemb_dim

        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(question_emb_dim)
        
        if args.use_charemb and not args.before_qemb:
            question_emb_dim += args.charemb_dim

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim
        
        if args.use_charemb:
            doc_input_size += args.charemb_dim

        if args.use_highway:
            self.doc_highway = layers.HighwayLayer(
                input_size=doc_input_size,
                drop_h=args.drop_h,
                drop_t=args.drop_t,
                dropout_rate=args.dropout_highway
            )

        # RNN document encoder
        if self.args.use_cnn:
            self.doc_rnn = layers.StackedConvolutions(
                input_size=doc_input_size,
                hidden_size=args.hidden_size,
                num_layers=args.doc_layers,
                kernel_size=args.kernel_size,
                dropout_rate=args.dropout_rnn,
                dropout_output=args.dropout_rnn_output,
                concat_layers=args.concat_rnn_layers,
                is_separated=args.is_separated,
                is_dilated=args.is_dilated,
                use_highway=args.highway_with_cnn
            )
        elif self.args.use_transformer:
            self.doc_rnn = nn.modules.TransformerEncoder(
                encoder_layer=nn.modules.TransformerEncoderLayer(
                    d_model=doc_input_size,
                    nhead=args.nhead,
                    dim_feedforward=args.dim_feedforward,
                    dropout=args.dropout_rnn
                ),
                num_layers=args.doc_layers
            )
        else:
            self.doc_rnn = layers.StackedBRNN(
                input_size=doc_input_size,
                hidden_size=args.hidden_size,
                num_layers=args.doc_layers,
                dropout_rate=args.dropout_rnn,
                dropout_output=args.dropout_rnn_output,
                concat_layers=args.concat_rnn_layers,
                rnn_type=self.RNN_TYPES[args.rnn_type],
                padding=args.rnn_padding,
            )


        if args.use_highway:
            self.question_highway = layers.HighwayLayer(
                input_size=question_emb_dim,
                drop_h=args.drop_h,
                drop_t=args.drop_t,
                dropout_rate=args.dropout_highway
            )

        # RNN question encoder
        if self.args.use_cnn:
            self.question_rnn = layers.StackedConvolutions(
                input_size=args.embedding_dim,
                hidden_size=args.hidden_size,
                num_layers=args.question_layers,
                kernel_size=args.kernel_size,
                dropout_rate=args.dropout_rnn,
                dropout_output=args.dropout_rnn_output,
                concat_layers=args.concat_rnn_layers,
                is_separated=args.is_separated,
                is_dilated=args.is_dilated,
                use_highway=args.highway_with_cnn
            )
        elif self.args.use_transformer:
            self.question_rnn = nn.modules.TransformerEncoder(
                encoder_layer=nn.modules.TransformerEncoderLayer(
                    d_model=args.embedding_dim,
                    nhead=args.nhead,
                    dim_feedforward=args.dim_feedforward,
                    dropout=args.dropout_rnn
                ),
                num_layers=args.question_layers
            )
        else:
            self.question_rnn = layers.StackedBRNN(
                input_size=args.embedding_dim,
                hidden_size=args.hidden_size,
                num_layers=args.question_layers,
                dropout_rate=args.dropout_rnn,
                dropout_output=args.dropout_rnn_output,
                concat_layers=args.concat_rnn_layers,
                rnn_type=self.RNN_TYPES[args.rnn_type],
                padding=args.rnn_padding,
            )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers
        if args.use_transformer:
            doc_hidden_size = doc_input_size
            question_hidden_size = args.embedding_dim

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

        if args.use_charemb:
            self.char_emb = nn.Embedding(
                len(args.characters) + 1,
                args.charemb_dim,
                padding_idx=0
            )


    def forward(self, x1, x1_f, x1_mask, x2, x2_mask, x1_char_emb, x2_char_emb):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        char_emb = question char indices       [batch * len_d * len_w]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]

        if self.args.use_charemb:
            x1_char_emb = self.char_emb(x1_char_emb)
            x1_char_emb, _ = torch.max(x1_char_emb, -2)

            x2_char_emb = self.char_emb(x2_char_emb)
            x2_char_emb, _ = torch.max(x2_char_emb, -2)

            if self.args.dropout_emb > 0:
                x1_char_emb = nn.functional.dropout(x1_char_emb, p=self.args.dropout_emb,
                                           training=self.training)
                x2_char_emb = nn.functional.dropout(x2_char_emb, p=self.args.dropout_emb,
                                           training=self.training)

            if self.args.before_qemb:
                x1_emb = torch.cat([x1_emb, x1_char_emb], 2)
            else:
                drnn_input.append(x1_char_emb)

            if self.args.before_qemb:
                x2_emb = torch.cat([x2_emb, x2_char_emb], 2)
        
        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN
        doc_input = torch.cat(drnn_input, 2)
        if self.args.use_highway:
            doc_input = self.doc_highway(doc_input)
        doc_hiddens = self.doc_rnn(doc_input, x1_mask)

        # Encode question with RNN + merge hiddens
        if self.args.use_charemb and not self.args.before_qemb:
            x2_emb = torch.cat([x2_emb, x2_char_emb], 2)
        if self.args.use_highway:
            x2_emb = self.question_highway(x2_emb)
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores
