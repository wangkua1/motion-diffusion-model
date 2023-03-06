import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from gthmr.lib.models.vibe_component import TemporalEncoder
import ipdb


class MDM(nn.Module):

    def __init__(self,
                 modeltype,
                 njoints,
                 nfeats,
                 num_actions,
                 translation,
                 pose_rep,
                 glob,
                 glob_rot,
                 latent_dim=256,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=4,
                 dropout=0.1,
                 ablation=None,
                 activation="gelu",
                 legacy=False,
                 data_rep='rot6d',
                 dataset='amass',
                 clip_dim=512,
                 arch='trans_enc',
                 emb_trans_dec=False,
                 clip_version=None,
                 video_dim=2048,
                 video_cond_input='concat',
                 video_arch="linear",
                 feature_mask_ratio=0.0,
                 feature_mask_block_size=5,
                 video_arch_experiment=0,
                 **kargs):
        """
        video_cond_input ... it's assumed that the features are papssed through a single projection layer. 
            "concatenate_input": concat the features to the input noise vector. 
            "add_input": add features to the input noise.
        """
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.video_dim = video_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.video_cond_input = video_cond_input
        if (self.cond_mode == "video") and (video_cond_input
                                            not in ('concat', 'add')):
            assert arch == 'trans_dec', "Video conditioning must be added to the input or through transformer decoder or both"
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep,
                                          self.input_feats + self.gru_emb_dim,
                                          self.latent_dim)

        self.d_model = self.latent_dim*2  if (self.cond_mode=='video' and self.video_cond_input=='concat') \
                                            else self.latent_dim

        self.sequence_pos_encoder = PositionalEncoding(self.d_model,
                                                       self.dropout)
        self.emb_trans_dec = emb_trans_dec

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(
                seqTransEncoderLayer, num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(
                seqTransDecoderLayer, num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim,
                              self.latent_dim,
                              num_layers=self.num_layers,
                              batch_first=True)
        else:
            raise ValueError(
                'Please choose correct architecture [trans_enc, trans_dec, gru]'
            )

        if (self.cond_mode == 'video') and (self.video_cond_input == 'concat'):
            self.final_layer = nn.Linear(self.latent_dim * 2, self.latent_dim)
        else:
            self.final_layer = lambda x: x

        self.embed_timestep = TimestepEmbedder(self.d_model,
                                               self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions,
                                                self.latent_dim)
                print('EMBED ACTION')
            if 'video' in self.cond_mode:
                # feature masking parameters
                self.feature_mask_ratio = feature_mask_ratio
                assert 0 <= feature_mask_ratio <= 1, "mask ratio must be in [0,1]"
                self.feature_mask_block_size = feature_mask_block_size

                # projection video features. Random video feature masking is handled here.
                self.embed_video = EmbedVideo(
                    video_dim,
                    latent_dim,
                    arch=video_arch,
                    feature_mask_ratio=self.feature_mask_ratio,
                    feature_mask_block_size=self.feature_mask_block_size,
                    arch_experiment=video_arch_experiment)

                # case where we concat input AND cross-attention, need to project context embeddings to 2*latent_dim
                if self.arch == 'trans_dec' and self.video_cond_input:
                    self.embed_video_crossattn = nn.Linear(
                        latent_dim, 2 * latent_dim)
                print("EMBED VIDEO")

        self.output_process = OutputProcess(self.data_rep, self.input_feats,
                                            self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [
            p for name, p in self.named_parameters()
            if not name.startswith('clip_model.')
        ]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device='cpu',
            jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        # bs, d = cond.shape
        # if force_mask:
        #     return torch.zeros_like(cond)
        # elif self.training and self.cond_mask_prob > 0.:
        #     mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
        #     return cond * (1. - mask)
        # else:
        #     return cond
        ### JW TMP...
        return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in [
            'humanml', 'kit'
        ] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(
                raw_text, context_length=context_length, truncate=True
            ).to(
                device
            )  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y={}):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(
                self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)
        if 'video' in self.cond_mode:
            features = y['features'].cuda()
            shape = features.shape
            video_emb = self.embed_video(features)  # (N,T,D)
            video_emb = video_emb.permute(1, 0, 2)  # (T,N,D)
            ## do not add to `emb`

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)  #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)  #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1,
                                      nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru),
                          axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            if 'video' in self.cond_mode:
                if self.video_cond_input == "concat":
                    # video_emb = self.mask_cond(video_emb, force_mask=force_mask)
                    x = torch.cat((x, video_emb), axis=-1)
                elif self.video_cond_input == "add":
                    x = x + video_emb

            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

            output = self.seqTransEncoder(xseq)[
                1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            output = self.final_layer(output)

        elif self.arch == 'trans_dec':
            assert self.cond_mode == 'video'
            if self.video_cond_input == "concat":
                # video_emb = self.mask_cond(video_emb, force_mask=force_mask)
                x = torch.cat((x, video_emb), axis=-1)
                # case where we concat input AND cross-attention, need to project context embeddings to 2*latent_dim
                video_emb = self.embed_video_crossattn(video_emb)
            elif self.video_cond_input == "add":
                x = x + video_emb
            xseq = torch.cat(
                (emb, x), axis=0
            )  # [seqlen+1, bs, d] concat diffusion timestep `t` embedding
            xseq = self.sequence_pos_encoder(
                xseq)  # [seqlen+1, bs, d] additive positional embedding
            output = self.seqTransDecoder(tgt=xseq, memory=video_emb)[
                1:]  # [seqlen, bs, d] transformer+cross attention
            output = self.final_layer(
                output
            )  # if concat, this will project the dimsnion back to latent_dim

        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):

    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(
            self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):

    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep != 'rot_vel':
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        else:
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]

class OutputProcess(nn.Module):

    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep != 'rot_vel':
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        else:
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
            
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):

    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output


class EmbedVideo(nn.Module):

    def __init__(self,
                 video_dim,
                 latent_dim,
                 arch="linear",
                 feature_mask_ratio=0.0,
                 feature_mask_block_size=5,
                 arch_experiment=0):
        """
        Do a linear projection and 
            'arch': a single linear layer projection. 
            `feature_mask_ratio`: ratio of frame blocks to mask out when doing random sampling

        """
        super().__init__()
        self.arch=arch
        self.arch_experiment=arch_experiment
        self.feature_mask_ratio = feature_mask_ratio
        self.feature_mask_block_size = feature_mask_block_size

        if arch == "linear":
            self.enc = nn.Linear(video_dim, latent_dim)

        elif arch == 'mlp':
            hidden_size = video_dim
            self.enc = nn.Sequential(
                nn.Linear(video_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)  # the output layer
            )

        elif arch == 'trans_enc':
            # define the transformer encoder based on preset experiment vals
            arch_experiment_kwargs = {
            0 : dict(d_model=512, num_heads=4, ff_size=1024, num_layers=8),
            1 : dict(d_model=128, num_heads=4, ff_size=256, num_layers=4),
            2 : dict(d_model=64, num_heads=4, ff_size=128, num_layers=4),
            }
            kargs = arch_experiment_kwargs[arch_experiment]
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=kargs['d_model'],
                nhead=kargs['num_heads'],
                dim_feedforward=kargs['ff_size'],
                dropout=0.1,
                activation='gelu'
            )
            self.seqTransEncoderVideo = nn.TransformerEncoder(
                    seqTransEncoderLayer, num_layers=kargs['num_layers'])

            # combine the trans_enc with projection heads
            self.enc = nn.Sequential(
                nn.Linear(video_dim, kargs['d_model']),
                self.seqTransEncoderVideo,
                nn.Linear(kargs['d_model'], latent_dim)
            )
            #import ipdb; ipdb.set_trace()
            
        else:
            raise
        

    def get_feature_mask(self, x):
        """
        Randomly mask out video features in blocks along the time dimension.

        Sort of similar approach to https://github.com/facebookresearch/mae/blob/6a2ba402291005b003a70e99f7c87d1a2c376b0d/models_mae.py

        Returns: 
            mask shape (N,T)
        """
        N, T, D = x.shape
        if self.feature_mask_block_size >= T:
            print(
                f"Warning: masking won't work bc block size [self.feature_mask_block_size] bigger than sample frames [T]"
            )
        mask_features = torch.ones((N, T), dtype=bool)
        if self.feature_mask_ratio <= 0:
            return mask_features

        # Divide the features into evenly-sized blocks (truncates so that leftover elements are never masked)
        # The actual mask ratio will never be more than requested. E.g. 0.3 may actually become 0.27 bc of truncation
        L = T // self.feature_mask_block_size  # number of blocks that fit in this sequence
        len_mask = int(L *
                       self.feature_mask_ratio)  # number of blocks to mask out

        # for each batch index, generate a unique permutation of block indices
        noise = torch.rand(N, L)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: smaller is mask
        ids_mask_blocks = ids_shuffle[:, :len_mask]  # block indexes to mask

        # convert block indices to frame indices
        ids_mask_frames = ids_mask_blocks * self.feature_mask_block_size  # first element of each block
        ids_mask_frames_ = [ids_mask_frames]
        for i in range(1, self.feature_mask_block_size):
            ids_mask_frames_.append(ids_mask_frames +
                                    i)  # add the adjacent elements
        ids_mask_frames_ = torch.hstack(ids_mask_frames_)

        # set the mask elements to zero and return
        mask_features[torch.arange(N).unsqueeze(1),
                      ids_mask_frames_] = torch.zeros_like(ids_mask_frames_,
                                                           dtype=bool)
        return mask_features

    def forward(self, x):
        """ 
        x has shape (N,T,D): N batches of T frames of dimension D . 
        Do random masking on randomly within the T sequence. 
        Put all frames through the same MLP. 
        """
        N, T, D = x.shape
        if self.training:
            mask = self.get_feature_mask(x).to(x.device)
            x = mask.unsqueeze(-1) * x

        shape = x.shape
        x = self.enc(x.view(N * T, D)).view(N, T,
                                            -1)  # put through encoder where
        return x
