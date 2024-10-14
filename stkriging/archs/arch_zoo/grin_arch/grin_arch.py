import torch
import torch.nn as nn
from .grin_layers import BiGRIL


class GRIN(nn.Module):
    def __init__(self,

                 d_in,
                 d_hidden,
                 d_ff,
                 ff_dropout=0.1,
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 merge='mean',
                 impute_only_holes=True):
        super(GRIN, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0
        self.d_emb = int(d_emb) if d_emb is not None else 0

        self.impute_only_holes = impute_only_holes

        self.bigrill = BiGRIL(input_size=self.d_in,
                              ff_size=d_ff,
                              ff_dropout=ff_dropout,
                              hidden_size=self.d_hidden,
                              embedding_size=self.d_emb,
                              n_nodes=None,
                              n_layers=n_layers,
                              kernel_size=kernel_size,
                              decoder_order=decoder_order,
                              global_att=global_att,
                              u_size=self.d_u,
                              layer_norm=layer_norm,
                              merge=merge)

    def forward(self, X, adj, unknown_nodes, u=None, **kwargs):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        adj = torch.tensor(adj, dtype=torch.float32).to(X.device)
        if len(adj.shape) == 2:
            adj = adj.repeat(X.shape[0],1, 1)
        mask = torch.ones_like(X)
        #print(X.shape)
        mask[torch.arange(X.shape[0])[:, None], :,unknown_nodes, :] = 0
        x = X.permute(0, 3, 2, 1)
        #x = rearrange(X, 'b s n c -> b c n s')
        x = x[:, 0:1, :, :]
        #print(x.shape)
        #print(adj.shape)

        if mask is not None:
            mask = mask.permute(0, 3, 2, 1)
            #mask = rearrange(mask, 'b s n c -> b c n s')
            mask = mask[:, 0:1, :, :].to(dtype=torch.bool)
        #print(mask.shape, 'mask')
        if u is not None:
            u = u.permute(0, 3, 2, 1)
            #u = rearrange(u, 'b s n c -> b c n s')

        # imputation: [batches, channels, nodes, steps] prediction: [4, batches, channels, nodes, steps]
        imputation, prediction = self.bigrill(x, adj, mask=mask, u=u, cached_support=False)

        return [imputation]