# Usage
```
#tiny model
tiny_args = {'stage_num_block':[2,2,5,2],
              'input_dim':3,
              'stem_dim':64,
              'stage_dim':64,
              'num_heads': 2,
              'mbconv_ksize': 3,
              'pooling_size': 2,
              'num_classes':1000,
              'block_size':(7,7),
              'grid_size':(7,7),
              'mbconv_expand_rate':4,
              'se_rate':0.25,
              'mlp_ratio':4,
              'qkv_bias':True,
              'qk_scale':None, 
              'drop':0., 
              'attn_drop':0.,
              'drop_path':0.,
              'act_layer':nn.GELU,
              'norm_layer':Channel_Layernorm}
```
#small model
```
small_args = {'stage_num_block':[2,2,5,2],
              'input_dim':3,
              'stem_dim':64,
              'stage_dim':96,
              'num_heads': 3,
              'mbconv_ksize': 3,
              'pooling_size': 2,
              'num_classes':1000,
              'block_size':(7,7),
              'grid_size':(7,7),
              'mbconv_expand_rate':4,
              'se_rate':0.25,
              'mlp_ratio':4,
              'qkv_bias':True,
              'qk_scale':None, 
              'drop':0., 
              'attn_drop':0.,
              'drop_path':0.,
              'act_layer':nn.GELU,
              'norm_layer':Channel_Layernorm}
```
#base model
```
base_args = {'stage_num_block':[2,6,14,2],
              'input_dim':3,
              'stem_dim':64,
              'stage_dim':96,
              'num_heads': 3,
              'mbconv_ksize': 3,
              'pooling_size': 2,
              'num_classes':1000,
              'block_size':(7,7),
              'grid_size':(7,7),
              'mbconv_expand_rate':4,
              'se_rate':0.25,
              'mlp_ratio':4,
              'qkv_bias':True,
              'qk_scale':None, 
              'drop':0., 
              'attn_drop':0.,
              'drop_path':0.,
              'act_layer':nn.GELU,
              'norm_layer':Channel_Layernorm}
```
#large model
```
large_args = {'stage_num_block':[2,6,14,2],
              'input_dim':3,
              'stem_dim':128,
              'stage_dim':128,
              'num_heads': 4,
              'mbconv_ksize': 3,
              'pooling_size': 2,
              'num_classes':1000,
              'block_size':(7,7),
              'grid_size':(7,7),
              'mbconv_expand_rate':4,
              'se_rate':0.25,
              'mlp_ratio':4,
              'qkv_bias':True,
              'qk_scale':None, 
              'drop':0., 
              'attn_drop':0.,
              'drop_path':0.,
              'act_layer':nn.GELU,
              'norm_layer':Channel_Layernorm}
```

```
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def test():
    print(f'MaxViT-T:{get_n_params(MaxViT(&&&&_args))}')
    print(MaxViT(&&&&_args)(torch.zeros(2,3,224,224)).shape)
    
```
