import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth
from typing import List
from typing import Iterable
import torch.nn.functional as F


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, dilation_this = 1):
        super().__init__()
        if dilation_this == 1:
            self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                              stride=stride, padding=kernel//2, bias = False, dilation = dilation_this))
            self.add_module('norm', nn.BatchNorm2d(out_channels))
            self.add_module('relu', nn.ReLU())
        else:
            self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                                padding='same', bias = False, dilation = dilation_this))
            self.add_module('norm', nn.BatchNorm2d(out_channels))
            self.add_module('relu', nn.ReLU())


    def forward(self, x):
        return super().forward(x)

class MyDecoder(nn.Sequential):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        ###
        self.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=16,
                                          kernel_size=(1, 1), stride=1, padding=0, bias=True))
        self.add_module('norm', nn.BatchNorm2d(16))
        self.add_module('relu', nn.ReLU())

        ###
        self.add_module('conv_b', nn.Conv2d(in_channels=16, out_channels=48,
                                            kernel_size=(3, 3), stride=1, padding=0, bias=True))
        self.add_module('norm_b', nn.BatchNorm2d(48))
        self.add_module('relu_b', nn.ReLU())

        ###

        self.add_module('conv_c', nn.Conv2d(in_channels=48, out_channels=out_channels,
                                            kernel_size=(1, 1), stride=1, padding=0, bias=True))

    def forward(self, x):
        return super().forward(x)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out

class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )

class ResidualAdd(nn.Module):
    """Just an util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )

class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)


def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr = curr + size
        yield chunk


class SegFormerEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            widths: List[int],
            depths: List[int],
            all_num_heads: List[int],
            patch_sizes: List[int],
            overlap_sizes: List[int],
            reduction_ratios: List[int],
            mlp_expansions: List[int],
            drop_prob: float = .0
    ):
        super().__init__()

        self.base = nn.ModuleList([])
        first_ch = [16, 24, 32, 48]
        self.base.append( ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=1, dilation_this=1) )  
        self.base.append( ConvLayer(first_ch[0], first_ch[1],  kernel=3, dilation_this=1) )                          
        self.base.append( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2, dilation_this=1) )                
        self.base.append( ConvLayer(first_ch[2], first_ch[3],  kernel=3, dilation_this=1) )                          
        grmul    = 1.7
        gr       = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]
        blk = HarDBlock(first_ch[3], gr[0], grmul, n_layers[0])
        self.base.append(blk)
        ch = blk.get_out_ch()
        self.base.append( ConvLayer(ch, 64, kernel=1) )
        self.base.append( nn.AvgPool2d(kernel_size=2, stride=2) ) 

        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                [in_channels, *widths],
                widths,
                patch_sizes,
                overlap_sizes,
                chunks(drop_probs, sizes=depths),
                depths,
                reduction_ratios,
                all_num_heads,
                mlp_expansions
            )
            ]
        )

    def forward(self, x):
        
        for idx_module in range(len(self.base)):
            x = self.base[idx_module](x)



        features = []
        features.append(x)                                                                 
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features



class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []

        out_channels = growth_rate

        link = []

        for i in range(10):
            dv = 2 ** i

            if layer % dv == 0:
                k = layer - dv
                link.append(k)

                if i > 0:
                    out_channels = out_channels * grmul


        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0

        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels = in_channels + ch

        return out_channels, in_channels, link



    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, have_dilation = False):
        super().__init__()

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0       


        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out

            if have_dilation:
                layers_.append( ConvLayer(inch, outch, dilation_this=1) )
            else:
                layers_.append(ConvLayer(inch, outch))


            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels = self.out_channels + outch

        self.layers = nn.ModuleList(layers_)


    def forward(self, x):

        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []

            for i in link:
                tin.append(layers_[i])

            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]

            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)

        out_ = []

        for i in range(t):
            if (i == 0 and self.keepBase) or \
               (i == t-1) or (i%2 == 1):
                out_.append(layers_[i])
            
        out = torch.cat(out_, 1)
        return out


class TransitionUp(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True)

        if concat:                            
            out = torch.cat([out, skip], 1)
          
        return out


class TRITNetEncoder_arch(nn.Module):
    def __init__(
            self,
            in_channels=64,
            widths=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            all_num_heads=[1, 2, 4, 8],
            patch_sizes=[7, 3, 3, 3],
            overlap_sizes=[2, 2, 2, 2],
            reduction_ratios=[8, 4, 2, 1],
            mlp_expansions=[4, 4, 4, 4, 4],
            drop_prob=0.0,
            embedding_size=8192
    ):
        super().__init__()
        self.embedding_size = embedding_size

        self.base = nn.ModuleList([])
        first_ch = [16, 24, 32, 48]
        self.base.append( ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=1, dilation_this=1) )  # conv0
        self.base.append( ConvLayer(first_ch[0], first_ch[1],  kernel=3, dilation_this=1) )                          # conv1
        self.base.append( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2, dilation_this=1) )                # conv2
        self.base.append( ConvLayer(first_ch[2], first_ch[3],  kernel=3, dilation_this=1) )                          # conv3
        grmul    = 1.7
        gr       = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]
        blk = HarDBlock(first_ch[3], gr[0], grmul, n_layers[0])
        self.base.append(blk)
        ch = blk.get_out_ch()
        self.base.append( ConvLayer(ch, 64, kernel=1) )
        self.base.append( nn.AvgPool2d(kernel_size=2, stride=2) ) 

        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                [in_channels, *widths],
                widths,
                patch_sizes,
                overlap_sizes,
                chunks(drop_probs, sizes=depths),
                depths,
                reduction_ratios,
                all_num_heads,
                mlp_expansions
            )
            ]
        )

        self.proj = nn.Sequential(
            nn.LazyLinear(8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU(),

            nn.Linear(8192, 8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU(),

            nn.Linear(8192, self.embedding_size, bias=False), 
        )

    def forward(self, x):
        for idx_module in range(len(self.base)):
            x = self.base[idx_module](x)
        for stage in self.stages:
            x = stage(x)



        h = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)  
        z = self.proj(h)  
        
        return z


class TRITNetEncoder(nn.Module):
    def __init__(self, **enc_kwargs):
        super().__init__()
        self.encoder = TRITNetEncoder_arch(**enc_kwargs)  # <- attribute name sets the prefix

    def forward(self, x):
        return self.encoder(x)


class TRITNet(nn.Module):
    def __init__(
            self,
            in_channels=64,
            widths=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            all_num_heads=[1, 2, 4, 8],
            patch_sizes=[7, 3, 3, 3],
            overlap_sizes=[2, 2, 2, 2],
            reduction_ratios=[8, 4, 2, 1],
            mlp_expansions=[4, 4, 4, 4, 4],
            drop_prob=0.0,
    ):
        super().__init__()

        cur_channels_count = widths[-1]            
        n_blocks = 3
        grmul    = 1.7
        gr       = [16,18,24]              #[10, 16, 18, 24, 32]
        n_layers = [4, 8, 8]               #[4, 4, 8, 8, 8]

        self.n_blocks       = n_blocks
        self.transUpBlocks  = nn.ModuleList([])
        self.denseBlocksUp  = nn.ModuleList([])
        self.conv1x1_up     = nn.ModuleList([])
        self.n_classes      = 3
    
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
    
        ###================================================================================================
        ### up-network
        ###================================================================================================
        for i in range(n_blocks-1,-1,-1):
            self.transUpBlocks.append(TransitionUp())        
            cur_channels_count = cur_channels_count + widths[i]

            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1))    
            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            self.denseBlocksUp.append(blk)                                     
            cur_channels_count = blk.get_out_ch()

        
        self.transUpBlocks.append(TransitionUp())                                                  
        cur_channels_count = cur_channels_count + 64                                                 
        self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1))       
        cur_channels_count = cur_channels_count//2                                                  
        blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])                               
        self.denseBlocksUp.append(blk)                                                                     
        cur_channels_count = blk.get_out_ch()                                                        



        ###================================================================================================
        ### output branches
        ###================================================================================================
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                            out_channels=self.n_classes, kernel_size=1, stride=1,
                            padding=0, bias=True)
        self.relu_on_finalConv = nn.ReLU()



    def forward(self, x):
        size_in = x.size()
        features = self.encoder(x)

            
        out_seg = features.pop()
        for i in range(self.n_blocks+1):
            skip = features.pop()
            out_seg = self.transUpBlocks[i](out_seg, skip, True)
            out_seg = self.conv1x1_up[i](out_seg)
            out_seg = self.denseBlocksUp[i](out_seg)
                  
        out_seg = self.finalConv(out_seg)
        out_seg_after_relu = self.relu_on_finalConv(out_seg)

        out_seg_final = F.interpolate(out_seg_after_relu, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

        return out_seg_final
