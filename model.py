import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Activation, Input
from tensorflow._api.v2.nn import depth_to_space
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
from cbam import *

class patch_expanding(layers.Layer):
    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, name='patch_expand', **kwargs):
        super(patch_expanding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector

        # Linear transformations that doubles the channels
        self.linear_trans1 = Conv2D(upsample_rate * embed_dim, kernel_size=1, use_bias=False,
                                    name='{}_linear_trans1'.format(name))
        #
        self.linear_trans2 = Conv2D(upsample_rate * embed_dim, kernel_size=1, use_bias=False,
                                    name='{}_linear_trans1'.format(name))
        self.prefix = name

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
            'upsample_rate': self.upsample_rate,
            'return_vector': self.return_vector,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        assert (L == H * W), 'input feature has wrong size'

        x = tf.reshape(x, (-1, H, W, C))

        x = self.linear_trans1(x)

        # rearange depth to number of patches
        x = depth_to_space(x, self.upsample_rate, data_format='NHWC', name='{}_d_to_space'.format(self.prefix))

        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1, L * self.upsample_rate * self.upsample_rate, C // 2))
        return x
class PatchEmbed(layers.Layer):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = (patch_size, patch_size)
        self.norm = norm_layer(epsilon=1e-6, name="norm") if norm_layer else layers.Activation('linear')

        self.proj = layers.Conv2D(filters=embed_dim, kernel_size=patch_size,
                                  strides=patch_size, padding='SAME',
                                  kernel_initializer=initializers.LecunNormal(),
                                  bias_initializer=initializers.Zeros(),
                                  name="proj")

    def call(self, x, **kwargs):
        _, H, W, _ = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding

        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            paddings = tf.constant([[0, 0],
                                    [0, self.patch_size[0] - H % self.patch_size[0]],
                                    [0, self.patch_size[1] - W % self.patch_size[1]]])
            x = tf.pad(x, paddings)

        # 下采样patch_size倍
        x = self.proj(x)
        B, H, W, C = x.shape
        # B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # [B, H, W, C] -> [B, H*W, C]
        x = tf.reshape(x, [B, -1, C])

        x = self.norm(x)
        return x, H, W

def window_partition(x, window_size: int):
    """
        将feature map按照window_size划分成一个个没有重叠的window
        Args:
            x: (B, H, W, C)
            window_size (int): window size(M)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    # transpose: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # reshape: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # reshape: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # reshape: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, -1])
    return x
class PatchMerging(layers.Layer):
    def __init__(self, dim: int, norm_layer=layers.LayerNormalization, name=None):
        super(PatchMerging, self).__init__(name=name)
        self.dim = dim
        self.reduction = layers.Dense(2*dim,
                                      use_bias=False,
                                      kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                                      name="reduction")
        self.norm = norm_layer(epsilon=1e-6, name="norm")

    def call(self, x, H, W):
        """
        x: [B, H*W, C]
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = tf.reshape(x, [B, H, W, C])
        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 != 0) or (W % 2 != 0)
        if pad_input:
            paddings = tf.constant([[0, 0],
                                    [0, 1],
                                    [0, 1],
                                    [0, 0]])
            x = tf.pad(x, paddings)

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = tf.concat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = tf.reshape(x, [B, -1, 4*C])  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x
class MLP(layers.Layer):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    k_ini = initializers.TruncatedNormal(stddev=0.02)
    b_ini = initializers.Zeros()

    def __init__(self, in_features, mlp_ratio=4.0, drop=0., name=None):
        super(MLP, self).__init__(name=name)
        self.fc1 = layers.Dense(int(in_features * mlp_ratio), name="fc1",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.act = layers.Activation("gelu")
        self.fc2 = layers.Dense(in_features, name="fc2",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.drop = layers.Dropout(drop)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x
class WindowAttention(layers.Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop_ratio (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop_ratio (float, optional): Dropout ratio of output. Default: 0.0
    """

    k_ini = initializers.GlorotUniform()
    b_ini = initializers.Zeros()

    def __init__(self,
                 dim,
                 window_size,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 name=None):
        super(WindowAttention, self).__init__(name=name)
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name="qkv",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.attn_drop = layers.Dropout(attn_drop_ratio)
        self.proj = layers.Dense(dim, name="proj",
                                 kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.proj_drop = layers.Dropout(proj_drop_ratio)

    def build(self, input_shape):
        # define a parameter table of relative position bias
        # [2*Mh-1 * 2*Mw-1, nH]
        self.relative_position_bias_table = self.add_weight(
            shape=[(2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads],
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=tf.float32,
            name="relative_position_bias_table"
        )

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = np.reshape(coords, [2, -1])  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = np.transpose(relative_coords, [1, 2, 0])   # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]

        self.relative_position_index = tf.Variable(tf.convert_to_tensor(relative_position_index),
                                                   trainable=False,
                                                   dtype=tf.int32,             # linux: tf.int64       windows：tf.int32
                                                   name="relative_position_index")

    def call(self, x, mask=None, training=None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            training: whether training mode
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape

        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        qkv = self.qkv(x)
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        # transpose: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale

        # relative_position_bias(reshape): [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = tf.gather(self.relative_position_bias_table,
                                           tf.reshape(self.relative_position_index, [-1]))
        relative_position_bias = tf.reshape(relative_position_bias,
                                            [self.window_size[0] * self.window_size[1],
                                             self.window_size[0] * self.window_size[1],
                                             -1])
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + tf.expand_dims(relative_position_bias, 0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn(reshape): [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask(expand_dim): [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + tf.expand_dims(tf.expand_dims(mask, 1), 0)
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        # multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        x = tf.matmul(attn, v)
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = tf.reshape(x, [B_, N, C])

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x
class SwinTransformerBlock(layers.Layer):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., name=None):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attn = WindowAttention(dim,
                                    window_size=(window_size, window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    attn_drop_ratio=attn_drop,
                                    proj_drop_ratio=drop,
                                    name="attn")
        self.drop_path = layers.Dropout(rate=drop_path, noise_shape=(None, 1, 1)) if drop_path > 0. \
            else layers.Activation("linear")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.mlp = MLP(dim, drop=drop, name="mlp")

    def call(self, x, attn_mask, training=None):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, [B, H, W, C])

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            paddings = tf.constant([[0, 0],
                                    [0, pad_r],
                                    [0, pad_b],
                                    [0, 0]])
            x = tf.pad(x, paddings)

        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = tf.reshape(x_windows, [-1, self.window_size * self.window_size, C])  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, training=training)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = tf.reshape(attn_windows,
                                  [-1, self.window_size, self.window_size, C])  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = tf.slice(x, begin=[0, 0, 0, 0], size=[B, H, W, C])

        x = tf.reshape(x, [B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x, training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)

        return x
class BasicLayer(layers.Layer):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = [
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else self.shift_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 name=f"block{i}")
            for i in range(depth)
        ]

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, name="downsample")
        else:
            self.downsample = None

    def create_mask(self, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = np.zeros([1, Hp, Wp, 1])  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        img_mask = tf.convert_to_tensor(img_mask, dtype=tf.float32)
        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])  # [nW, Mh*Mw]
        # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)

        return attn_mask

    def call(self, x, H, W, training=None):
        attn_mask = self.create_mask(H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask, training=training)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W
# def output(x):
#     B, H_W, C = x.shape
#     x = tf.reshape(x, [B, 288, 288, -1])
#     x = Conv2D(1, 1, padding='same')(x)
#     x = Activation('sigmoid')(x)
#     return x

class output(layers.Layer):
    def __init__(self, name=None):
        super(output, self).__init__(name=name)
        self.conv = layers.Conv2D(1, 1, padding='same')
        self.act = layers.Activation('sigmoid')
    def call(self, x, training=None):
        B, H_W, C = x.shape
        x = tf.reshape(x, [B, 288, 288, -1])
        x = self.conv(x, training=training)
        x = self.act(x)
        return x


class SwinTransformer():
    def __init__(self, patch_size=4,
                 embed_dim=96,
                 window_size_1=7,
                 depths_encode_1=(2, 2, 2, 2), depths_decode_1=(2, 2, 2),
                 num_heads_encode_1=(3, 6, 12, 24), num_heads_decode_1=(12, 6, 3),
                 window_size_2=14,
                 depths_encode_2=(2, 2, 2, 2), depths_decode_2=(2, 2, 2),
                 num_heads_encode_2=(3, 6, 12, 24), num_heads_decode_2=(12, 6, 3),
                 mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=layers.LayerNormalization, name=None, **kwargs):
        # super().__init__(name=name)

        self.num_layers_encode1 = len(depths_encode_1)
        self.num_layers_decode1 = len(depths_decode_1)
        self.num_layers_encode2 = len(depths_encode_2)
        self.num_layers_decode2 = len(depths_decode_2)

        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.pos_drop = layers.Dropout(drop_rate)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size=patch_size,
                                      embed_dim=embed_dim,
                                      norm_layer=norm_layer)

        # build layers encode_1
        # stochastic depth decay rule
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths_encode_1))]
        self.stage_layers_encode_1 = []
        for i_layer in range(self.num_layers_encode1):
            if i_layer == 3:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   depth=depths_encode_1[i_layer],
                                   num_heads=num_heads_encode_1[i_layer],
                                   window_size=window_size_1,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths_encode_1[:i_layer]):sum(depths_encode_1[:i_layer + 1])],
                                   downsample=None,
                                   name=f"layer{i_layer}")
                self.stage_layers_encode_1.append(layer)
            else:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   depth=depths_encode_1[i_layer],
                                   num_heads=num_heads_encode_1[i_layer],
                                   window_size=window_size_1,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths_encode_1[:i_layer]):sum(depths_encode_1[:i_layer + 1])],
                                   downsample=PatchMerging,
                                   name=f"layer{i_layer}")
                self.stage_layers_encode_1.append(layer)

        # build layers decode_1
        # stochastic depth decay rule
        self.patch_expanding = patch_expanding
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths_decode_1))]
        self.stage_layers_decode_1 = []
        for i_layer in range(self.num_layers_decode1):
            layer = BasicLayer(dim=int(embed_dim * 2 ** (2 - i_layer)),
                               depth=depths_decode_1[i_layer],
                               num_heads=num_heads_decode_1[i_layer],
                               window_size=window_size_1,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths_decode_1[:i_layer]):sum(depths_decode_1[:i_layer + 1])],
                               downsample=None,
                               name=f"layer{i_layer+4}")
            self.stage_layers_decode_1.append(layer)

        # build layers encode_2
        # stochastic depth decay rule
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths_encode_2))]
        self.stage_layers_encode_2 = []
        for i_layer in range(self.num_layers_encode2):
            if i_layer == 3:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   depth=depths_encode_2[i_layer],
                                   num_heads=num_heads_encode_2[i_layer],
                                   window_size=window_size_2,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths_encode_2[:i_layer]):sum(depths_encode_2[:i_layer + 1])],
                                   downsample=None,
                                   name=f"layer{i_layer+7}")
                self.stage_layers_encode_2.append(layer)
            else:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   depth=depths_encode_2[i_layer],
                                   num_heads=num_heads_encode_2[i_layer],
                                   window_size=window_size_2,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths_encode_2[:i_layer]):sum(depths_encode_2[:i_layer + 1])],
                                   downsample=PatchMerging,
                                   name=f"layer{i_layer+7}")
                self.stage_layers_encode_2.append(layer)

        # build layers decode_2
        # stochastic depth decay rule
        self.patch_expanding = patch_expanding
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths_decode_2))]
        self.stage_layers_decode_2 = []
        for i_layer in range(self.num_layers_decode2):
            layer = BasicLayer(dim=int(embed_dim * 2 ** (2 - i_layer)),
                               depth=depths_decode_2[i_layer],
                               num_heads=num_heads_decode_2[i_layer],
                               window_size=window_size_2,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths_decode_2[:i_layer]):sum(depths_decode_2[:i_layer + 1])],
                               downsample=None,
                               name=f"layer{i_layer + 11}")
            self.stage_layers_decode_2.append(layer)

        self.norm = norm_layer(epsilon=1e-6, name="norm")
        self.output = output()


    def call(self, input, skip_1,  training=None):
        # 位置编码 token化
        x, H, W = self.patch_embed(input)
        x = self.pos_drop(x, training=training)

        skip_connection_1 = []
        skip_1[1] = tf.reshape(skip_1[1], [2, 72*72, -1])
        skip_1[2] = tf.reshape(skip_1[2], [2, 36*36, -1])
        skip_1[3] = tf.reshape(skip_1[3], [2, 18*18, -1])

        x = tf.reshape(x, [2, 72, 72, -1])
        x = squeeze_excite_block(x)
        x = tf.reshape(x, [2, 72 * 72, -1])
        skip_connection_1.append(x)
        x, H, W = self.stage_layers_encode_1[0](x, 72, 72, training=training)
        x = tf.reshape(x, [2, 36, 36, -1])
        x = squeeze_excite_block(x)
        x = tf.reshape(x, [2, 36 * 36, -1])
        skip_connection_1.append(x)
        x, H, W = self.stage_layers_encode_1[1](x, 36, 36, training=training)
        x = tf.reshape(x, [2, 18, 18, -1])
        x = squeeze_excite_block(x)
        x = tf.reshape(x, [2, 18 * 18, -1])
        skip_connection_1.append(x)
        x, H, W = self.stage_layers_encode_1[2](x, 18, 18, training=training)
        x, H, W = self.stage_layers_encode_1[3](x, 9, 9, training=training)



        x = patch_expanding(num_patch=(9, 9), embed_dim=768, upsample_rate=2, return_vector=True)(x)
        x = Concatenate()([x, skip_connection_1[2], skip_1[3]]) # cbam 放这儿
        x = Dense(384, use_bias=False)(x)
        x, H, W = self.stage_layers_decode_1[0](x, 18,  18, training=training)

        x = patch_expanding(num_patch=(18, 18), embed_dim=384, upsample_rate=2, return_vector=True)(x)
        x = Concatenate()([x, skip_connection_1[1], skip_1[2]])
        x = Dense(192, use_bias=False)(x)
        x, H, W = self.stage_layers_decode_1[1](x, 36, 36, training=training)

        x = patch_expanding(num_patch=(36, 36), embed_dim=192, upsample_rate=2, return_vector=True)(x)
        x = Concatenate()([x, skip_connection_1[0], skip_1[1]])
        x = Dense(96, use_bias=False)(x)
        x, H, W = self.stage_layers_decode_1[2](x, 72, 72, training=training)
        x = self.output(x)

        # x = input * output_1
        #
        # x, H, W = self.patch_embed(x)
        # x = self.pos_drop(x, training=training)
        # skip_connection_2.append(x)
        #
        # x, H, W = self.stage_layers_encode_2[0](x, 72, 72, training=training)
        # skip_connection_2.append(x)
        # x, H, W = self.stage_layers_encode_2[1](x, 36, 36, training=training)
        # skip_connection_2.append(x)
        # x, H, W = self.stage_layers_encode_2[2](x, 18, 18, training=training)
        # skip_connection_2.append(x)
        # x, H, W = self.stage_layers_encode_2[3](x, 9, 9, training=training)
        #
        # x = patch_expanding(num_patch=(9, 9), embed_dim=768, upsample_rate=2, return_vector=True)(x)
        # x = Concatenate()([x, skip_connection_1[2], skip_connection_2[2]])
        # x = Dense(384, use_bias=False)(x)
        # x, H, W = self.stage_layers_decode_2[0](x, 18, 18, training=training)
        #
        # x = patch_expanding(num_patch=(18, 18), embed_dim=384, upsample_rate=2, return_vector=True)(x)
        # x = Concatenate()([x, skip_connection_1[1], skip_connection_2[1]])
        # x = Dense(192, use_bias=False)(x)
        # x, H, W = self.stage_layers_decode_2[1](x, 36, 36, training=training)
        #
        # x = patch_expanding(num_patch=(36, 36), embed_dim=192, upsample_rate=2, return_vector=True)(x)
        # x = Concatenate()([x, skip_connection_2[0], skip_connection_2[0]])
        # x = Dense(96, use_bias=False)(x)
        # x, H, W = self.stage_layers_decode_2[2](x, 72, 72, training=training)
        # x = self.output(x)
        # x = Concatenate()([output_1, output_2])

        # x = self.norm(x)  # [B, L, C]
        # x = tf.reduce_mean(x, axis=1)

        return x

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def conv_block(inputs, filters):
    x = inputs

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = cbam_module(x)

    return x

def encoder1(inputs):
    skip_connections = []
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs) # 论文VGG
    names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]

    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("conv4_block6_out").output
    return output, skip_connections

def decoder1(inputs, skip_connections):
    num_filters = [512, 256, 64, 32]
    skip_connections = skip_connections[:3]
    skip_connections.reverse()
    x = inputs
    for i, f in enumerate(num_filters):
        if i != 3:
            x = UpSampling2D((2, 2), interpolation='bilinear')(x)
            x = Concatenate()([x, skip_connections[i]])
            x = conv_block(x, f)
        else:
            x = UpSampling2D((2, 2), interpolation='bilinear')(x)
            x = conv_block(x, f)
    return x

def ASPP(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x

def build_model(shape):
    inputs = Input(shape, batch_size=2)
    x, skip_1 = encoder1(inputs)
    x = ASPP(x, 64)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)

    x = inputs * outputs1

    outputs2 = SwinTransformer().call(x, skip_1) # .call() .forward() encoder2 decoder2

    output = Concatenate()([outputs1, outputs2])

    model = Model(inputs, output)
    return model

if __name__ == "__main__":
    model = build_model((288, 288, 3))
    model.summary()