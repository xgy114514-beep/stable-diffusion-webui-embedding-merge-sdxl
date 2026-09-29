# stable-diffusion-webui-embedding-merge

"""
WebUI 依赖:

1) 类 <modules.textual_inversion.textual_inversion.Embedding> 用于创建嵌入。
     必需字段: <.vec> = 实际张量, <.vectors> = 第一维大小, <.shape> = 最后一维大小。
2) 对象 <modules.sd_hijack.model_hijack.embedding_db> 被滥用创建临时嵌入。
     字段 <.word_embeddings> 和 <.ids_lookup> 的处理是从 </modules/textual_inversion/textual_inversion.py> 复制的，
     参考这里的 <register_embedding()>。
     更新: 不再需要，因为上游实现了 <register_embedding_by_name()>
3) 嵌入的保存是通过手动构造正确的 .pt 文件形状完成的，然后调用
     <modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload)>。
4) <modules.sd_hijack.StableDiffusionModelHijack.get_prompt_lengths(text)> 被钩住但未替换。
5) <encode_embedding_init_text()> 的一部分从 <sd_hijack_clip.py> 和 <sd_hijack_open_clip.py> 转换为
     <tokens_to_vectors()>；它使用 <shared.sd_model.cond_stage_model.wrapped>，然后调用
     <.model.token_embedding.wrapped()> 用于 SD2，或 <.transformer.text_model.embeddings.token_embedding.wrapped()> 用于 SD1。
6) 大量代码复制自 <https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer>：
     它获取 <shared.sd_model.cond_stage_model.wrapped> 并检查
     <FrozenCLIPEmbedder> 和 <FrozenOpenCLIPEmbedder>，参考这里的 <tokens_to_text()>。
7) 在解析提示时多次调用 <shared.sd_model.cond_stage_model.tokenize_line(line)>。
     代码非常依赖于其返回值！源文件在 </modules/sd_hijack_clip.py>
     也可以调用 <shared.sd_model.cond_stage_model.tokenize()>。
8) 如果检测到任何运行时嵌入，方法 <p.cached_params()> 被伪造为始终唯一，以防止错误缓存。
"""

import re
import os
import torch
import json
import html
import time
import types
import traceback
import threading
import gradio
import modules
from modules import shared, scripts, script_callbacks, devices, processing, sd_models
from modules.shared import opts, cmd_opts
from modules.textual_inversion.textual_inversion import Embedding
import open_clip.tokenizer

def _webui_embedding_merge_():

    class Exception_From_EmbeddingMergeExtension(Exception):
        pass
    class Exception_From_EmbeddingMergeExtension_():
        def __init__(self,_):
            self._ = _
        def __getattr__(self,_):
            raise Exception_From_EmbeddingMergeExtension(self._)

    def gr_tab():
        with gradio.Blocks(analytics_enabled=False) as block:
            gradio.HTML('<style>#tab_embedding_merge_extension p::before,#tab_embedding_merge_extension p::after,#tab_embedding_merge_extension code::before,#tab_embedding_merge_extension code::after{display:none!important}</style>')
            with gradio.Row():
                with gradio.Accordion('嵌入合并扩展！ (点击此处查看使用说明)', open=False):
                    with gradio.Accordion('介绍...', open=False):
                        gradio.Markdown('''
## 目的:

你知道吗，StableDiffusion 通过所谓的标记（token）来读取你的提示？它们是多维数值向量，共同构成单词和短语。

实际上，可以通过简单的合并（相加）不同向量来创建新词，结果可能同时表示两个事物！

然而，这并不总是有效，有时不会得到你期望的结果，但绝对值得尝试。

基本上，这个扩展将纯粹通过标记合并（无需在实际图像上进行训练）来创建文本反转嵌入，既可以在生成时自动完成，也可以在其选项卡上手动完成。

## 用法:

`EM` 选项卡可用于：
- 检查你的提示或特定词语
- 从文本片段创建 TI 嵌入，可合并或不合并
- 检查你的合并表达式是否正确
''')
                    gradio.Markdown('''
### 太长不看版:

使用语法 <`'事物A'+'事物B'`> 在你的正向或负向提示中将术语 "事物A" 和 "事物B" 合并成一个单一的嵌入。

也可以使用 <`'你的词语'*0.5`>（或任何数字，默认为 1.0）来增加或减少 "你的词语" 的本质（甚至可以设为零来禁用该部分提示）。

要将注意力与圆括号 ( ) 一起使用，请将 < > 放在它们外面，如 `(<'一'+'二'>:0.9)`  
在一个提示中使用任意数量的 <>，随心所欲；你也可以将现有的 TI 嵌入名称放在 `' '` 里面。

~~如果你需要字面的 <' 出于某种原因，请在它们之间加一个空格。~~ 你的提示中不能出现字面的 <' 任何地方；但如果加一个空格（`< '`），此扩展将忽略它。

如果其他扩展与此语法冲突，请将尖括号改为花括号：`{'同样有效'*4}`

## 查看文本或嵌入向量

你可以将原始提示（不带任何其他特殊语法）粘贴到 EM 选项卡的文本框中，查看 WebUI 如何解析它。所有检测到的文本反转嵌入都将被提取出来，并与文字标记一起呈现给你。例如：

>星际火车，杰作，作者 Danh Víµ
''')
                    with gradio.Accordion('关于表格列及其行分组的更多信息...', open=False):
                        gradio.Markdown('''
### 行:

- `按无分组` = 将提示作为一个整体解释，从实际标记中提取所有字符
- `按逗号分组` = 按标签上的逗号分割提示，删除逗号但保留原始空格字符
- `按部分分组` (默认) = 在 TI 嵌入处分割，将文本部分连接在一起，保留空格
- `按单词分组` = 仅在末尾实际产生空格字符的标记之后分割
- `按标记分组` = 除那些用多个向量表示的字符外，在所有地方分割
- `按向量分组` = 显示所有分离的向量，即使是 TI 嵌入的

### 列:

- `索引` = 该行的一个向量索引或索引范围（包含）
- `向量数` = 该行的最终向量数量（清晰可见）
- `文本` = 从标记重建的原始文本，为清晰起见用引号括起来
- `标记` = 表示该行的 CLIP 标记编号列表；对于 TI 嵌入，为 \* 或 \*_X，其中 X 是当前嵌入向量的索引
- `最小值` = 该向量或分组向量的最低（负）值
- `最大值` = 最大值
- `总和` = 所有值带符号的总和
- `绝对值` = 每个值的模之和，无符号（始终为正）
- `长度` = L2 范数向量长度，平方值总和的平方根（近似计算）
- `标准差` = 向量值的标准差。

### 你为什么需要它:

确保你的提示按预期方式解释（例如，检测到现有的 TI 嵌入）。你还可以通过这种方式探索 CLIP 标记。

如果你在底部的文本框中输入一个新名称，你当前的整个提示将被转换为一个具有该名称的单一文本反转嵌入（并存储在 `/embeddings/embedding_merge/` 子目录中）。你可以用于：

- 创建一个简化的部分以便在提示中快速使用（不过不推荐，因为你以后会丢失原始文本），但没有其他好处；
- 通过使用现有嵌入进行初始化，为实际训练准备 TI 嵌入。
''')
                    gradio.Markdown('''
## 测试合并表达式:

在 EM 选项卡中，你可以输入一个以单引号开头的“合并表达式”，查看此扩展将如何解析和组合它。它应包含用单引号括起来的文字文本或 TI 嵌入，以及它们之间的特殊运算符。例如：

>'greg rutkowski'/4+'gustav dore'*0.75
''')
                    with gradio.Accordion('关于合并表达式语法的更多信息...', open=False):
                        gradio.Markdown('''
### 表达式语法:

- `'一' + '二'` = 通过简单求和所有值来混合向量。如果长度不同，较短的部分将在右侧用零填充。
- `'一' - '二'` = 同上，但是减法。注意 + 和 - 只能放在文本部分之间，优先级最低。
- `'文本' * 数字` = 将引用文字的所有向量乘以数值。可以使用浮点数 (0.85) 和负数 (-1)，但不能使用算术表达式。
- `'文本' / 数字` = 除以数字，与乘法类似。应用于前面的文本文字，但会考虑到之前的类似操作，所以可以同时乘除 (\*3/5)。
- `'文本' : 数字` = 改变文字的向量数量，缩小或扩大（用零填充）。只接受不带符号的整数！
- `'文本' :+ 数字` 和 `'文本'  :- 数字` = 循环移位此标记中的向量，例如 +1 将每个向量的索引向前移动一位，在最后处回绕。
- `'文本',数字` (可链式，如 `'a',B,'c','d',E,F…`) = 用其数值索引连接文本与一个标记（因此，要获得任何纯标记——使用左侧空字符串：`'',256`）。特殊标记：`0000` = "起始标记" (索引 49406)，`000` = "结束标记" (索引 49407)，`00` = "填充标记" (对于 SD1 也是 49407，但对于 SD2 是 0)。标记编号 `0` 不是零向量，但出于某种原因计为符号 "!"，后面没有空格，这通常无法正常输入。

要对加法（或减法）的结果应用乘法（或除法）、裁剪或移位，你不能使用括号；相反，尝试以下语法：

- `'一' + '二' =* 数字` = 将乘 '一' 和 '二' 的和，但不仅乘 '二'
- `'一' + '二' =/ 数字` = 除以和（或左侧任意数量的和），实际上是“所有结果”
- `'一' + '二' =: 数字` = 裁剪或扩大结果
- `'一' + '二' =:+ 数字` 或 `'一' + '二' =:- 数字` = 旋转结果

因此，以下操作是相同的：

>`'a'/2 + 'b'/2 + '':1 - 'd'`  
`'a'+'b' =* 0.5 + 'c'*0 + 'd'*-1`

没有真正的“连接”运算符（因为你稍后可以连接几个单独的合并表达式），但如果你需要，可以通过添加相同文本的放大和移位版本来模拟。  
操作 "," 具有最高优先级（它将在执行任何其他操作之前直接构造字符串），因此你不能将任何内容连接到加法或乘法的结果。只在你的文本中通过索引添加标记时使用它。

例如，重复一个双向量单词，得到 4 个向量，两两相等：

> 'artstation' + 'artstation' :4 :+2  
> 'artstation','artstation'

你可以使用移位将同一文本的几个向量连接在一起。例如，给定一个 4 向量单词，你可以将这些向量合并为一个：

> 'kuvshinov' + 'kuvshinov':-1 + 'kuvshinov':-2 + 'kuvshinov':-3 =: 1  
> '',1836 + '',85 + '',43074 + '',341

请注意，这些索引指的是 "ku|v|shino|v[空格]"，无法从原始文本输入，因为它会被解析为 "ku[空格]"，"v[空格]" 和 "shino[空格]"，它们是不同的标记！

当你合并不等长的字符串时，最短的会用零向量填充；如果你想用其他东西填充，你应该检查向量计数并相应地进行连接：

> 'close-up',00,00 + 'out-of-frame' + 'cropped',00,00,00,00  
> 'up',00,00+'of-frame'+'',00,00,00 =:5:+2 + 'close-'+'out-'+'cropped',00

### 你为什么需要它:

准备你的表达式并修复任何错误。你可以通过大致比较表格中的数字来评估其正确性（例如，添加向量通常会导致更高的 `Abs` 值；而乘法会直接明显改变所有数字）。

如果出于某种原因你无法在运行时使用合并提示的语法，你至少可以输入一个名称并从你的合并表达式创建一个常规 TI 嵌入。然后即使没有安装此扩展，你也可以使用它！

你还可以检查训练好的文本嵌入的数值参数，并将其与“正常”向量进行比较。例如，非常大的 `Len` 或 `Std` 意味着某些地方出了问题，至少你可以尝试除以它来修复。
''')
                    gradio.Markdown('''
## 提示中的多个合并表达式:

如果你在 EM 选项卡的提示中任何地方放置一个有效的合并表达式，用尖括号 <'…' …> 或花括号 {'…' …} 括起来（在 `<` 或 `{` 和 `'` 之间没有空格），它将被解析并合并成一个临时的文本反转嵌入，该嵌入将替换表达式本身。生成的提示将由这些嵌入和表达式之间的任何内容连接而成。例如：

>A photo of <'cat'+'dog'>, {'4k'+'dynamic lighting'+'science fiction'=/3} masterpiece
''')
                    with gradio.Accordion('更多尖括号/花括号使用示例...', open=False):
                        gradio.Markdown('''
### 更多示例:


结合不同的主体或风格，产生融合的概念：

> A realistic photo of the <'girl'+'doll'> in rainbow dress standing on a shore.  
Art by <'greg rutkowski'*X+'hayao miyazaki'*Y> style.

注意：
- 当所有主题具有相同数量的向量时效果最好（也可以通过 BREAK 语句大致模拟：`… photo of the girl in rainbow … BREAK … photo of the doll in rainbow …`）；
- 你不必除以添加部分的数量，特别是如果你的主题非常不同（例如，不包含相同的标记）；
- 在第二个例子中，通过将每个部分乘以某个数字（X 和 Y 是 0.0 到 1.0 之间的数字），你可以得到一个加权组合或插值。

改变提示中单个词的权重：

> A <'peacock'*X> is standing on a top of <'giraffe'*Y>.  
worst quality, ugly, <'bad anatomy,':0> blurry, cropped

其中 X 和 Y 可以是 0.0 到 1.0 甚至更高的数字，最高可达 5。这样你可以直接改变主题之间的相对影响。

注意：
- 通常 0.5 到 1.5 之间的值不会真正改变任何东西，看起来像普通的 1.0
- 低于 0.5 且接近 0.0 的值确实会大大降低主题权重！直到其完全消失（否则不可能，例如即使零注意力 `(word:0)` 也不能从提示中消除 "word"）
- 高数字可能会增加对象的存在感，不是数量而是本质。非常高的乘数（高于 10）会破坏主题，但不会以其他方式改变图像。

通过将负提示的一部分向量归零，可以在不改变文本其余部分的情况下了解该部分的效果。由于 WebUI 在任意逗号处分割长提示（然后将结果部分合并），简单删除某部分可能会严重改变结果。
''')
                    gradio.Markdown('''
## 在运行时在提示中使用合并表达式！

你实际上可以在 WebUI 的 txt2img 或 img2img 提示中，将合并表达式放在尖括号或花括号中。此扩展将拦截主提示和负提示，解析并合并表达式，创建临时的 TI 嵌入，WebUI 将“看到”它们而不是你的原始文本。在生成信息中，会出现像 <'EM_1'> 这样的内部无意义名称，但额外的参数 "EmbeddingMerge" 将包含原始的合并表达式。要快速恢复你的提示，只需将完整的生成信息（从 .txt 或 PNG Info）粘贴到 EM 选项卡的文本框中（它也应该适用于官方的“粘贴”工具栏按钮）——其临时嵌入将替换回表达式，例如：

> a photo of <'EM_1'>  
Negative prompt: {'EM_2'}  
Steps: 8, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1374372309, Size: 512x512, Model hash: c6bbc15e32, Model: sd-v1-5-inpainting, EmbeddingMerge: "<'EM_1'>=<'sky' * 2/4 + 'forest' * 3/4>, {'EM_2'}={'blurry'+'cropped'}", Conditional mask weight: 1

供你参考，复制语法本身的起始标记：
- `<'` = `<'',27,6>` 或 `<'',27,262>`
- `{'` = `<'',90,6>` 或 `<'',90,262>`

''')
                    with gradio.Accordion('限制...', open=False):
                        gradio.Markdown('''
### 什么不起作用：

#### 将属性绑定到对象：

> Photo of a <'blonde'+'boy'> in <'red'+'shirt'> wearing <'green'+'pants'> and <'blue'+'shoes'>

– 结果不是所请求的，而是任何东西。

#### 将艺术家折叠成单个标记：

> Painting by <'William' + '-' + 'Adolphe'+'Adolphe':+1 + 'Bouguereau'+'Bouguereau':+1+'Bouguereau':+2 =:1>. A girl, masterpiece

– 结果几乎无法与将该项归零区分开来。

#### 像 word2vec 中那样减去概念：

> Full-body photo of a <'king'-'man'+'woman'>  
Detailed photo of <'yellow'-'red'> car

– 通常导致完全破坏的构图。

#### 通过对词语取反来模拟负提示：

> A portrait of the princess. <'frame, black-white'*-1>  
A cat is chasing a dog. <''-'road'-'grass'>

– 仍然会将这些概念添加到正向提示中，但存在感很奇怪。不过，你可能会在小的负值（`-0.1-0.0`）中找到更多运气。
''')
            with gradio.Row():
                gr_text = gradio.Textbox(value='', lines=4, max_lines=16, interactive=True, label='你的提示（无权重/注意力，不要转义括号）；或者你的合并表达式（如果第一个字符是单引号）；或者用于恢复提示的生成信息')
            with gradio.Row():
                with gradio.Column(scale=1):
                    gr_button = gradio.Button('解析！',variant='primary')
                with gradio.Column(scale=3):
                    gr_radio = gradio.Radio(choices=('按无分组','按逗号分组','按部分分组','按单词分组','按标记分组','按向量分组'), value='按部分分组', type='index', interactive=True, label='按以下方式分组/分割表格：（当不是以单引号开头时 - 即仅用于提示，不用于合并）')
            with gradio.Box():
                gr_html = gradio.HTML(label='输出')
            with gradio.Row():
                gr_true = gradio.Checkbox(value=True,visible=False,show_label=False)
                gr_false = gradio.Checkbox(value=False,visible=False,show_label=False)
                gr_name = gradio.Textbox(value='', lines=1, max_lines=1, interactive=True, label='在此处为你的新嵌入键入一个名称，该嵌入将存储上方按钮下一次解析/合并的结果：（可选；成功时清除）')
            with gradio.Row():
                gr_tensors = gradio.Checkbox(value=True,label="保存为 .safetensors（取消勾选则保存为 .pt）")
            gr_button.click(fn=gr_func, inputs=[gr_name,gr_text,gr_radio,gr_tensors,gr_true], outputs=[gr_html,gr_name,gr_text], show_progress=False)
            gr_radio.change(fn=gr_func, inputs=[gr_name,gr_text,gr_radio,gr_tensors,gr_false], outputs=[gr_html,gr_name,gr_text], show_progress=False)
        return [(block,'EM','embedding_merge_extension')]

    def tokens_to_text():
        try:
            # https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer
            class VanillaClip:
                def __init__(self, clip):
                    self.clip = clip
                def vocab(self):
                    return self.clip.tokenizer.get_vocab()
                def byte_decoder(self):
                    return self.clip.tokenizer.byte_decoder
            class OpenClip:
                def __init__(self, clip):
                    self.clip = clip
                    self.tokenizer = open_clip.tokenizer._tokenizer
                def vocab(self):
                    return self.tokenizer.encoder
                def byte_decoder(self):
                    return self.tokenizer.byte_decoder
            clip = shared.sd_model.cond_stage_model
            if hasattr(clip,'embedders'):
                clip = clip.embedders[0]
            if clip is None:
                return None
            clip = clip.wrapped
            typename = type(clip).__name__.split('.')[-1]
            if typename=='FrozenOpenCLIPEmbedder':
                clip = OpenClip(clip)
            else:
                clip = VanillaClip(clip)
            vocab = {v: k for k, v in clip.vocab().items()}
            byte_decoder = clip.byte_decoder()
            def _tokens_to_text(tokens):
                nonlocal vocab, byte_decoder
                code = []
                ids = []
                current_ids = []
                class_index = 0
                def dump(last=False):
                    nonlocal code, ids, current_ids
                    words = [vocab.get(x, '') for x in current_ids]
                    try:
                        word = bytearray([byte_decoder[x] for x in ''.join(words)]).decode('utf-8')
                    except UnicodeDecodeError:
                        if last:
                            word = '<ERR>' * len(current_ids)
                        elif len(current_ids) > 4:
                            id = current_ids[0]
                            ids += [id]
                            local_ids = current_ids[1:]
                            code += [([id], '<ERR>')]

                            current_ids = []
                            for id in local_ids:
                                current_ids.append(id)
                                dump()
                            return
                        else:
                            return
                    word = word.replace('</w>', ' ')
                    code += [(current_ids, word)]
                    ids += current_ids
                    current_ids = []
                for token in tokens:
                    token = int(token)
                    current_ids.append(token)
                    dump()
                dump(last=True)
                return [c for c in code if len(c[0])!=0]
            return _tokens_to_text
        except:
            traceback.print_exc()
            return None

    def str_to_escape(line):
        res = re.sub(r'([()[\]\\])',r'\\\1',line)
        return res

    def get_model_clips():
        sd_model = shared.sd_model
        clip = sd_model.cond_stage_model
        if clip is None:
            clip = sd_model.text_processing_engine if hasattr(sd_model,'text_processing_engine') else None
            if clip is None:
                clip_l = sd_model.text_processing_engine_l if hasattr(sd_model,'text_processing_engine_l') else None
                clip_g = sd_model.text_processing_engine_g if hasattr(sd_model,'text_processing_engine_g') else None
                if clip_l is not None:
                    if clip_g is not None:
                        # print(f"[EmbeddingMerge] get_model_clips: Forge SDXL, clip_l={type(clip_l).__name__}, clip_g={type(clip_g).__name__}")
                        return (clip_l,clip_g)
                    # print(f"[EmbeddingMerge] get_model_clips: Forge single clip, clip_l={type(clip_l).__name__}")
                    return (clip_l,)
                raise Exception_From_EmbeddingMergeExtension('找不到 CLIP 模型！')
        if(hasattr(clip,'embedders')):
            try:
                result = (clip.embedders[0],clip.embedders[1])
                # print(f"[EmbeddingMerge] get_model_clips: A1111 SDXL, embedders={[type(c).__name__ for c in result]}")
                return result # SDXL
            except:
                pass
        # print(f"[EmbeddingMerge] get_model_clips: A1111 SD1/SD2, clip={type(clip).__name__}")
        return (clip,) # SD1 or SD2

    def get_embedding_db():
        # Path 1: Standard A1111 / Forge hijack embedding database
        try:
            db = modules.sd_hijack.model_hijack.embedding_db
            if db is not None:
                # print(f"[EmbeddingMerge] get_embedding_db: 使用 model_hijack.embedding_db (类型: {type(db).__name__})")
                return (db,)
        except Exception as e:
            pass  # print(f"[EmbeddingMerge] get_embedding_db: 标准 hijack 路径失败: {e}")
        
        # Path 2: Try sd_hijack directly
        try:
            db = modules.sd_hijack.embedding_db
            if db is not None:
                # print(f"[EmbeddingMerge] get_embedding_db: 使用 sd_hijack.embedding_db (类型: {type(db).__name__})")
                return (db,)
        except Exception as e:
            pass  # print(f"[EmbeddingMerge] get_embedding_db: sd_hijack 路径失败: {e}")
        
        # Path 3: Forge - try shared.sd_model directly
        sd_model = shared.sd_model
        try:
            if hasattr(sd_model, 'embedding_db') and sd_model.embedding_db is not None:
                # print(f"[EmbeddingMerge] get_embedding_db: 使用 sd_model.embedding_db (类型: {type(sd_model.embedding_db).__name__})")
                return (sd_model.embedding_db,)
        except Exception as e:
            pass  # print(f"[EmbeddingMerge] get_embedding_db: sd_model.embedding_db 失败: {e}")
        
        # Path 4: Forge SDXL - collect ALL clip embedding databases (BOTH clip_l AND clip_g!)
        clips = get_model_clips()
        all_dbs = []
        for clip in clips:
            try:
                if hasattr(clip, 'embeddings') and hasattr(clip.embeddings, 'register_embedding_by_name'):
                    all_dbs.append(clip.embeddings)
                    # print(f"[EmbeddingMerge] get_embedding_db: 收集 clip.embeddings (类型: {type(clip.embeddings).__name__})")
            except:
                pass
        if len(all_dbs)>0:
            # print(f"[EmbeddingMerge] get_embedding_db: 共收集 {len(all_dbs)} 个 embedding 数据库 (Forge SDXL)")
            return tuple(all_dbs)
        
        # Path 5: Fallback to clip.embeddings (may be raw Embedding layers - won't work for registration)
        result = [c.embeddings for c in clips]
        # print(f"[EmbeddingMerge] get_embedding_db: 回退到 clip.embeddings (共 {len(result)} 个, 类型: {[type(r).__name__ for r in result]})")
        return result

    def tokenize_line(clip,text):
        if hasattr(clip,'encode_embedding_init_text'):
            return clip.tokenize_line(str_to_escape(text))
        old = clip.emphasis.name
        clip.emphasis.name = 'None'
        try:
            res = clip.tokenize_line(text)
        finally:
            clip.emphasis.name = old
        return res

    def _collect_all_chunks(chunk_obj):
        """从 PromptChunk 中收集所有子块的 tokens 和 fixes（支持超过75标记的长提示词）。"""
        all_tokens = []
        all_fixes = []

        def _collect_from(chunk, is_first=True):
            if hasattr(chunk, 'tokens'):
                tokens = list(chunk.tokens)
                if is_first:
                    all_tokens.extend(tokens)
                else:
                    # 跳过后续chunk的起始标记
                    all_tokens.extend(tokens[1:] if len(tokens) > 1 else tokens)
            if hasattr(chunk, 'fixes'):
                all_fixes.extend(chunk.fixes)

        # 尝试 __getitem__ 模式（部分fork使用）
        idx = 0
        has_getitem = False
        while True:
            try:
                sub = chunk_obj[idx]
                has_getitem = True
                _collect_from(sub, idx == 0)
                idx += 1
            except (IndexError, TypeError):
                break

        if not has_getitem:
            # 尝试 next_chunk 模式（标准A1111）
            _collect_from(chunk_obj)
            next_chunk = chunk_obj
            while hasattr(next_chunk, 'next_chunk') and next_chunk.next_chunk is not None:
                next_chunk = next_chunk.next_chunk
                _collect_from(next_chunk, is_first=False)

        return all_tokens, all_fixes

    def _collect_all_chunk_tokens(chunk_obj):
        """从所有 PromptChunk 子块中收集 tokens（支持超过75标记的长提示词）。"""
        all_tokens, _ = _collect_all_chunks(chunk_obj)
        return all_tokens

    # CLIP context window max: 77 positions (start + 75 tokens + end)
    CLIP_MAX_TOKENS = 75
    # 池化目标：超长文本默认压缩到的向量数（可改为1实现"单概念嵌入"）
    POOL_TARGET_VECTORS = CLIP_MAX_TOKENS

    def pool_embeddings(embeddings, num_vectors=POOL_TARGET_VECTORS):
        """将 Token 嵌入向量池化到指定数量。
        
        Args:
            embeddings: [N, dim] 张量
            num_vectors: 目标向量数 (1=均值池化/单概念, >1=分组均值池化/多向量)
        
        Returns:
            [num_vectors, dim] 池化后的张量
        """
        n = embeddings.size(0)
        if num_vectors >= n:
            return embeddings  # 不需要池化
        
        if num_vectors == 1:
            # 均值池化：全部压缩为单个向量（单概念嵌入）
            return embeddings.mean(dim=0, keepdim=True)
        
        # 分组均值池化：将 N 个向量均匀分成 num_vectors 组，每组取均值
        pooled = []
        for i in range(num_vectors):
            start = i * n // num_vectors
            end = (i + 1) * n // num_vectors
            chunk = embeddings[start:end]
            pooled.append(chunk.mean(dim=0))
        return torch.stack(pooled)

    def _get_all_token_embeddings(clip, text):
        """获取全部 Token 嵌入向量（不限制 CLIP 上下文窗口）。用于池化。"""
        token_result = tokenize_line(clip, text)
        chunk_obj = token_result[0]
        all_tokens = _collect_all_chunk_tokens(chunk_obj)
        total_count = len(all_tokens) - 1  # 跳过首个起始标记
        usable_tokens = all_tokens[1:total_count+1]
        if len(usable_tokens) == 0:
            return None
        token_tensor = torch.tensor(usable_tokens, device=devices.device)
        # Try multiple paths for compatibility with Forge / different CLIP wrappers
        try:
            return clip.text_encoder.transformer.text_model.embeddings.token_embedding.wrapped(token_tensor)
        except:
            pass
        try:
            return clip.wrapped.model.token_embedding.wrapped(token_tensor)
        except:
            pass
        try:
            raw_clip = clip.wrapped if hasattr(clip,'wrapped') else clip
            return raw_clip.model.token_embedding(token_tensor)
        except:
            pass
        try:
            raw_clip = clip.wrapped if hasattr(clip,'wrapped') else clip
            return raw_clip.transformer.text_model.embeddings.token_embedding(token_tensor)
        except:
            pass
        try:
            cond_model = clip.cond_stage_model if hasattr(clip,'cond_stage_model') else clip
            return cond_model.wrapped.model.token_embedding.wrapped(token_tensor)
        except:
            pass
        raw_clip = clip.wrapped if hasattr(clip,'wrapped') else clip
        if hasattr(raw_clip,'model') and hasattr(raw_clip.model,'token_embedding'):
            tok_emb = raw_clip.model.token_embedding
            tok_emb = tok_emb.wrapped if hasattr(tok_emb,'wrapped') else tok_emb
            return tok_emb(token_tensor)
        return clip.text_encoder.transformer.text_model.embeddings.token_embedding.wrapped(token_tensor)

    def encode_embedding_init_text(clip,text,length=999,pool_to=None):
        if hasattr(clip,'encode_embedding_init_text'):
            try:
                return clip.encode_embedding_init_text(text,length)
            except:
                pass

        # 池化模式：获取全部 Token 嵌入（无上限），然后池化压缩
        if pool_to is not None:
            full_embeddings = _get_all_token_embeddings(clip, text)
            if full_embeddings is None:
                return None
            # 如果有 length 限制，先截取
            if full_embeddings.size(0) > length:
                full_embeddings = full_embeddings[:length]
            return pool_embeddings(full_embeddings, pool_to)

        # 常规模式：受 CLIP 上下文窗口限制（最多75个有效标记）
        token_result = tokenize_line(clip,text)
        chunk_obj = token_result[0]
        # 收集所有 chunk 的 tokens，但受 CLIP 上下文窗口限制（最多75个有效标记）
        all_tokens = _collect_all_chunk_tokens(chunk_obj)
        total_count = len(all_tokens) - 1  # 从所有chunk计算真实总数（跳过首个起始标记）
        count = min(total_count, length, CLIP_MAX_TOKENS)
        usable_tokens = all_tokens[1:count+1]
        token_tensor = torch.tensor(usable_tokens, device=devices.device)
        # Try multiple paths for compatibility with Forge / different CLIP wrappers
        try:
            return clip.text_encoder.transformer.text_model.embeddings.token_embedding.wrapped(token_tensor)
        except:
            pass
        try:
            return clip.wrapped.model.token_embedding.wrapped(token_tensor)
        except:
            pass
        try:
            raw_clip = clip.wrapped if hasattr(clip,'wrapped') else clip
            return raw_clip.model.token_embedding(token_tensor)
        except:
            pass
        try:
            raw_clip = clip.wrapped if hasattr(clip,'wrapped') else clip
            return raw_clip.transformer.text_model.embeddings.token_embedding(token_tensor)
        except:
            pass
        try:
            cond_model = clip.cond_stage_model if hasattr(clip,'cond_stage_model') else clip
            return cond_model.wrapped.model.token_embedding.wrapped(token_tensor)
        except:
            pass
        # Last resort: try any known path on the wrapped model
        raw_clip = clip.wrapped if hasattr(clip,'wrapped') else clip
        if hasattr(raw_clip,'model') and hasattr(raw_clip.model,'token_embedding'):
            tok_emb = raw_clip.model.token_embedding
            tok_emb = tok_emb.wrapped if hasattr(tok_emb,'wrapped') else tok_emb
            return tok_emb(token_tensor)
        return clip.text_encoder.transformer.text_model.embeddings.token_embedding.wrapped(token_tensor)

    def text_to_vectors(orig_text):
        try:
            both = []
            for clip,lg in zip(get_model_clips(),('clip_l','clip_g')):
                res = []
                text = orig_text.lstrip().lower()
                token_result = tokenize_line(clip,text)
                chunk_obj = token_result[0]
                # 收集所有chunk的tokens和fixes（支持超过75标记的长提示词）
                all_tokens_raw, all_fixes = _collect_all_chunks(chunk_obj)
                fixes = all_fixes
                # 从收集的所有tokens计算真实总数（跳过首个起始标记）
                total_count = len(all_tokens_raw) - 1
                tokens = all_tokens_raw[1:total_count+1]
                start = 0
                for fix in fixes:
                    name = fix.embedding.name.lower()
                    tensor = fix.embedding.vec
                    if type(tensor)==dict:
                        tensor = tensor[lg]
                    num = fix.embedding.vectors
                    off = fix.offset
                    if num!=tensor.size(0):
                        return None
                    lenname = len(name)
                    if off!=start:
                        test = 0
                        while True:
                            pos = text.find(name,test)
                            if pos<0:
                                return None
                            test = pos+lenname
                            sub = text[0:test]
                            part_result = tokenize_line(clip,sub)
                            # 收集子分词的完整tokens（支持长文本）
                            sub_tokens_raw, _ = _collect_all_chunks(part_result[0])
                            sub_total = len(sub_tokens_raw) - 1
                            vec = min(off-start, CLIP_MAX_TOKENS)
                            need = tokens[start:off+num]
                            if sub_tokens_raw[1:sub_total+1]==need:
                                trans = encode_embedding_init_text(clip,text,vec)
                                t = trans.to(device=devices.device,dtype=torch.float32)
                                res.append((t,sub[:pos],need[:vec]))
                                text = text[pos:]
                                start = off
                                break
                    if text[0:lenname]!=name:
                        return None
                    tensor = tensor.to(device=devices.device,dtype=torch.float32)
                    res.append((tensor,name,None))
                    start += num
                    text = text[lenname:].lstrip()
                if text!='':
                    part_result = tokenize_line(clip,text)
                    # 收集子分词的完整tokens（支持长文本）
                    part_tokens_raw, _ = _collect_all_chunks(part_result[0])
                    part_total = len(part_tokens_raw) - 1
                    need = tokens[start:]
                    if part_tokens_raw[1:part_total+1]!=need:
                        return None
                    trans = encode_embedding_init_text(clip,text,CLIP_MAX_TOKENS)
                    trans = trans.to(device=devices.device,dtype=torch.float32)
                    # 截断 need 以匹配 CLIP 上下文窗口上限（trans 最多75个向量）
                    need = need[:trans.size(0)]
                    res.append((trans,text,need))
                print(f"[EmbeddingMerge] text_to_vectors: clip[{lg}] 提取了 {len(res)} 个片段")
                both.append(res)
            # Return None if all inner results are empty (Forge compatibility)
            if all(len(inner) == 0 for inner in both):
                return None
            return both
        except:
            traceback.print_exc()
            return None

    def text_to_tokens(text):
        try:
            both = []
            for clip in get_model_clips():
                tokens = clip.tokenize([text])[0]
                both.append(tokens)
            if len(both)>1:
                if (both[0]-both[1]).abs().max().item() != 0:
                    print('EM: text_to_tokens',both)
                    return None
            return both[0]
        except:
            return None

    def tokens_to_vectors(pair):
        try:
            res = []
            for clip,arr in zip(get_model_clips(),pair):
                raw_clip = clip.wrapped if hasattr(clip,'wrapped') else clip
                token_embedding = None
                # Try multiple paths to find the token embedding layer (Forge compatibility)
                try:
                    if hasattr(raw_clip,'model') and hasattr(raw_clip.model,'token_embedding'):
                        token_embedding = raw_clip.model.token_embedding
                except:
                    pass
                if token_embedding is None:
                    try:
                        token_embedding = raw_clip.transformer.text_model.embeddings.token_embedding
                    except:
                        pass
                if token_embedding is None:
                    try:
                        cond_model = clip.cond_stage_model if hasattr(clip,'cond_stage_model') else clip
                        raw_cond = cond_model.wrapped if hasattr(cond_model,'wrapped') else cond_model
                        if hasattr(raw_cond,'model') and hasattr(raw_cond.model,'token_embedding'):
                            token_embedding = raw_cond.model.token_embedding
                    except:
                        pass
                if token_embedding is None:
                    try:
                        cond_model = clip.cond_stage_model if hasattr(clip,'cond_stage_model') else clip
                        raw_cond = cond_model.wrapped if hasattr(cond_model,'wrapped') else cond_model
                        token_embedding = raw_cond.transformer.text_model.embeddings.token_embedding
                    except:
                        pass
                if token_embedding is None:
                    continue
                tok_emb = token_embedding.wrapped if hasattr(token_embedding,'wrapped') else token_embedding
                device = tok_emb.weight.device if hasattr(tok_emb,'weight') else devices.device
                tensor = torch.tensor([arr],dtype=torch.int,device=device)
                tokens = tok_emb(tensor).to(devices.device)
                res.append(tokens)
            if len(res)==0:
                return None
            if len(res)>1:
                if len(res[0]) != len(res[1]):
                    print('EM: tokens_to_vectors',res)
                    return None
            return res
        except:
            traceback.print_exc()
            return None

    def to_float(num):
        if num is None:
            return None
        try:
            return float(num)
        except:
            return None

    def to_int(num):
        if num is None:
            return None
        try:
            return int(num)
        except:
            return None

    def grab_vectors(text):
        try:
            both = []
            tv_result = text_to_vectors(text)
            if tv_result is None:
                print(f"[EmbeddingMerge] grab_vectors: text_to_vectors 返回 None for '{text[:60]}'")
                return None
            for idx, res in enumerate(tv_result):
                if res is None:
                    print(f"[EmbeddingMerge] grab_vectors: clip[{idx}] 结果为 None for '{text[:60]}'")
                    return None
                if len(res)==0:
                    print(f"[EmbeddingMerge] grab_vectors: clip[{idx}] 结果为空列表 for '{text[:60]}'")
                    res = text_to_vectors(',')[len(both)][0][0][0:0]
                else:
                    cat_list = [ten[0] for ten in res]
                    if len(cat_list)==0:
                        print(f"[EmbeddingMerge] grab_vectors: clip[{idx}] cat_list 为空 for '{text[:60]}'")
                        res = text_to_vectors(',')[len(both)][0][0][0:0]
                    else:
                        res = torch.cat(cat_list);
                        print(f"[EmbeddingMerge] grab_vectors: clip[{idx}] 合并 {len(cat_list)} 个张量 -> shape={res.shape}")
                both.append(res)
            if len(both)>1:
                if len(both[0]) != len(both[1]):
                    print(f"[EmbeddingMerge] grab_vectors: 两个 CLIP 向量长度不一致 {len(both[0])} vs {len(both[1])}")
                    return None
            print(f"[EmbeddingMerge] grab_vectors: 成功提取 {len(both)} 个 CLIP 结果, 向量数={len(both[0])}")
            return both
        except:
            traceback.print_exc()
            return None

    reg_clean = re.compile(r'\s+')
    reg_oper = re.compile(r'(=?)(?:([*/,])([+-]?[0-9]*(?:\.[0-9]*)?(?:L|G)?)|:([+-]?)(-?[0-9]+))')
    sdxl_sizes = {
      'L': 768,
      'G': 1280,
    }
    def merge_parser(text,only_count):
        clips = get_model_clips()
        vocab = None
        def check_vocab(token2):
            nonlocal vocab
            if vocab is None:
                vocab = []
                for clip in clips:
                    wrapped = clip.wrapped
                    typename = type(wrapped).__name__.split('.')[-1]
                    if typename=='FrozenCLIPEmbedder':
                        voc = wrapped.tokenizer.get_vocab()
                    elif typename=='FrozenOpenCLIPEmbedder':
                        voc = open_clip.tokenizer._tokenizer.encoder
                    else:
                        return True
                    vocab.append({v: k for k, v in voc.items()})
            t = token2[0]
            if len(vocab)>1:
                if len(token2)>1:
                    return (t in vocab[0]) and (token2[1] in vocab[1])
                return (t in vocab[0]) and (t in vocab[1])
            return t in vocab[0]
        orig = '"'+text+'"'
        text = text.replace('\0',' ')+' '
        length = len(text)
        arr = []
        left = 0
        quot = False
        join = False
        while left<length:
            pos = text.find("'",left)
            if pos<0:
                pos = length
            take = text[left:pos]
            if left>0:
                if take=='' and not quot:
                    join = True
                elif quot:
                    if join:
                        arr[-1] = (arr[-1][0]+"'"+take,True)
                        join = False
                    else:
                        arr.append((take,True))
                else:
                    arr.append((take.strip(),False))
            quot = not quot
            left = pos+1
        if not quot:
            return (None,'最后一个引号未关闭于 '+orig)
        if len(arr)>0 and arr[-1][0]=='':
            arr.pop()
        actions = []
        combine = False
        for param, quot in arr:
            one = param
            if quot:
                if combine:
                    actions[-1]['V'] = param
                    combine = False
                else:
                    actions.append({
                      'A': None,
                      'V': param,
                      'O': one,
                    })
                continue
            elif combine:
                return (None,'错误的连接 "'+param+'" 在 '+orig)
            param = reg_clean.sub('',param)
            while param!='':
                m = reg_oper.match(param)
                if not m:
                    if param=='+' or param=='-':
                        actions.append({
                          'A': False,
                          'V': param=='+',
                          'O': one,
                        })
                        break
                    return (None,'错误的表达式 "'+param+'" 在 '+orig)
                m_flag = m.group(1)=='='
                m_mul = m.group(2)
                m_val = m.group(3)
                m_shift = m.group(4)
                m_size = m.group(5)
                m_tok = -1
                m_clip = None
                if m_val is not None:
                    if len(m_val)>0:
                        m_clip = m_val[-1]
                        if (m_clip=='L') or (m_clip=='G'):
                            m_val = m_val[:-1]
                            if len(clips)<2:
                                return (None,'后缀 L 或 G 只能与 SDXL 模型一起使用： "'+param+'" 在 '+orig)
                        else:
                            m_clip = None
                    if m_mul==',':
                        if m_flag:
                            return (None,'连接不支持 \'=\' 前缀： "'+param+'" 在 '+orig)
                        if m_clip is not None:
                            return (None,'连接不支持 L 或 G 后缀： "'+param+'" 在 '+orig)
                        if (len(m_val)>0) and (m_val[0]=='0'):
                            if m_val=='0':
                                m_tok = 0
                            elif m_val=='00':
                                m_tok = -2
                            elif m_val=='000':
                                m_tok = -3
                            elif m_val=='0000':
                                m_tok = -4
                            else:
                                m_tok = None
                        elif m_val=='':
                            m_tok = -5
                            combine = True
                            m_val = None
                        else:
                            m_tok = to_int(m_val)
                            if (m_tok is not None) and not (m_tok>=0):
                                m_tok = None
                        if m_tok is None:
                            return (None,'连接参数错误 "'+param+'" 在 '+orig)
                    else:
                        m_val = to_float(m_val)
                        if m_val is None:
                            return (None,'乘法参数错误 "'+param+'" 在 '+orig)
                        m_mul = m_mul=='*'
                    m_size = -1
                    m_shift = 0
                else:
                    m_size = int(m_size)
                    if m_shift=='+':
                        m_shift = m_size
                        m_size = -1
                    elif m_shift=='-':
                        m_shift = -m_size
                        m_size = -1
                    else:
                        m_shift = 0
                    m_val = 1
                    m_mul = None
                actions.append({
                  'A': True,
                  'V': m_val,
                  'W': m_mul,
                  'S': m_size,
                  'R': m_shift,
                  'F': m_flag,
                  'T': m_tok,
                  'C': m_clip,
                  'O': one,
                })
                param = param[len(m.group(0)):]
        if combine:
            return (None,'未完成的连接于 '+orig)
        actions.append({
          'A': None,
          'V': None,
        })
        can_file = True
        can_add = False
        can_mul = False
        for act in actions:
            act['M'] = False
            A = act['A']
            if A==None:
                if act['V']==None:
                    if can_file:
                        return (None,'最后一个 + 或 - 之后需要带引号的字符串于 '+orig)
                    act['M'] = True
                    break
                if can_file:
                    can_add = True
                    can_mul = True
                    can_file = False
                else:
                    return (None,'带引号的字符串前缺少 + 或 - 于 \''+act['O']+'\' 在 '+orig)
            elif A==True:
                if can_mul:
                    can_file = False
                    can_add = True
                    can_mul = True
                    if act['F']:
                        act['M'] = True
                else:
                    return (None,'不能在此处进行乘法或修改 "'+act['O']+'" 于 '+orig)
            else:
                if can_add:
                    can_file = True
                    can_mul = False
                    can_add = False
                    act['M'] = True
                else:
                    return (None,'不能在此处合并 "'+act['O']+'" 于 '+orig)
        left = None
        right = None
        add = 0
        for act in actions:
            if act['M'] and (left is not None):
                if add!=0:
                    if only_count:
                        if left>right:
                            right = left
                    else:
                        (vectors1_0,length1_0) = left[0].size()
                        (vectors2_0,length2_0) = right[0].size()
                        (vectors1_1,length1_1) = left[1].size() if len(left)>1 else (vectors1_0,length1_0)
                        (vectors2_1,length2_1) = right[1].size() if len(right)>1 else (vectors2_0,length2_0)
                        if (length1_0!=length2_0) or (length1_1!=length2_1) or (vectors1_0!=vectors1_1) or (vectors2_0!=vectors2_1) or (len(left)!=len(right)):
                            return (None,'不能合并不同的嵌入于 '+orig)
                        if vectors1_0!=vectors2_0:
                            if vectors1_0<vectors2_0:
                                target = [torch.zeros(vectors2_0,length1_0).to(device=devices.device,dtype=torch.float32)]
                                target[0][0:vectors1_0] = left[0]
                                if len(left)>1:
                                    target.append(torch.zeros(vectors2_1,length1_1).to(device=devices.device,dtype=torch.float32))
                                    target[1][0:vectors1_1] = left[1]
                                left = target
                            else:
                                target = [torch.zeros(vectors1_0,length2_0).to(device=devices.device,dtype=torch.float32)]
                                target[0][0:vectors2_0] = right[0]
                                if len(right)>1:
                                    target.append(torch.zeros(vectors1_1,length2_1).to(device=devices.device,dtype=torch.float32))
                                    target[1][0:vectors2_1] = right[1]
                                right = target
                        if add>0:
                            right[0] = left[0]+right[0]
                            if len(left)>1 and len(right)>1:
                                right[1] = left[1]+right[1]
                        else:
                            right[0] = left[0]-right[0]
                            if len(left)>1 and len(right)>1:
                                right[1] = left[1]-right[1]
                left = None
            A = act['A']
            if A==None:
                line = act['V']
                if line==None:
                    return (right,None)
                right = grab_vectors(line)
                if right==None:
                    return (None,'解析失败 \''+line+'\' 于 '+orig)
                if only_count:
                    right = right[0].size(0)
            elif A==False:
                if act['V']:
                    add = 1
                else:
                    add = -1
                left = right
                right = None
            else:
                s = act['S']
                r = act['R']
                t = act['T']
                if only_count:
                    if t!=-1:
                        right += 1
                    elif (r==0)and(s>=0):
                        right = s
                else:
                    if t!=-1:
                        if t<0:
                            if t==-2:
                                t = [clip.id_pad for clip in clips]
                            elif t==-3:
                                t = [clip.id_end for clip in clips]
                            elif t==-4:
                                t = [clip.id_start for clip in clips]
                            else:
                                res = grab_vectors(act['V'])
                                t = None
                                if res is None:
                                    return (None,'解析失败 \''+act['V']+'\' 于 '+orig)
                        else:
                            if len(clips)>1:
                                t = [t,t]
                            else:
                                t = [t]
                        if t is not None:
                            if not check_vocab(t):
                                return (None,'未知的标记值 \''+str(t[0])+'\' 于 '+orig)
                            res = tokens_to_vectors(t)
                        if res is None:
                            return (None,'转换标记失败 \''+str(t)+'\' 于 '+orig)
                        if right is None:
                            right = res
                        else:
                            if len(right)>1 and len(res)>1:
                                right = [torch.cat([right[0],res[0]]),torch.cat([right[1],res[1]])]
                            else:
                                right = [torch.cat([right[0],res[0]])]
                    elif r!=0:
                        right[0] = right[0].roll(r,dims=0)
                        if len(right)>1:
                            right[1] = right[1].roll(r,dims=0)
                    else:
                        if s>=0:
                            (vectors,length) = right[0].size()
                            if vectors>s:
                                if len(right)>1:
                                    right = [right[0][0:s],right[1][0:s]]
                                else:
                                    right[0] = right[0][0:s]
                            elif vectors<s:
                                target = [torch.zeros(s,length).to(device=devices.device,dtype=torch.float32)]
                                target[0][0:vectors] = right[0]
                                if len(right)>1:
                                    (vectors,length) = right[1].size()
                                    target.append(torch.zeros(s,length).to(device=devices.device,dtype=torch.float32))
                                    target[1][0:vectors] = right[1]
                                right = target
                        elif act['W']==True:
                            if act['C']==None:
                                right = [r*act['V'] for r in right]
                            else:
                                s = sdxl_sizes[act['C']]
                                right = [(r*act['V'] if r.shape[-1]==s else r) for r in right]
                        elif  act['W']==False:
                            if act['C']==None:
                                right = [r/act['V'] for r in right]
                            else:
                                s = sdxl_sizes[act['C']]
                                right = [(r/act['V'] if r.shape[-1]==s else r) for r in right]
        if right is not None:
            if not only_count:
                print(f"[EmbeddingMerge] merge_parser: 合并完成, shape={[r.shape for r in right]}, 非零值={[r.abs().sum().item() for r in right]}")
        return (right,None)

    def grab_embedding_cache():
        db = get_embedding_db()[0]
        field = '__embedding_merge_cache_'
        if hasattr(db,field):
            cache = getattr(db,field)
        else:
            cache = {'_':0,'-':0,'/':0}
            setattr(db,field,cache)
        return cache

    def register_embedding(name,embedding):
        dbs = get_embedding_db()
        for self in dbs:
            model = shared.sd_model
            if hasattr(self,'register_embedding_by_name'):
                try:
                    result = self.register_embedding_by_name(embedding,model,name)
                    print(f"[EmbeddingMerge] register_embedding: '{name}' register_embedding_by_name(embedding,model,name) -> {result is not None}")
                    return result
                except TypeError:
                    try:
                        result = self.register_embedding_by_name(embedding,name)
                        print(f"[EmbeddingMerge] register_embedding: '{name}' register_embedding_by_name(embedding,name) -> {result is not None}")
                        return result
                    except Exception as e:
                        print(f"[EmbeddingMerge] register_embedding: '{name}' register_embedding_by_name 两参数版本也失败: {e}")
                        continue
                except Exception as e:
                    print(f"[EmbeddingMerge] register_embedding: '{name}' register_embedding_by_name 失败: {e}")
                    traceback.print_exc()
                    continue
            
            # Check if this is a raw Embedding layer (nn.Embedding) - can't use for registration
            if not hasattr(self, 'word_embeddings') and not hasattr(self, 'ids_lookup'):
                print(f"[EmbeddingMerge] register_embedding: '{name}' 跳过 - 当前 DB ({type(self).__name__}) 缺少 word_embeddings/ids_lookup")
                continue
            
            # Manual registration path (from /modules/textual_inversion/textual_inversion.py)
            try:
                ids = model.cond_stage_model.tokenize([name])[0]
                first_id = ids[0]
            except Exception as e:
                print(f"[EmbeddingMerge] register_embedding: '{name}' 无法 tokenize 名称: {e}")
                continue
            if embedding is None:
                if self.word_embeddings[name] is None:
                    return
                del self.word_embeddings[name]
            else:
                self.word_embeddings[name] = embedding
            if first_id not in self.ids_lookup:
                if embedding is None:
                    return
                self.ids_lookup[first_id] = []
            save = [(ids, embedding)] if embedding is not None else []
            old = [x for x in self.ids_lookup[first_id] if x[1].name!=name]
            self.ids_lookup[first_id] = sorted(old + save, key=lambda x: len(x[0]), reverse=True)
            print(f"[EmbeddingMerge] register_embedding: '{name}' 通过 word_embeddings fallback 注册成功, first_id={first_id}")
            return embedding
        
        print(f"[EmbeddingMerge] register_embedding: '{name}' 注册失败！尝试了 {len(dbs)} 个 DB，都不可用。")
        # Last resort for Forge: try text_processing_engine directly
        sd_model = shared.sd_model
        for attr in ('text_processing_engine_l', 'text_processing_engine_g'):
            if hasattr(sd_model, attr):
                engine = getattr(sd_model, attr)
                if hasattr(engine, 'embeddings') and hasattr(engine.embeddings, 'register_embedding_by_name'):
                    try:
                        result = engine.embeddings.register_embedding_by_name(embedding, sd_model, name)
                        print(f"[EmbeddingMerge] register_embedding: '{name}' 通过 {attr}.embeddings 注册 -> {result is not None}")
                        return result
                    except Exception as e:
                        print(f"[EmbeddingMerge] register_embedding: '{name}' {attr}.embeddings 注册失败: {e}")
        return None

    def make_temp_embedding(name,vectors,cache,fake):
        embed = None
        if name in cache:
            embed = cache[name]
            if fake>0:
                return
        else:
            if fake>0:
                if len(get_model_clips())>1:
                    vectors = [torch.zeros((fake,16)),torch.zeros((fake,16))]
                else:
                    vectors = [torch.zeros((fake,16))]
        shape = vectors[-1].size()
        if len(vectors)>1:
            vectors_dict = {'clip_g':vectors[1],'clip_l':vectors[0]}
            print(f"[EmbeddingMerge] make_temp_embedding: '{name}' SDXL 双 CLIP 嵌入, shape clip_l={vectors[0].shape}, clip_g={vectors[1].shape}, 非零值数={vectors[0].abs().sum().item():.6f}/{vectors[1].abs().sum().item():.6f}")
        else:
            vectors_dict = vectors[0]
            print(f"[EmbeddingMerge] make_temp_embedding: '{name}' 单 CLIP 嵌入, shape={vectors[0].shape}, 非零值数={vectors[0].abs().sum().item():.6f}")
        if embed is None:
            embed = Embedding(vectors_dict,name)
            cache[name] = embed
        embed.vec = vectors_dict
        embed.step = None
        embed.vectors = shape[0]
        embed.shape = shape[-1]
        embed.cached_checksum = None
        embed.filename = ''
        register_embedding(name,embed)

    def reset_temp_embeddings(prod,unregister):
        cache = grab_embedding_cache()
        num = cache[prod]
        cache[prod] = 0
        for a,b in (('<','>'),('{','}')):
            i = num
            while i>0:
                tgt = a+"'EM"+prod+str(i)+"'"+b
                if tgt in cache:
                    embed = cache[tgt]
                    if type(embed.vec)==dict:
                        for k,v in embed.vec.items():
                            embed.vec[k] = torch.zeros((0,v.shape[-1]),device=v.device)
                    else:
                        embed.vec = torch.zeros((0,embed.vec.shape[-1]),device=embed.vec.device)
                    embed.vectors = 0
                    embed.cached_checksum = None
                    del cache[tgt]
                    if unregister:
                        register_embedding(tgt,None)
                i = i-1
        return cache

    def add_temp_embedding(vectors,cache,prod,curly,fake):
        if fake>0:
            prod = '/'
            num = (cache[prod] or 0)
            if fake>num:
                cache[prod] = fake
            num = fake
        else:
            prod = '_' if prod else '-'
            num = 1+(cache[prod] or 0)
            cache[prod] = num
        name = "'EM"+prod+str(num)+"'"
        if curly:
            name = '{'+name+'}'
        else:
            name = '<'+name+'>'
        make_temp_embedding(name,vectors,cache,fake)
        return name

    def parse_infotext(text):
        orig = text
        text += '\n'
        pos = re.search(r"\bEmbeddingMerge:\s*(\"?[<{])'EM_",text)
        if pos is None:
            return (None,orig)
        head = text[:pos.span(0)[0]].rstrip()
        if len(head)>0 and head[-1]==',':
            head = head[:-1]
        text = text[pos.span(1)[0]:]
        if len(text)<2:
            return (None,orig)
        what = text[0]
        if what=='"':
            unquoted = None
        else:
            if what=='<':
                unquoted = '>'
            elif what=='{':
                unquoted = '}'
            else:
                return (None,orig)
        if unquoted is not None:
            stop = min_or_all(text.find(unquoted+','),text.find(unquoted+'\n'),-1)
            if stop<0:
                return (None,orig)
            stop += 1
            tail = text[stop:]
            line = text[:stop]
        else:
            stop = (text+'\n').find('\n')
            part = text[:stop]
            left = 0
            while True:
                right = part.find('"',left+1)
                if right<0:
                    return (None,orig)
                try:
                    line = json.loads('['+part[:right+1].strip()+']')[0]
                    break
                except:
                    left = right
            tail = part[right+1:]+text[stop:]
        return (line,head+tail)

    def parse_mergeseq(seq):
        res = None
        seq = seq.lstrip()
        while True:
            left = seq[0:5]
            if left=="<'EM_":
                right = "'>="
            elif left=="{'EM_":
                right = "'}="
            else:
                return res
            stop = seq.find(right)
            if stop<1:
                return res
            what = seq[0:stop+2]
            seq = seq[stop+3:]
            left = seq[0:2]
            if left=="<'":
                right = '>, '
            elif left=="{'":
                right = '}, '
            else:
                return res
            stop = min_or_all(seq.find(right+"<'"),seq.find(right+"{'"),len(seq))
            repl = seq[0:stop+1]
            seq = seq[stop+3:]
            if res is None:
                res = {}
            res[what] = repl

    def min_or_all(a,b,n):
        if a>=0:
            if b>=0:
                if a<b:
                    return a
                return b
            else:
                return a
        elif b>=0:
            return b
        return n

    def dict_replace(di,text):
        for key in di:
            text = text.replace(key,di[key])
        return text

    gr_lock = threading.Lock()

    def gr_func(gr_name,gr_text,gr_radio,gr_tensors,store):
        with gr_lock:
            try:
                sd_models.reload_model_weights()
            except:
                pass
            try:
                sd_models.forge_model_reload()
            except:
                pass
            gr_orig = gr_text
            font = 'font-family:Consolas,Courier New,Courier,monospace;'
            table = '<style>.webui_embedding_merge_table,.webui_embedding_merge_table td,.webui_embedding_merge_table th{border:1px solid gray;border-collapse:collapse}.webui_embedding_merge_table td,.webui_embedding_merge_table th{padding:2px 5px !important;text-align:center !important;vertical-align:middle;'+font+'font-weight:bold;}.webui_embedding_merge_table{margin:6px auto !important;}</style>'
            (reparse,request) = parse_infotext(gr_text)
            if reparse is not None:
                reparse = parse_mergeseq(reparse)
                if reparse is None:
                    return ('<center><b>提示恢复失败！</b></center>',gr_name,gr_orig)
                else:
                    request = dict_replace(reparse,request)
                    return ('<center><b>提示已恢复。</b></center>',gr_name,request)
            if gr_text[:1]=="'":
                (two,err) = merge_parser(gr_text,False)
                if (two is not None) and two[0].numel()==0:
                    err = '结果是零向量！'
                if err is not None:
                    txt = '<b style="'+font+'">'+html.escape(err)+'</b>'
                else:
                    txt = table
                    both = False
                    for res in two:
                        if res is None:
                            continue
                        if both:
                            txt += '<strong>↑ CLIP (L) / OpenClip (G) ↓</strong>'
                        txt += '<table class="webui_embedding_merge_table"><tr><th>索引</th><th>最小值</th><th>最大值</th><th>总和</th><th>绝对值</th><th>长度</th><th>标准差</th>'
                        i = 1
                        for one in res:
                            txt += '<tr><td>{}</td>{}</tr>'.format(i,tensor_info(one))
                            i += 1
                        txt += '<tr><td colspan="7">&nbsp;</td></tr>'
                        txt += '<tr><td>全部:</td>{}</tr>'.format(tensor_info(res))
                        txt += '</table>'
                        both = True
                if store:
                    for idx, t in enumerate(two):
                        print(f"[EmbeddingMerge] gr_func: merge_parser 结果 two[{idx}] shape={t.shape}, 非零值={t.abs().sum().item():.6f}, 零向量={t.numel()==0 or t.abs().max().item()==0}")
                return ('<center>'+txt+'</center>',need_save_embed(store,gr_name,two,gr_tensors),gr_orig)
            if gr_text.find("<'")>=0 or gr_text.find("{'")>=0:
                cache = reset_temp_embeddings('-',False)
                used = {}
                (mer,err) = merge_one_prompt(cache,None,{},used,gr_text,False,False)
                if err is not None:
                    txt = '<b style="'+font+'">嵌入合并失败 - '+html.escape(err)+'</b>'
                    return ('<center>'+txt+'</center>',gr_name,gr_orig)
                gr_text = mer
            by_none = 0
            by_comma = 1
            by_parts = 2
            by_words = 3
            by_tokens = 4
            by_vectors = 5
            tok2txt = tokens_to_text()
            if gr_radio!=by_comma:
                two = text_to_vectors(gr_text)
                if (gr_radio==by_none) and (two is not None) and (len(two[0])!=0):
                    two = [[r] for r in two]
            else:
                two = [[],[]]
                split = gr_text.split(',')
                for part in split:
                    one = text_to_vectors(part.strip())
                    if one:
                        two[0].append(one[0])
                        if(len(one)>1):
                            two[1].append(one[1])
                        else:
                            two[1] = None
                    else:
                        two = None
                        break
            if (two is None) or (len(two[0])==0):
                if gr_text.strip()=='':
                    return ('',gr_name,gr_orig)
                txt = '<b>解析失败！（可能嵌入名称内有额外空格或标记解析异常）。现在不显示嵌入向量（仅显示文本标记）：</b><br/><br/>'
                tokens = text_to_tokens(gr_text)
                if tokens:
                    txt += table+'<tr><th>索引</th><th>向量数</th><th>文本</th><th>标记</th></tr>'
                    if tok2txt:
                        pairs = tok2txt(tokens)
                    else:
                        pairs = [([tok],'<ERROR>') for tok in tokens]
                    index = 1
                    for arr, text in pairs:
                        length = len(arr)
                        if length==0:
                            continue
                        txt += '<tr><td>'+(str(index) if length==1 else str(index)+'-'+str(index+length-1))+'</td><td>'+str(length)+'</td><td>'+html.escape('"'+text+'"')+'</td><td>'+(', '.join([str(a) for a in arr]))+'</td></tr>'
                        index += length
                    txt += '</table>'
                return ('<center>'+txt+'</center>',gr_name,gr_orig)
            both = []
            for res in two:
                if res is None:
                    continue
                txt = '<table class="webui_embedding_merge_table"><tr><th>索引</th><th>向量数</th><th>文本</th><th>标记</th><th>最小值</th><th>最大值</th><th>总和</th><th>绝对值</th><th>长度</th><th>标准差</th></tr>'
                index = 1
                join = False
                if gr_radio==by_words:
                    join = True
                    gr_radio = by_tokens
                elif (gr_radio==by_none) or (gr_radio==by_comma):
                    r_res = []
                    for one in res:
                        r_tensor = []
                        r_name = ''
                        r_tokens = []
                        for tensor, name, tokens in one:
                            r_tensor.append(tensor)
                            if tok2txt and tokens and gr_radio==by_none:
                                split = tok2txt(tokens)
                                name = ''
                                tokens = []
                                for s_tokens, s_name in split:
                                    name += s_name
                                    tokens += s_tokens
                            r_name += name
                            if tokens:
                                r_tokens += tokens
                            else:
                                r_tokens += ['*_'+str(tensor.size(0))]
                                if gr_radio==by_none:
                                    r_name += ' '
                        if len(r_tensor)>0:
                            r_res.append((torch.cat(r_tensor),r_name,r_tokens))
                    res = r_res
                    gr_radio = by_parts
                for tensor, name, tokens in res:
                    split = None
                    size = tensor.size(0)
                    span = ''
                    if gr_radio!=by_parts:
                        span = ' rowspan="'+str(size)+'"'
                        if tokens and tok2txt:
                            split = tok2txt(tokens)
                            if join:
                                comb = []
                                last = -1
                                for s_arr, s_text in split:
                                    if (last<0) or (comb[last][1][-1:]==' '):
                                        comb.append((s_arr,s_text))
                                        last += 1
                                    else:
                                        comb[last] = (comb[last][0]+s_arr,comb[last][1]+s_text)
                                split = comb
                        if gr_radio==by_tokens:
                            if split is not None:
                                span = ' rowspan="'+str(len(split))+'"'
                            else:
                                span = ''
                    if gr_radio==by_vectors:
                        head = '<td'+span+'>'+str(size)+'</td>'
                    else:
                        head = '<td'+span+'>'+(str(index) if size==1 else str(index)+'-'+str(index+size-1))+'</td><td'+span+'>'+str(size)+'</td>'
                    if split is None:
                        head += '<td'+span+'>'+html.escape('"'+name+'"')+'</td>'
                    if (gr_radio==by_vectors) or ((gr_radio==by_tokens) and (tokens is not None)):
                        i = 0
                        part = 0
                        j = 0
                        ten = None
                        column = ''
                        toks = None
                        for one in list(tensor):
                            index += 1
                            i += 1
                            use = one
                            if split is not None:
                                if part==0:
                                    pair = split[j]
                                    part = len(pair[0])
                                    if gr_radio==by_tokens:
                                        column = '<td>'+html.escape('"'+pair[1]+'"')+'</td>'
                                        toks = ', '.join([str(t) for t in pair[0]])
                                    else:
                                        column = '<td rowspan="'+str(part)+'">'+html.escape('"'+pair[1]+'"')+'</td>'
                                    j += 1
                            part -= 1
                            if gr_radio==by_tokens:
                                if ten==None:
                                    ten = []
                                ten.append(one)
                                if part>0:
                                    continue
                                use = torch.stack(ten)
                                tok = toks if tokens else '*'
                            else:
                                tok = tokens[i-1] if tokens else '*_'+str(i)
                            txt += '<tr>{}{}<td>{}</td>{}</tr>'.format(('<td>'+str(index-1)+'</td>' if gr_radio==by_vectors else '')+head,column,tok,tensor_info(use))
                            column = ''
                            head = ''
                            ten = None
                    else:
                        index += size   
                        txt += '<tr>{}<td>{}</td>{}</tr>'.format(head,', '.join([str(t) for t in tokens]) if tokens else '*',tensor_info(tensor))
                txt += '</table>'
                both.append(txt)
            txt = table+'<strong>↑ CLIP (L) / OpenClip (G) ↓</strong>'.join(both)
            return ('<center>'+txt+'</center>',need_save_embed(store,gr_name,two,gr_tensors),gr_orig)

    def tensor_info(tensor):
        return '<td>{:>-14.8f}</td><td>{:>+14.8f}</td><td>{:>+14.8f}</td><td>{:>14.8f}</td><td>{:>14.8f}</td><td>{:>14.8f}</td>'.format(tensor.min().item(),tensor.max().item(),tensor.sum().item(),tensor.abs().sum().item(),torch.linalg.norm(tensor,ord=2),tensor.std()).replace(' ','&nbsp;')

    merge_dir = None

    def need_save_embed(store,name,pair,tensors):
        if not store:
            return name
        name = ''.join( x for x in name if (x.isalnum() or x in '._- ')).strip()
        if name=='':
            return name
        try:
            # Helper: extract tensor from potentially nested structures
            # pair[r] could be: tensor, (tensor, text, tokens), or [(tensor, text, tokens)]
            def extract_tensor(item):
                t = item
                while isinstance(t, (list, tuple)) and not torch.is_tensor(t):
                    if len(t) == 0:
                        return None
                    t = t[0]
                return t if torch.is_tensor(t) else None

            print(f"[EmbeddingMerge] need_save_embed: 开始保存 '{name}', tensors={tensors}, clip数={len(pair)}")
            if type(pair[0])==list:
                r0_list = []
                for r in pair[0]:
                    t = extract_tensor(r)
                    if t is not None:
                        r0_list.append(t)
                    else:
                        print(f"[EmbeddingMerge] need_save_embed: 无法从 pair[0]元素提取张量: {type(r).__name__}")
                if len(r0_list)==0:
                    print(f"[EmbeddingMerge] need_save_embed: r0_list 为空，放弃保存")
                    return name
                vectors = [torch.cat(r0_list)]
                if (len(pair)>1) and (pair[1] is not None):
                    r1_list = []
                    for r in pair[1]:
                        t = extract_tensor(r)
                        if t is not None:
                            r1_list.append(t)
                    if len(r1_list)>0:
                        vectors.append(torch.cat(r1_list))
                    else:
                        print(f"[EmbeddingMerge] need_save_embed: r1_list 为空（pair[1]数据丢失？），仅保存 clip_l")
            else:
                vectors = [pair[0]]
                if (len(pair)>1) and (pair[1] is not None):
                    vectors.append(pair[1])
            for i, v in enumerate(vectors):
                print(f"[EmbeddingMerge] need_save_embed: vectors[{i}] shape={v.shape}, 非零值数={v.abs().sum().item():.6f}, max={v.abs().max().item():.6f}")
            if len(vectors)>1:
                print(f"[EmbeddingMerge] need_save_embed: SDXL 双CLIP模式, clip_l.shape={vectors[0].shape}, clip_g.shape={vectors[1].shape}")

            # 检查总向量数是否超过 CLIP 上限，超限则自动池化压缩
            total_vecs = max(v.size(0) for v in vectors)
            if total_vecs > CLIP_MAX_TOKENS:
                orig_text = None
                if type(pair[0]) == list and len(pair[0]) > 0:
                    text_parts = []
                    for item in pair[0]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            text_parts.append(str(item[1]))
                    orig_text = ''.join(text_parts)

                if orig_text and len(orig_text) > 0:
                    pooled_to = POOL_TARGET_VECTORS
                    print(f"[EmbeddingMerge] need_save_embed: 向量数 {total_vecs} 超过 CLIP 上限 {CLIP_MAX_TOKENS}，对文本进行池化压缩 -> {pooled_to} 个向量")
                    new_vectors = []
                    for idx, clip in enumerate(get_model_clips()):
                        pooled = encode_embedding_init_text(clip, orig_text, pool_to=pooled_to)
                        if pooled is not None:
                            new_vectors.append(pooled.to(device=devices.device, dtype=torch.float32))
                            print(f"[EmbeddingMerge] need_save_embed: clip[{idx}] 池化后 shape={pooled.shape}")
                        elif idx < len(vectors):
                            # 池化失败则截断
                            capped = vectors[idx][:CLIP_MAX_TOKENS]
                            new_vectors.append(capped)
                            print(f"[EmbeddingMerge] need_save_embed: clip[{idx}] 池化失败，截断到 shape={capped.shape}")
                    if len(new_vectors) > 0:
                        vectors = new_vectors
                else:
                    # 无法获取文本，直接截断
                    print(f"[EmbeddingMerge] need_save_embed: 无法获取原始文本进行池化，截断向量到 {CLIP_MAX_TOKENS}")
                    vectors = [v[:CLIP_MAX_TOKENS] for v in vectors]

            if tensors:
                # ---- Save .safetensors -> embeddings/embedding_merge/sdxl/ ----
                from safetensors.torch import save_file
                st_dir = os.path.join(merge_dir, 'sdxl')
                os.makedirs(st_dir, exist_ok=True)
                if len(vectors)>1:
                    vec_l = vectors[0]
                    vec_g = vectors[1]
                    save_file({'clip_g': vec_g.cpu(), 'clip_l': vec_l.cpu()},
                              os.path.join(st_dir, name) + '.safetensors')
                    print(f"[EmbeddingMerge] need_save_embed: .safetensors sdxl/{name}.safetensors clip_g.shape={vec_g.shape}, clip_l.shape={vec_l.shape}")
                else:
                    vector = vectors[0]
                    shape = vector.shape[-1]
                    if shape == 768:
                        s_g = list(vector.size()); s_g[-1] = 1280
                        save_file({'clip_g': torch.zeros(s_g).cpu(), 'clip_l': vector.cpu()},
                                  os.path.join(st_dir, name) + '.safetensors')
                    elif shape == 1280:
                        s_l = list(vector.size()); s_l[-1] = 768
                        save_file({'clip_g': vector.cpu(), 'clip_l': torch.zeros(s_l).cpu()},
                                  os.path.join(st_dir, name) + '.safetensors')
                    else:
                        save_file({'emb_params': vector.cpu()},
                                  os.path.join(st_dir, name) + '.safetensors')
                    print(f"[EmbeddingMerge] need_save_embed: .safetensors sdxl/{name}.safetensors shape={vector.shape}")
            else:
                # ---- Save .pt -> embeddings/embedding_merge/ ----
                target_pt = os.path.join(merge_dir, name)
                if len(vectors)>1:
                    pt_data = {'clip_g': vectors[1].cpu(), 'clip_l': vectors[0].cpu()}
                else:
                    pt_data = {
                        'string_to_token': {'*': 265},
                        'string_to_param': {'*': vectors[0].cpu()},
                        'name': name, 'step': 0,
                        'sd_checkpoint': None, 'sd_checkpoint_name': None,
                    }
                torch.save(pt_data, target_pt + '.pt')
                print(f"[EmbeddingMerge] need_save_embed: .pt 已保存 -> {target_pt}.pt")

            for db in get_embedding_db():
                try:
                    db.add_embedding_dir(merge_dir)
                except:
                    pass
                try:
                    db.load_textual_inversion_embeddings(force_reload=True)
                except:
                    db.load_textual_inversion_embeddings()
            print(f"[EmbeddingMerge] need_save_embed: '{name}' 保存成功!")
            return ''
        except:
            traceback.print_exc()
            return name

    def embedding_merge_dir():
        try:
            nonlocal merge_dir
            merge_dir = os.path.join(cmd_opts.embeddings_dir,'embedding_merge')
            # 显式注册到所有 embedding 数据库，确保子目录能被扫描到
            for db in get_embedding_db():
                try:
                    db.add_embedding_dir(merge_dir)
                except:
                    pass
            os.makedirs(merge_dir, exist_ok=True)
        except:
            pass

    def raise_sd_error(p,msg):
        class Exception_From_EmbeddingMergeExtension_():
            def __getattribute__(self,_):
                raise Exception_From_EmbeddingMergeExtension(msg)
        p.__class__ = Exception_From_EmbeddingMergeExtension_

    em_regexp = re.compile(r"<'EM[_/-]\d+'>|{'EM[_/-]\d+'}")

    def merge_one_prompt(cache,texts,parts,used,prompt,prod,only_count):
        #if len(get_model_clips())>1:
        #    return (None,'要启用 SDXL 支持，请切换到 https://github.com/klimaleksus/stable-diffusion-webui-embedding-merge 的 "sdxl" 分支')
        try:
            cnt = 0
            if (prompt is None) or (prompt==''):
                return (prompt,None)
            if texts is not None:
                if prompt in texts:
                    return (texts[prompt],None)
            orig = prompt
            has_expr = "<'" in prompt or "{'" in prompt
            # print(f"[EmbeddingMerge] merge_one_prompt: 处理提示 '{prompt[:100]}'... (含表达式: {has_expr}, only_count={only_count})")
            if not has_expr:
                if texts is not None:
                    texts[orig] = prompt
                return (prompt,None)
            left = 0
            merge_count = 0
            while True:
                curly = prompt.find("{'",left)
                left = prompt.find("<'",left)
                if (curly>=0 and curly<left) or (left<0):
                    left = curly
                    curly = True
                else:
                    curly = False
                if left<0:
                    if texts is not None:
                        texts[orig] = prompt
                    if merge_count>0:
                        pass  # print(f"[EmbeddingMerge] merge_one_prompt: 共处理 {merge_count} 个合并表达式, 最终提示: {prompt[:120]}...")
                    return (prompt,None)
                eph = em_regexp.match(prompt[left:])
                if eph is not None:
                    left += len(eph.group(0))
                    continue
                right = left
                while True:
                    right = prompt.find('}' if curly else '>',right+1)
                    if right<0:
                        if curly:
                            return (None,'未找到闭合的 "}" 在 "{\'" 之后')
                        else:
                            return (None,'未找到闭合的 ">" 在 "<\'" 之后')
                    if (prompt.count("'",left,right)&1)==0:
                        break
                part = prompt[left+1:right].strip()
                # print(f"[EmbeddingMerge] merge_one_prompt: 发现合并表达式 '{part}' (only_count={only_count})")
                if part in parts:
                    embed = parts[part]
                else:
                    (res,err) = merge_parser(part,only_count)
                    if err is not None:
                        # print(f"[EmbeddingMerge] merge_one_prompt: 解析失败: {err}")
                        return (None,err)
                    if only_count:
                        if (res is None) or (res==0):
                            embed = ''
                        else:
                            embed = add_temp_embedding(None,cache,prod,curly,res)
                            # print(f"[EmbeddingMerge] merge_one_prompt: only_count 模式, 向量数={res}, 嵌入名={embed}")
                    else:
                        if (res is None) or (res[0].numel()==0):
                            embed = ''
                            # print(f"[EmbeddingMerge] merge_one_prompt: 向量为空，跳过")
                        else:
                            embed = add_temp_embedding(res,cache,prod,curly,0)
                            # print(f"[EmbeddingMerge] merge_one_prompt: 合并完成, 向量 shape={[r.shape for r in res]}, 嵌入名={embed}")
                    if used is not None:
                        used[embed] = part
                    parts[part] = embed
                merge_count += 1
                prefix = prompt[:left].rstrip()+' '+embed
                left = len(prefix)
                prompt = prefix+' '+(prompt[right+1:].lstrip())
        except:
            traceback.print_exc()
            return (None,'致命错误？')

    fake_cached_params_counter = time.time()
    def fake_cached_params(self,*ar,**kw):
        nonlocal fake_cached_params_counter
        fake_cached_params_counter += 1
        return (*(self.em_orig_cached_params(*ar,**kw)),id(_webui_embedding_merge_),fake_cached_params_counter)

    cached_state = None

    '''
    import hunter
    @hunter.wrap(local=True,actions=[hunter.VarsSnooper,hunter.CallPrinter])
    def pretty_print(clas, indent=0, dupl=None):
        if dupl is None:
            dupl = {}
        me = id(clas)
        tab = ' ' * indent
        if clas is None:
            print(tab + ': None')
            return
        print(tab +  type(clas).__name__ +  ':')
        indent += 4
        tab = ' ' * indent
        if me in dupl:
            print(tab + '[CIRCULAR]')
            return
        dupl[me] = True
        for k,v in clas.__dict__.items():
            if '__dict__' in dir(v):
                pretty_print(v,indent,dupl)
            else:
                print(tab +  k + ': ' + str(v))
    import code
    code.interact(local=locals())
    '''

    def hook_infotext(hook):
        if hasattr(processing,'create_infotext'):
            field = '__embedding_merge_wrapper'
            old = getattr(processing,'create_infotext')
            if hasattr(old,field):
                old = getattr(old,field)
                if not hook:
                    setattr(processing,'create_infotext',old)
            if hook:
                def create_infotext(p,*ar,**kw):
                    res = old(p,*ar,**kw)
                    if 'EmbeddingMerge' in p.extra_generation_params:
                        (reparse,request) = parse_infotext(res)
                        if reparse is not None:
                            parse = parse_mergeseq(reparse)
                            matches = em_regexp.findall(request)
                            if (matches is not None) and len(matches)>0:
                                used = {}
                                for match in matches:
                                    used[match] = True
                                gen = ''
                                drop = False
                                for embed,text in parse.items():
                                    if embed in used:
                                        gen += embed+'='+text+', '
                                    else:
                                        drop = True
                                if gen!='' and drop:
                                    gen = gen[:-2]
                                    orig = p.extra_generation_params['EmbeddingMerge']
                                    if gen!=orig:
                                        p.extra_generation_params['EmbeddingMerge'] = gen
                                        res = old(p,*ar,**kw)
                                        p.extra_generation_params['EmbeddingMerge'] = orig
                    return res
                setattr(create_infotext,field,old)
                setattr(processing,'create_infotext',create_infotext)

    def embedding_merge_extension(p,processed):
        if processed is not None:
            hook_infotext(False)
            return
        hook_infotext(True)
        nonlocal cached_state
        use_hr = hasattr(p,'hr_prompt')
        arr = [
            p.all_prompts,
            p.prompt if type(p.prompt)==list else [p.prompt],
            p.all_negative_prompts,
            p.negative_prompt if type(p.negative_prompt)==list else [p.negative_prompt],
        ]
        if use_hr:
            arr += [
                p.all_hr_prompts,
                p.hr_prompt if type(p.hr_prompt)==list else [p.hr_prompt],
                p.all_hr_negative_prompts,
                p.hr_negative_prompt if type(p.hr_negative_prompt)==list else [p.hr_negative_prompt],
            ]
        restart = True
        if 'EmbeddingMerge' in p.extra_generation_params:
            restart = False
        elif em_regexp.search(' '.join([' '.join(one) for one in arr if one is not None])) is not None:
            restart = False
            # print("[EmbeddingMerge] 警告：检测到临时嵌入（如 <'EM_1'>）！")
        if restart or (cached_state is None):
            # print(f"[EmbeddingMerge] embedding_merge_extension: 初始化/重置缓存 (restart={restart}, cached_state={'空' if cached_state is None else '存在'})")
            cached_state = {
                'cache': reset_temp_embeddings('_',False),
                'texts': {},
                'parts': {},
                'used': {},
            }
        cache = cached_state['cache']
        texts = cached_state['texts']
        parts = cached_state['parts']
        used = cached_state['used']
        # print(f"[EmbeddingMerge] embedding_merge_extension: 开始处理提示, 模型类型: {'SDXL' if len(get_model_clips())>1 else 'SD1/SD2'}")
        for idx_one, one in enumerate(arr):
            if one is not None:
                for idx_p, prompt_text in enumerate(one):
                    if prompt_text:
                        has_expr = "<'" in prompt_text or "{'" in prompt_text
                        pass  # print(f"[EmbeddingMerge] embedding_merge_extension: arr[{idx_one}][{idx_p}] = '{prompt_text[:120]}' (含表达式: {has_expr})")
        for one in arr:
            ok = False
            fail = None
            if one is not None:
                for i in range(len(one)):
                    orig_prompt = one[i]
                    (res,err) = merge_one_prompt(cache,texts,parts,used,one[i],True,False)
                    if err is not None:
                        if fail is None:
                            fail = err
                        # print(f"[EmbeddingMerge] embedding_merge_extension: 提示处理失败: {err}")
                    else:
                        one[i] = res
                        ok = True
                        if res != orig_prompt:
                            pass  # print(f"[EmbeddingMerge] embedding_merge_extension: 提示已修改: '{orig_prompt[:80]}...' -> '{res[:80]}...'")
            if not ok and fail is not None:
                raise_sd_error(p,'\n\n嵌入合并失败 - '+err+'\n')
                return
        p.all_prompts = arr[0]
        p.all_negative_prompts = arr[2]
        p.prompt = arr[1] if type(p.prompt)==list else arr[1][0]
        p.negative_prompt = arr[3] if type(p.negative_prompt)==list else arr[3][0]
        if use_hr:
            p.all_hr_prompts = arr[4]
            p.all_hr_negative_prompts = arr[6]
            p.hr_prompt = arr[5] if type(p.hr_prompt)==list else arr[5][0]
            p.hr_negative_prompt = arr[7] if type(p.hr_negative_prompt)==list else arr[7][0]
        gen = ''
        was_used = False
        for embed in used:
            was_used = True
            if embed!='':
                if embed[0]=='<':
                    gen += embed+'=<'+used[embed]+'>, '
                else:
                    gen += embed+'={'+used[embed]+'}, '
        if gen!='':
            p.extra_generation_params['EmbeddingMerge'] = gen[:-2]
            # print(f"[EmbeddingMerge] embedding_merge_extension: EmbeddingMerge 参数 = {gen[:-2]}")
        if was_used:
            # print(f"[EmbeddingMerge] embedding_merge_extension: 共使用了 {len(used)} 个临时嵌入，已设置 cached_params 钩子")
            orig = getattr(p,'cached_params',None)
            if orig is not None:
                setattr(p,'em_orig_cached_params',orig)
                setattr(p,'cached_params',types.MethodType(fake_cached_params,p))
        else:
            pass  # print(f"[EmbeddingMerge] embedding_merge_extension: 未检测到任何合并表达式")

    try:
        cls = modules.sd_hijack.StableDiffusionModelHijack
        get_prompt_lengths = cls.get_prompt_lengths
        field = '__embedding_merge_wrapper'
        def hook_prompt_lengths(self,text,*ar,**kw):
            has_expr = text.find("<'")>=0 or text.find("{'")>=0
            # print(f"[EmbeddingMerge] hook_prompt_lengths: text='{str(text)[:100]}'... (含表达式: {has_expr})")
            if not has_expr:
                return get_prompt_lengths(self,text,*ar,**kw)
            (res,err) = merge_one_prompt(grab_embedding_cache(),None,{},None,text,True,True)
            # print(f"[EmbeddingMerge] hook_prompt_lengths: 合并结果='{str(res)[:100]}'... err={err}")
            if err is not None:
                return -1,-1
            return get_prompt_lengths(self,res,*ar,**kw)
        if hasattr(get_prompt_lengths,field):
            get_prompt_lengths = getattr(get_prompt_lengths,field)
        setattr(hook_prompt_lengths,field,get_prompt_lengths)
        cls.get_prompt_lengths = hook_prompt_lengths
    except:
        traceback.print_exc()

    def on_infotext_pasted(infotext,result):
        if 'EmbeddingMerge' in result:
            reparse = result['EmbeddingMerge']
            if reparse[:1]=='"':
                try:
                    reparse = json.loads('['+reparse.strip()+']')[0]
                    reparse = parse_mergeseq(reparse)
                except:
                    reparse = None
            else:
                reparse = parse_mergeseq(reparse)
            request = None
        else:
            (reparse,request) = parse_infotext(infotext)
            if reparse is not None:
                reparse = parse_mergeseq(reparse)
        if reparse is not None:
            if 'Prompt' in result:
                if (request is not None) and (result['Prompt']==infotext):
                    result['Prompt'] = request
                result['Prompt'] = dict_replace(reparse,result['Prompt'])
            if 'Negative prompt' in result:
                result['Negative prompt'] = dict_replace(reparse,result['Negative prompt'])
            if 'Hires prompt' in result:
                result['Hires prompt'] = dict_replace(reparse,result['Hires prompt'])
            if 'Hires negative prompt' in result:
                result['Hires negative prompt'] = dict_replace(reparse,result['Hires negative prompt'])
    setattr(_webui_embedding_merge_,'on_infotext_pasted',on_infotext_pasted)
    def on_model_loaded(*ar,**kw):
        reset_temp_embeddings('/',True)
    setattr(_webui_embedding_merge_,'on_model_loaded',on_model_loaded)

    def on_script_unloaded():
        hook_infotext(False)
        reset_temp_embeddings('_',True)
        reset_temp_embeddings('-',True)
        reset_temp_embeddings('/',True)
        try:
            cls = modules.sd_hijack.StableDiffusionModelHijack
            get_prompt_lengths = cls.get_prompt_lengths
            field = '__embedding_merge_wrapper'
            if hasattr(get_prompt_lengths,field):
                cls.get_prompt_lengths = getattr(get_prompt_lengths,field)
        except:
            traceback.print_exc()
        try:
            db = get_embedding_db()[0]
            field = '__embedding_merge_cache_'
            if hasattr(db,field):
                delattr(db,field)
        except:
            traceback.print_exc()
    setattr(_webui_embedding_merge_,'on_script_unloaded',on_script_unloaded)
    setattr(_webui_embedding_merge_,'embedding_merge_extension',embedding_merge_extension)
    embedding_merge_dir()
    return gr_tab

class EmbeddingMergeExtension(scripts.Script):
    def title(self):
        return '嵌入合并'
    def show(self,is_img2img):
        return scripts.AlwaysVisible
    def process(self,p):
        # print(f"[EmbeddingMerge] Script.process() 被调用, p.prompt='{str(p.prompt)[:150]}', is_img2img={self.is_img2img if hasattr(self,'is_img2img') else '?'}")
        if hasattr(_webui_embedding_merge_,'embedding_merge_extension'):
            getattr(_webui_embedding_merge_,'embedding_merge_extension')(p,None)
    def postprocess(self,p,processed):
        # print(f"[EmbeddingMerge] Script.postprocess() 被调用, p.prompt='{str(p.prompt)[:150]}'")
        if hasattr(_webui_embedding_merge_,'embedding_merge_extension'):
            getattr(_webui_embedding_merge_,'embedding_merge_extension')(p,processed)

script_callbacks.on_ui_tabs(_webui_embedding_merge_())
script_callbacks.on_infotext_pasted(_webui_embedding_merge_.on_infotext_pasted)
script_callbacks.on_script_unloaded(_webui_embedding_merge_.on_script_unloaded)
try:
    script_callbacks.on_model_loaded(_webui_embedding_merge_.on_model_loaded)
except:
    pass

#EOF