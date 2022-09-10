# A.Note1:词向量、SVD分解与Word2Vec

CS224n是顶级院校斯坦福出品的深度学习与自然语言处理方向专业课程，核心内容覆盖RNN、LSTM、CNN、transformer、bert、问答、摘要、文本生成、语言模型、阅读理解等前沿内容。

本篇笔记对应斯坦福CS224n自然语言处理专项课程的第1个知识板块：NLP与词向量。首先介绍了自然语言处理(NLP)的概念及其面临的问题，进而介绍词向量和其构建方法（包括基于共现矩阵降维和Word2Vec）。

本节内容：
- 自然语言处理/Natural Language Processing(NLP)
- 词向量/Word Vectors
- SVD矩阵分解
- Skip-gram
- 负例采样
- transformer
- CBOW
- 层次化softmax
- Word2Vec
## 1.自然语言处理介绍
### 1.1 自然语言处理的特别之处

人类的语言有什么特别之处？人类语言是一个专门用来表达意义的系统，语言文字是上层抽象表征，NLP与计算机视觉或任何其他机器学习任务都有很大的不同。

大多数单词只是一个语言学以外的符号：单词是一个映射到所指(signified 想法或事物)的能指(signifier)。例如，“rocket”一词指的是火箭的概念，因此可以引申为火箭的实例。当我们使用单词和字母来表达符号时，也会有一些例外，例如“whoompaa”的使用。

最重要的是，这些语言的符号可以被编码成几种形式：声音、手势、文字等等，然后通过连续的信号传输给大脑；大脑本身似乎也能以一种连续的方式对这些信号进行解码。人们在语言哲学和语言学方面做了大量的工作来概念化人类语言，并将词语与其参照、意义等区分开来。

Natural language is a discrete[离散的] / symbolic[符号的] / categorical[分类的] system.

### 1.2 自然语言处理任务

自然语言处理有不同层次的任务，从语言处理到语义解释再到语篇处理。自然语言处理的目标是通过设计算法使得计算机能够“理解”语言，从而能够执行某些特定的任务。不同的任务的难度是不一样的：

- 简单任务
  - 拼写检查 Spell Checking
  - 关键词检索 Keyword Search
  - 同义词查找 Finding Synonyms
- 中级任务
  - 解析来自网站、文档等的信息
- 复杂任务
  - 机器翻译 Machine Translation
  - 语义分析 Semantic Analysis
  - 指代消解 Coreference
  - 问答系统 Question Answering

### 1.3 如何表征词汇

在所有的NLP任务中，第一个也是可以说是最重要的共同点是我们如何将单词表示为任何模型的输入。在这里我们不会讨论早期的自然语言处理工作是将单词视为原子符号 atomic symbols。

为了让大多数的自然语言处理任务能有更好的表现，我们首先需要了解单词之间的相似和不同。有了词向量，我们可以很容易地将其编码到向量本身中。

## 2.词向量

使用词向量编码单词， $$N$$ 维空间足够我们编码语言的所有语义，每一维度都会编码一些我们使用语言传递的信息。

**简单的one-hot向量无法给出单词间的相似性**，例如：

$$
\left(w^{\text {hotel }}\right)^{T} w^{\text {motel }}=\left(w^{\text {hotel }}\right)^{T} w^{\text {cat }}=0
$$

我们需要将维度 $$\mid V \mid $$ 减少至一个低纬度的子空间，来获得稠密的词向量，获得词之间的关系。

## 3.基于SVD降维的词向量

基于词共现矩阵与SVD分解是构建词嵌入(即词向量)的一种方法。

- 我们首先遍历一个很大的数据集和统计词的共现计数矩阵 $$X$$
- 然后对矩阵 $$X$$ 进行SVD分解得到 $$USV^T$$
- 再然后我们使用 $$U$$ 的行来作为字典中所有词的词向量

接下来我们讨论一下矩阵 $$X$$ 的几种选择。

### 3.1 词-文档矩阵

最初的解决方案是基于词-文档共现矩阵完成的。我们猜想相关连的单词在同一个文档中会经常出现：

- 例如，banks，bonds，stocks，moneys 等等，出现在一起的概率会比较高
- 但是，banks，octopus，banana，hockey 不大可能会连续地出现

我们根据这个情况来建立一个Word-Document矩阵， $$X$$ 是按照以下方式构建：遍历数亿的文档和当词 $$i$$ 出现在文档 $$j$$ ，我们对 $$X_{ij}$$ 加一。

这显然是一个很大的矩阵 $$\mathbb{R}^{\mid V \mid \times M}$$ ，它的规模是和文档数量 $$M$$ 成正比关系。因此我们可以尝试更好的方法。

### 3.2 基于滑窗的词共现矩阵

全文档统计是一件非常耗时耗力的事情，我们可以进行调整对一个文本窗内的数据进行统计，计算每个单词在特定大小的窗口中出现的次数，得到共现矩阵 $$X$$ 。

下面为一个简单示例，我们基于滑窗（前后长度为1）对文本进行共现矩阵的构建。

- I enjoy flying.
- I like NLP.
- I like deep learning.

![20220824220857-2022-08-24-22-08-57](https://cdn.jsdelivr.net/gh/ironartisan/picRepo/20220824220857-2022-08-24-22-08-57.png)

> 补充：图中的数字是根据示例中的三句话和滑窗大小来计算出来的。

使用单词共现矩阵：

- 生成维度为 $$ \mid V \mid \times \mid V \mid $$ 的共现矩阵 $$X$$
- 在 $$X$$ 上应用SVD从而得到 $$X=USV^T$$
- 选择 $$U$$ 前 $$k$$ 行得到  $$k$$ 维的词向量
- $$ \frac{\sum_{i=1}^{k} \sigma_{i}}{\sum_{i=1}^{\mid V \mid} \sigma_{i}} $$ 表示第一个 $$k$$ 维包含的方差量

### 3.3 应用SVD对共现矩阵降维

我们对矩阵 $$X$$ 使用SVD，观察奇异值(矩阵 $$S$$ 上对角线上元素)，根据方差百分比截断，留下前 $$k$$ 个元素：

$$\frac{\sum_{i=1}^{k} \sigma_{i}}{\sum_{i=1}^{\mid V \mid} \sigma_{i}}$$

然后取子矩阵 $$U_{1:|V|, 1: k}$$ 作为词嵌入矩阵。这就给出了词汇表中每个词的 $$k$$ 维表示。

![20220824222315-2022-08-24-22-23-15](https://cdn.jsdelivr.net/gh/ironartisan/picRepo/20220824222315-2022-08-24-22-23-15.png)

通过选择前  $$k$$  个奇异向量来降低维度：

![20220824222340-2022-08-24-22-23-41](https://cdn.jsdelivr.net/gh/ironartisan/picRepo/20220824222340-2022-08-24-22-23-41.png)

前面提到的方法给我们提供了足够的词向量来编码语义和句法(part of speech)信息，但也带来了一些问题：
- 矩阵的维度会经常发生改变(经常增加新的单词和语料库的大小会改变)
- 矩阵会非常的稀疏，因为很多词不会共现
- 矩阵维度一般会非常高 $$\approx 10^{6} \times 10^{6}$$
- 需要在 $$X$$ 上加入一些技巧处理来解决词频的极剧的不平衡
  
> 基于SVD的方法的计算复杂度很高，并且很难合并新单词或文档。

对上述讨论中存在的问题存在以下的解决方法：

- 忽略功能词，例如“the”，“he”，“has”等等
- 使用ramp window，即根据文档中单词之间的距离对共现计数进行加权
- 使用皮尔逊相关系数并将负计数设置为 0 ，而不是只使用原始计数
  
> 基于计数的方法可以有效地利用统计量，但下述基于迭代的方式可以在控制复杂度的情况下有效地在大语料库上构建词向量。

## 4.迭代优化算法 - Word2Vec

Word2Vec是一个迭代模型，该模型能够根据文本进行迭代学习，并最终能够对给定上下文的单词的概率对词向量进行编码呈现，而不是计算和存储一些大型数据集(可能是数十亿个句子)的全局信息。

这个想法是设计一个模型，该模型的参数就是词向量。然后根据一个目标函数训练模型，在每次模型的迭代计算误差，基于优化算法调整模型参数（词向量），减小损失函数，从而最终学习到词向量。大家知道在神经网络中对应的思路叫“反向传播”，模型和任务越简单，训练它的速度就越快。

基于迭代的方法一次捕获一个单词的共现情况，而不是像SVD方法那样直接捕获所有的共现计数。

已经很多研究者按照这个思路测试了不同的方法。[Collobert et al., 2011]设计的模型首先将每个单词转换为向量。对每个特定的任务(命名实体识别、词性标注等等)，他们不仅训练模型的参数，同时也训练单词向量，计算出了非常好的词向量的同时取得了很好的性能。

一个非常有效的方法是Word2Vec。Word2Vec是google开源的软件包，包含以下核心内容：
- 两个算法：continuous bag-of-words(CBOW)和skip-gram
  - CBOW是根据中心词周围的上下文单词来预测该词的词向量
  - skip-gram则相反，是根据中心词预测周围上下文的词的概率分布
- 两个训练方法：negative sampling和hierarchical softmax
  - Negative sampling通过抽取负样本来定义目标
  - hierarchical softmax通过使用一个有效的树结构来计算所有词的概率来定义目标

**Word2Vec依赖于语言学中一个非常重要的假设「分布相似性」，即相似的词有相似的上下文。**

### 4.1 语言模型

我们先来了解一下语言模型。从一个例子开始：

`我喜欢漂亮女孩`

一个好的语言模型会给这个句子很高的概率，因为在句法和语义上这是一个完全有效的句子。相似地，句子 `女孩批量的我` 会得到一个很低的概率，因为这是一个无意义的句子。

在数学上，我们可以称为对给定 $$n$$ 个词的序列的概率是：
$$
P\left(w_{1}, w_{2}, \cdots, w_{n}\right)
$$

在一元语言模型方法(Unigram model)中，我们假设单词的出现是完全独立的，从而分解概率：

$$
P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=1}^{n} P\left(w_{i}\right)
$$

严谨一点说，上述假设是不合理的，因为下一个单词是高度依赖于前面的单词序列的。如果使用上述的语言模型，可能会让一个无意义的句子具有很高的概率。所以我们让序列的概率取决于序列中的单词和其旁边的单词的成对概率。我们称之为`bigram`模型：

$$
P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=2}^{n} P\left(w_{i} \mid w_{i-1}\right)
$$

确实，只关心邻近单词还是有点简单，大家考虑连续的 n 个词共现会得到 n-gram。但即使使用 bigram 都可以带来相对 unigram显著的提升。考虑在词-词共现矩阵中，共现窗口为 1 ，我们基本上能得到这样的成对的概率。但是，这又需要计算和存储大量数据集的全局信息。

既然我们已经理解了如何考虑具有概率的单词序列，那么让我们观察一些能够学习这些概率的示例模型。

### 4.2 CBOW连续词袋模型

这一方法是把 {"我","喜欢","漂亮","女孩"} 作为上下文，希望从这些词中能够预测或者生成中心词` 喜欢`。这样的模型我们称之为continuous bag-of-words(CBOW)模型。

CBOW是从上下文中预测中心词的方法，在这个模型中的每个单词，我们希望学习两个向量：
- $$v$$ (输入向量，即上下文词)
- $$u$$ (输出向量，即中心词)

模型输入是one-hot形式的词向量表示。输入的one-hot向量或者上下文我们用 $$x^{(c)}$$ 表示，输出用 $$y^{(c)}$$ 表示。在CBOW模型中，因为我们只有一个输出，因此我们把 $$y $$ 称为是已知中心词的的one-hot向量。

下面我们定义模型的未知参数。

我们创建两个矩阵， $$\mathcal{v} \in \mathbb{R}^{n \times \mid V \mid}$$ 和 $$\mathcal{u} \in \mathbb{R}^{\mid V \mid \times n}$$。其中：

$$n$$ 是嵌入空间的任意维度大小

$$v$$ 是输入词矩阵，使得当其为模型的输入时， $$v$$  的第$$i$$  列是词 $$w_i$$  的 $$n$$ 维嵌入向量，定义这个 $$n \times 1$$ 的向量为 $$v_i$$ 

相似地， $$u$$ 是输出词矩阵。当其为模型的输入时， $$u$$ 的第$$j$$ 行是词 $$w_i$$  的 $$n$$ 维嵌入向量。我们定义  $$u$$ 的这行为  $$u_j$$

注意实际上对每个词 $$w_i$$ 我们需要学习两个词向量(即输入词向量 $$v_i$$ 和输出词向量 $$u_i$$ )。

首先我们对CBOW模型作出以下定义：

$$w_i$$ ：词汇表 $$V$$ 中的单词 $$i$$
$$\mathcal{V} \in \mathbb{R}^{n \times|V|}$$ ：输入词矩阵
$$v_i$$ ： $$v$$ 的第 $$i$$ 列，单词 $$w_i$$ 的输入向量表示
$$\mathcal{u} \in \mathbb{R}^{|V| \times n}$$ ：输出词矩阵
$$u_i$$ ：  $$u$$ 的第 $$i$$ 行，单词 $$w_i$$ 的输出向量表示

分解为以下步骤：
- 我们为大小为 $$m$$ 的输入上下文词汇，生成one-hot词向量$$\left(x^{(c-m)}, \cdots, x^{(c-1)}, x^{(c+1)}, \cdots, x^{(c+m)}\in \mathbb{R}^{\mid V \mid}\right)$$
- 基于上述one-hot输入计算得到嵌入词向量 $$\left(v_{c-m}=\mathcal{V} x^{(c-m)}, v_{c-m+1}=\mathcal{V} x^{(c-m+1)}, \cdots, v_{c+m}=\mathcal{V} x^{(c+m)} \in \mathbb{R}^{n}\right)$$
- 对上述的词向量求平均值 $$\hat{v}=\frac{v_{c-m}+v_{c-m+1}+\cdots+v_{c+m}}{2 m} \in \mathbb{R}^{n}$$
- 计算分数向量 $$z=\mathcal{U} \hat{v} \in \mathbb{R}^{\mid V \mid V\mid V \mid}$$ 。相似的词对向量的点积值大，这会令相似的词更为靠近，从而获得更高的分数
- 将分数通过softmax转换为概率 $$\hat{y}=\operatorname{softmax}(z) \in \mathbb{R}^{\mid V \mid}$$
-  我们希望生成的概率 $$\hat{y} \in \mathbb{R}^{|V|}$$ 与实际的概率 $$y \in \mathbb{R}^{\mid V \mid}$$ （实际是one-hot表示）越接近越好（我们后续会构建交叉熵损失函数并对其进行迭代优化）

这里softmax是一个常用的函数。它将一个向量转换为另外一个向量，其中转换后的向量的第 $$i$$ 个元素是 $$\frac{e^{\hat{y}_{i}}}{\sum_{k=1}^{|V|} e^{\hat{y}_{k}}}$$ 。因为该函数是一个指数函数，所以值一定为正数。通过除以 $$\sum_{k=1}^{|V|} e^{\hat{y}_{k}}$$ 来归一化向量(使得 $$\sum_{k=1}^{|V|} \hat{y}_{k}=1$$ )得到概率。

下图是CBOW模型的计算图示：

![20220824235201-2022-08-24-23-52-02](https://cdn.jsdelivr.net/gh/ironartisan/picRepo/20220824235201-2022-08-24-23-52-02.png)

如果有 $$u$$ 和 $$v$$  ，我们知道这个模型是如何工作的，那我们如何更新参数，学习这两个矩阵呢？和所有的机器学习任务相似，我们会构建目标函数，这里我们会使用交叉熵 $$H(\hat{y}, y)$$ 来构建损失函数，它也是信息论里提的度量两个概率分布的距离的方法。

$$H(\hat{y}, y)=-\sum_{j=1}^{|V|} y_{j} \log \left(\hat{y}_{j}\right)$$

上面的公式中，$$y$$ 是one-hot向量。因此上面的损失函数可以简化为：

$$
H(\hat{y}, y)=-y_{j} \log \left(\hat{y}_{j}\right)
$$

$$c$$是正确词的one-hot向量的索引。如果我们精准地预测 $$\hat{y}_{c}=1$$ ，可以计算此时 $$H(\hat{y}, y)=-1 \log (1)=0$$ 。因此，对于完全准确的预测，我们不会面临任何惩罚或者损失。

我们考虑一个相反的情况，预测非常差并且标准答案 $$\hat{y}_{c}=0.01$$  。进行相似的计算可以得到损失函数值 $$H(\hat{y}, y)=-1 \log (0.01)=4.605$$ ，这表示目前损失较大，和标准答案差距较大。

从上面的例子可以看出，对于概率分布，交叉熵为我们提供了一个很好的距离度量。因此我们的优化目标函数公式为：

$$
\begin{aligned}
\operatorname{minimize} J &=-\log P\left(w_{c} \mid w_{c-m}, \cdots, w_{c-1}, w_{c+1}, \cdots, w_{c+m}\right) \\
&=-\log P\left(u_{c} \mid \hat{v}\right) \\
&=-\log \frac{\exp \left(u_{c}^{T} \hat{v}\right)}{\sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)} \\
&=-u_{c}^{T} \hat{v}+\log \sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)
\end{aligned}
$$


我们使用SGD（随机梯度下降）来更新所有相关的词向量 $$u_c$$ 和 $$v_j$$ 。

当 $$\hat{y}=y$$ 时， $$\hat{y} \mapsto H(\hat{y}, y)$$ 为最小值。如果我们找到一个 $$H(\hat{y}, y)$$ 使得 $$H(\hat{y}, y)$$ 接近最小值，那么 $$\hat{y} \approx y$$ 。这意味着我们的模型非常善于根据上下文预测中心词！

为了学习向量(矩阵 $$U$$ 和 $$V$$ )，CBOW定义了一个损失函数，衡量它在预测中心词方面的表现。然后，我们通过更新矩阵 $$U$$ 和 $$V$$ 随机梯度下降来优化损失函数。

SGD对一个窗口计算梯度和更新参数：

$$
\begin{aligned}
&\mathcal{U}_{\text {new }} \leftarrow \mathcal{U}_{\text {old }}-\alpha \nabla_{\mathcal{U}} J \\
&\mathcal{V}_{\text {old }} \leftarrow \mathcal{V}_{\text {old }}-\alpha \nabla_{\mathcal{V}} J
\end{aligned}
$$

### 4.3 Skip-Gram模型

Skip-Gram模型与CBOW大体相同，但模型对于输入输出 $$x$$ 和 $$y$$ 进行了交换，即CBOW中的  $$x$$ 现在是  $$y$$  ，  $$y$$  现在是  $$x$$ 。输入的one-hot向量(中心词)我们表示为  $$x$$  ，输出向量为 $$y^{(j)}$$ 。我们定义的  $$U$$  和  $$V$$  是和CBOW一样的。

**Skip-Gram模型：在给定中心词的情况下预测周围的上下文词。**

首先我们定义一些符号标记

$$w_i$$ ：词汇表 $$V$$ 中的单词 $$i$$
$$\mathcal{V} \in \mathbb{R}^{n \times\mid V \mid}$$ ：输入词矩阵
$$v_i$$ ： $$v$$ 的第 $$i$$ 列，单词 $$w_i$$ 的输入向量表示
$$\mathcal{u} \in \mathbb{R}^{\mid V \mid \times n}$$ ：输出词矩阵
$$u_i$$ ：  $$u$$ 的第 $$i$$ 行，单词 $$w_i$$ 的输出向量表示

Skip-Gram的工作方式可以拆解为以下步骤：
- 生成中心词的one-hot向量 $$x \in \mathbb{R}^{|V|}$$
- 我们对中心词计算得到词嵌入向量 $$v_{c}=\mathcal{V} x \in \mathbb{R}^{|V|}$$
- 生成分数向量$$z=\mathcal{U} v_{c}$$
- 将分数向量转化为概率$$\hat{y}=\operatorname{softmax}(z)$$,注意$$\hat{y}_{c-m}, \cdots, \hat{y}_{c-1}, \hat{y}_{c+1}, \cdots, \hat{y}_{c+m}$$是每个上下文词出现的概率
- 我们希望我们生成的概率向量匹配真实概率$$y^{(c-m)}, \cdots, y^{(c-1)}, y^{(c+1)}, \cdots, y^{(c+m)}$$,one-hot向量是实际的输出

和CBOW模型一样，我们需要生成一个目标函数来评估这个模型。与CBOW模型的一个主要的不同是我们引用了一个朴素的贝叶斯假设来拆分概率。这是一个很强(朴素)的条件独立假设。换而言之，给定中心词，所有输出的词是完全独立的(即公式1至2行)

$$
\begin{aligned}
\operatorname{minimize} J &=-\log P\left(w_{c-m}, \cdots, w_{c-1}, w_{c+1}, \cdots, w_{c+m} \mid w_{c}\right) \\
&=-\log \prod_{j=0, j \neq m}^{2 m} P\left(w_{c-m+j} \mid w_{c}\right) \\
&=-\log \prod_{j=0, j \neq m}^{2 m} P\left(u_{c-m+j} \mid v_{c}\right) \\
&=-\log \prod_{j=0, j \neq m}^{2 m} \frac{\exp \left(u_{c-m+j}^{T} v_{c}\right)}{\sum_{k=1}^{|V|} \exp \left(u_{k}^{T} v_{c}\right)} \\
&=-\sum_{j=0, j \neq m}^{2 m} u_{c-m+j}^{T} v_{c}+2 m \log \sum_{k=1}^{|V|} \exp \left(u_{k}^{T} v_{c}\right)
\end{aligned}
$$

通过这个目标函数（损失函数），我们可以计算出与未知参数相关的梯度，并且在每次迭代中通过SGD来更新它们。

注意：$$J=-\sum_{j=0, j \neq m}^{2 m} \log P\left(u_{c-m+j} \mid v_{c}\right)=\sum_{j=0, j \neq m}^{2 m} H\left(\hat{y}, y_{c-m+j}\right)$$

其中 $$ H\left(\hat{y}, y_{c-m+j}\right)$$ 是向量 $$\hat{y}$$ 的概率和one-hot向量 $$y_{c-m+j}$$ 之间的交叉熵。

只有一个概率向量 $$\hat{y}$$ 是被计算的。Skip-Gram对每个上下文单词一视同仁：该模型计算每个单词在上下文中出现的概率，而与它到中心单词的距离无关。

Skip-Gram模型的计算图示：

![20220825004220-2022-08-25-00-42-21](https://cdn.jsdelivr.net/gh/ironartisan/picRepo/20220825004220-2022-08-25-00-42-21.png)

### 4.4 负例采样

我们再回到需要优化的目标函数上，我们发现在词表很大的情况下，对 $${|V|}$$ 的求和计算量是非常大的。任何的更新或者对目标函数的评估都要花费 $$O(|V|)$$ 的时间复杂度。一个简单的想法是不去直接计算，而是去求近似值。

**因为softmax标准化要对对所有分数求和，CBOW和Skip Gram的损失函数J计算起来很昂贵！**

在每一个训练的时间步，我们不去遍历整个词汇表，而仅仅是抽取一些负样例。我们对噪声分布 $$P_{n}(w)$$ “抽样”，这个概率是和词频的排序相匹配的。Mikolov在论文《Distributed Representations of Words and Phrases and their Compositionality》中提出了负采样。虽然负采样是基于Skip-Gram模型，但实际上是对一个不同的目标函数进行优化。

考虑一组词对 $$(w,c)$$ ，这组词对是训练集中出现过的中心词和上下文词吗？我们通过 $$P(D=1 \mid w,c)$$ 表示 $(w,c)$$ 在语料库出现过， $$P(D=0 \mid w,c)$$ 表示 $(w,c)$$ 在语料库中没有出现过。这是一个二分类问题，我们基于sigmoid函数建模：

$$P(D=1 \mid w, c, \theta)=\sigma\left(v_{c}^{T} v_{w}\right)=\frac{1}{1+e^{\left(-v_{c}^{T} v_{w}\right)}}$$

sigmoid函数是softmax的二分类版本，可用于建立概率模型：$$\sigma(x)=\frac{1}{1+e^{-x}}$$

![20220825094811-2022-08-25-09-48-11](https://cdn.jsdelivr.net/gh/ironartisan/picRepo/20220825094811-2022-08-25-09-48-11.png)

现在，我们建立一个新的目标函数，如果中心词和上下文词确实在语料库中，就最大化概率 $$P(D=1 \mid w,c)$$ ，如果中心词和上下文词确实不在语料库中，就最大化概率 $$P(D=0 \mid w,c)$$

我们对这两个概率采用一个简单的极大似然估计的方法(这里我们把 $$\theta$$ 作为模型的参数，在我们的例子是 $$v$$ 和 $$u$$ )

$$
\begin{aligned}
\theta &=\underset{\theta}{\operatorname{argmax}} \prod_{(w, c) \in D} P(D=1 \mid w, c, \theta) \prod_{(w, c) \in \widetilde{D}} P(D=0 \mid w, c, \theta) \\
&=\underset{\theta}{\operatorname{argmax}} \prod_{(w, c) \in D} P(D=1 \mid w, c, \theta) \prod_{(w, c) \in \widetilde{D}}(1-P(D=1 \mid w, c, \theta)) \\
&=\underset{\theta}{\operatorname{argmax}} \sum_{(w, c) \in D} \log P(D=1 \mid w, c, \theta)+\sum_{(w, c) \in \widetilde{D}} \log (1-P(D=1 \mid w, c, \theta)) \\
&=\underset{\theta}{\arg \max _{(w, c) \in D}} \sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}+\sum_{(w, c) \in \widetilde{D}} \log \left(1-\frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}\right) \\
&=\underset{\theta}{\arg \max _{\theta}} \sum_{(w,)} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}+\sum_{(w, c) \in \widetilde{D}} \log \left(\frac{1}{1+\exp \left(u_{w}^{T} v_{c}\right)}\right)
\end{aligned}
$$

这里最大化似然函数等同于最小化负对数似然：

$$J=-\sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}-\sum_{(w, c) \in \widetilde{D}} \log \left(\frac{1}{1+\exp \left(u_{w}^{T} v_{c}\right)}\right)$$

注意 $$\widetilde{D}$$ 是“假的”或者“负的”语料。我们可以从语料库中随机抽样词汇构建负样例 $$\widetilde{D}$$ 。

对于Skip-Gram模型，我们对给定中心词 $$c$$ 来观察的上下文单词 $$c-m+j$$ 的新目标函数为

$$
-\log \sigma\left(u_{c-m+j}^{T} \cdot v_{c}\right)-\sum_{k=1}^{K} \log \sigma\left(-\tilde{u}_{k}^{T} \cdot v_{c}\right)
$$

对CBOW模型，我们对给定上下文向量 $$\hat{v}=\frac{v_{c-m}+v_{c-m+1}+\cdots+v_{c+m}}{2 m}$$ 来观察中心词 $$u_c$$ 的新的目标函数为：

$$
-\log \sigma\left(u_{c}^{T} \cdot \hat{v}\right)-\sum_{k=1}^{K} \log \sigma\left(-\tilde{u}_{k}^{T} \cdot \hat{v}\right)
$$

在上面的公式中， $$\left\{\tilde{u}_{k} \mid k=1, \cdots, K\right\}$$ 是从 $$P_{n}(w)$$ 中抽样的词汇。关于计算选择某个词作为负样本的概率，可以使用随机选择。但论文作者给出了如下效果更好的公式：

$$
p\left(w_{i}\right)=\frac{f\left(w_{i}\right)^{\frac{3}{4}}}{\sum_{j=0}^{m} f\left(w_{j}\right)^{\frac{3}{4}}}
$$
公式中， $$f(w_i)$$ 代表语料库中单词 $$w_i$$ 出现的频率。上述公式更加平滑，能够增加低频词的选取可能。

### 4.5 层次化Softmax

Mikolov 在论文《Distributed Representations of Words and Phrases and their Compositionality》中提出了 hierarchical softmax（层次化softmax），相比普通的softmax这是一种更有效的替代方法。在实际中，hierarchical softmax 对低频词往往表现得更好，负采样对高频词和较低维度向量表现得更好。

Hierarchical softmax 使用一个二叉树来表示词表中的所有词。树中的每个叶结点都是一个单词，而且只有一条路径从根结点到叶结点。在这个模型中，没有词的输出表示。相反，图的每个节点(根节点和叶结点除外)与模型要学习的向量相关联。单词作为输出单词的概率定义为从根随机游走到单词所对应的叶的概率。计算成本变为 $$O(log(\mid V \mid))$$ 而不是 $$O(\mid V \mid)$$ 。

在这个模型中，给定一个向量  $$w_i$$ 的下的单词  $$w$$ 的概率  $$p(w \mid w_i)$$ ，等于从根结点开始到对应w的叶结点结束的随机漫步概率。这个方法最大的优势是计算概率的时间复杂度仅仅是 $$O(log(\mid V \mid))$$ ，对应着路径的长度。下图是 Hierarchical softmax 的二叉树示意图：

![20220825141655-2022-08-25-14-16-55](https://cdn.jsdelivr.net/gh/ironartisan/picRepo/20220825141655-2022-08-25-14-16-55.png)

令 $$L(w)$$ 为从根结点到叶结点 $$w$$ 的路径中节点数目。例如，上图中的 $$L(w)$$ 为3。我们定义 $$n(w,i)$$ 为与向量 $$v_n(w, i)$$ 相关的路径上第 $$i$$ 个结点。因此  $$n(w,1)$$ 是根结点，而 $$n(w, L(w))$$ 是 $$w$$ 的父节点。现在对每个内部节点 $$n$$ ，我们任意选取一个它的子节点，定义为 $$ch(n)$$ (一般是左节点)。然后，我们可以计算概率为

$$
\begin{aligned}
p\left(w \mid w_{i}\right)=\prod_{j=1}^{L(w)-1} \sigma([n(w, j+1)&\left.=\operatorname{ch}(n(w, j))] \cdot v_{n(w, j)}^{T} v_{w_{i}}\right) \\
& \text { 其中 }[x]= \begin{cases}1 & \text { if } x \text { is true } \\
-1 & \text { otherwise }\end{cases}
\end{aligned}
$$

这个公式看起来非常复杂，我们来展开讲解一下。
- 首先，我们将根据从根节点 $(n(w, 1))$ 到叶节点 $(w)$ 的路径的形状 (左右分支) 来计算相乘的项。如果 我们假设 $\operatorname{ch}(n)$ 一直都是 $n$ 的左节点，然后当路径往左时 $[n(w, j+1)=\operatorname{ch}(n(w, j))]$ 的值返 回 1 ，往右则返回0。
- 此外， $[n(w, j+1)=\operatorname{ch}(n(w, j))]$ 提供了归一化的作用。在节点 $n$ 处，如果我们将去往左和右 节点的概率相加，对于 $v_{n}^{T} v_{w_{i}}$ 的任何值则可以检查， $\sigma\left(v_{n}^{T} v_{w_{i}}\right)+\sigma\left(-v_{n}^{T} v_{w_{i}}\right)=1$ 。归一化也 保证了 $\sum_{w=1}^{|V|} P\left(w \mid w_{i}\right)=1$ ，和普通的softmax是一样的。

最后我们计算点积来比较输入向量 $v_{w_{i}}$ 对每个内部节点向量 $v_{n(w, j)}^{T}$ 的相似度。下面我们给出一个例子。 以上图中的 $w_{2}$ 为例，从根节点要经过两次左边的边和一次右边的边才到达 $w_{2}$ ，因此

$$
\begin{aligned}
p\left(w_{2} \mid w_{i}\right) &=p\left(n\left(w_{2}, 1\right), \text { left }\right) \cdot p\left(n\left(w_{2}, 2\right), \text { left }\right) \cdot p\left(n\left(w_{2}, 3\right), \text { right }\right) \\
&=\sigma\left(v_{n\left(w_{2}, 1\right)}^{T} v_{w_{i}}\right) \cdot \sigma\left(v_{n\left(w_{2}, 2\right)}^{T} v_{w_{i}}\right) \cdot \sigma\left(-v_{n\left(w_{2}, 3\right)}^{T} v_{w_{i}}\right)
\end{aligned}
$$
我们训练模型的目标是最小化负的对数似然 $-\log P\left(w \mid w_{i}\right)$ 。不是更新每个词的输出向量，而是更新更新 二叉树中从根结点到叶结点的路径上的节点的向量。

该方法的速度由构建二叉树的方式确定，并将词分配给叶节点。Mikolov在论文《Distributed Representations of Words and Phrases and their Compositionality》中使用的是哈夫曼树，在树中分配高频词到较短的路径。

## 参考链接

- <https://web.stanford.edu/class/cs224n/>