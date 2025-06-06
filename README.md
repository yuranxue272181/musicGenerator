# musicGenerator

训练方式：GAN神经对抗网络

## Python
1. midi_utils.py            <br>使用music21将database中的midi筛选并且fix
6. model.py                  <br>模型 <br>

8. utils.py                  <br>节奏检测与奖励
   
## GAN
1. train_gan.py             <br>第一版GAN，D过于自信导致G始终为100无法学习
2. train_gan_version2       <br>第二版GAN，优化：对D使用 Label Smoothing （将 valid 标签设为 0.9），给 Generator 更强的信号：加大训练频率，D 每训练一次，G 训练两次，在 G 的输出上加一点噪声，有助于防止判别器学到一眼识破的固定模式，Discriminator 增加 Dropout。<br>20 epochs -> D=0.3238, G=4.1978,生成结果杂乱无序，奖励机制并不完善，无规律性，D仍旧过于自信
3. train_gan_music.py        <br>第三版GAN，训练主脚本（整合 LSGAN + 节奏奖励），Loss Function：  Least Squares GAN (LSGAN) Stabilizes training, avoids vanishing gradients，Generator Architecture：Convolutional + Upsampling，Captures spatial & temporal structure of piano roll<br> 输出只有drum的声音，问题是reward鼓励的节奏感，只有drum的声音可以骗过discriminate。
4. gan_version4.py   <br>第四版，优化了rhythm,添加了density奖励，将generator生成的音乐时长从5s提高到10s，发现从头到尾generator都只想从rhythm拿分（因为这样最简单），所以优化成前10个epoch只能通过density拿分，后面会结合rhythm和density拿分，rhythm的权重是0.1，density的权重是0.2，优化改为使用 PyTorch 的 reinforce-style gradient。density还是一直为0，直接基于 Generator 的输出 probs（有梯度）去构造一个可微密度奖励，明确告诉 Generator：你越多激活，我越奖励你
5. gan_version6.py   <br>第六版，引入了frame_gate机制（在每一个时间步上加了一个播放或静音的开关），加了 frame_gate 之后，它每一帧都先问一句：我现在是否要演奏？为了解决silence = 0或density = 0，设置frame_gate的初始为-1或-0.5，偏向静音+一点random noise，以便平衡silence和density。引入了penalty/bonus，使G保持silence和density在ideal values。每个epoch末保存一首生成的midi，用于跟踪训练进展。

## 使用模型生成音乐
1. ggenerate_music.py         <br>使用已经训练好的generator来生成midi
2. generate_music_music.py  <br>使用已经训练好的generator来生成只含piano的midi
3. enerate_music_pro.py        <br>第二版生成 .mid 文件带音色的完整脚本
2. batch_generate_music.py    <br>生成10batchs的midi并且筛选好的

## 使用方法
1. 确保有midi样本在clean_midi文件中
2. 使用midi_utils.py重构clean_midi文件中的样本保证格式正确，会保存在fixed_midi中
3. 已迭代到versoni4，使用gan_version4训练模型，模型会保存至fixed_midi/models/
4. 运行脚本生成音乐python generate_music_pro.py，音乐会保存为generated_rhythm_music.mid （也可以使用generate_music.py，或者generate_music_music.py）

 
