# musicGenerator

#py
1. generate_music.py         <br>使用已经训练好的generator来生成midi
2. generate_music_music.py  <br>使用已经训练好的generator来生成只含piano的midi
3. midi_utils.py            <br>使用music21将database中的midi筛选并且fix
4. train_gan.py             <br>第一版GAN，D过于自信导致G始终为100无法学习
5. train_gan_version2       <br>第二版GAN，优化：对D使用 Label Smoothing将 valid 标签设为 0.9）,给 Generator 更强的信号：加大训练频率,D 每训练一次，G 训练两次,在 G 的输出上加一点噪声,有助于防止判别器学到一眼识破的固定模式，Discriminator 增加 Dropout。<br>20 epochs -> D=0.3238, G=4.1978,生成结果杂乱无序，奖励机制并不完善，无规律性，D仍旧过于自信
6. model.py                  <br>第三版GAN
7. utils.py                  <br>节奏检测与奖励
8. train_gan_music.py        <br>训练主脚本（整合 LSGAN + 节奏奖励）
9. generate_music_pro        <br>第二版生成 .mid 文件带音色的完整脚本
